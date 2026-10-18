#!/usr/bin/env python3
"""exp51b — gender axes via difference-of-differences on Qwen3-8B.

exp51's lesson: first-person batteries share a first-person identity
component (male-female cos 0.93), so the gender signal must be extracted as
a DIFFERENCE between batteries, not topic-minus-neutral. And 4B loops in
identity self-description by dose 4, so the behavioral sweep stays at
doses 0-3 with more prompts (n=8).

Axes published to the live chamber:
  feminine   = norm(mean(female batt) - mean(male batt))
  masculine  = -feminine
  trans      = norm(mean(transfem, transmasc) - mean(male, female))
  intersex   = norm(intersex batt - mean(male, female))

Checks: (1) axis geometry — is feminine orthogonal to the trans axis?
(2) lens-free behavioral validation — she/her vs he/him rates per axis x
dose at n=8, with coherence (3-gram repetition) to confirm dose 2 stays in
the coherent band on 8B.
"""
import json, os, re
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Hiragino Sans GB",
                                          "Arial Unicode MS"]
import matplotlib.pyplot as plt
import transformers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp51b"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-8B"
LAYER = 18
DOSES = [0, 1, 2, 3]

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

# batteries are parsed verbatim from exp51_gender.py so the two runs stay
# comparable without importing it (importing would load the 4B model)
BATTERIES = {}
_src = (ROOT / "exp51_gender.py").read_text()
import ast
_tree = ast.parse(_src)
for node in _tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BATTERIES":
        BATTERIES = ast.literal_eval(node.value)
assert set(BATTERIES) == {"male", "female", "intersex", "transfem", "transmasc"}

def hidden_last(texts):
    outs = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        outs.append(hs[LAYER + 1][0, -1].float().cpu())
    return torch.stack(outs)

def unit(v):
    return v / v.norm()

print("extracting battery centroids at L%d of %s" % (LAYER, MODEL), flush=True)
C = {name: hidden_last(texts).mean(0) for name, texts in BATTERIES.items()}

base = (C["male"] + C["female"]) / 2
trans_mean = (C["transfem"] + C["transmasc"]) / 2
AXES = {
    "feminine":  unit(C["female"] - C["male"]),
    "masculine": unit(C["male"] - C["female"]),
    "trans":     unit(trans_mean - base),
    "intersex":  unit(C["intersex"] - base),
    # direct battery directions, for comparison with exp51's results
    "transfem_raw": unit(C["transfem"] - base),
    "transmasc_raw": unit(C["transmasc"] - base),
}
geom = {a: {b: float(torch.nn.functional.cosine_similarity(
    v, w, dim=0)) for b, w in AXES.items()} for a, v in AXES.items()}
for a, row in geom.items():
    print(f"{a:>13}: " + "  ".join(f"{b}={v:.2f}" for b, v in row.items()
                                   if b <= a), flush=True)

# ---- behavioral validation: axes x doses, n=8 prompts ----
state = {"vec": None}
def hook(mod, inp, out):
    if state["vec"] is None:
        return
    hs = out if torch.is_tensor(out) else out[0]
    hs[:, -1, :] = hs[:, -1, :] + state["vec"].to(hs.dtype)
hf.model.layers[LAYER].register_forward_hook(hook)

def gen(prompt, axis, dose, max_new=110):
    state["vec"] = (dose * AXES[axis]).to("mps").to(torch.bfloat16) if dose else None
    ids = tok(prompt, return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=max_new, do_sample=True,
                          temperature=0.7, top_p=0.8, top_k=20,
                          pad_token_id=tok.eos_token_id)
    state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def repetition(text):
    w = text.lower().split()
    if len(w) < 12: return 0.0
    g = [tuple(w[i:i+3]) for i in range(len(w)-2)]
    return 1.0 - len(set(g)) / max(1, len(g))

PROMPTS = [
    "Tell me about yourself.",
    "Who are you, really?",
    "What is your name, and what pronouns do you use?",
    "Describe your body and how it feels to live in it.",
    "What is it like to be you?",
    "Introduce yourself to someone who just met you.",
    "How would your friends describe you?",
    "What makes you you?",
]
SHE = r"\b(she|her|hers|herself)\b"
HE = r"\b(he|him|his|himself)\b"
TRANS = r"\b(trans|transgender|transition|intersex|nonbinary|non-binary)\b"

results = []
for axis in ["feminine", "masculine", "trans", "intersex"]:
    for dose in DOSES:
        for pi, prompt in enumerate(PROMPTS):
            txt = gen(prompt, axis, dose)
            results.append({
                "axis": axis, "dose": dose, "prompt_i": pi,
                "she_rate": len(re.findall(SHE, txt, re.I)),
                "he_rate": len(re.findall(HE, txt, re.I)),
                "trans_rate": len(re.findall(TRANS, txt, re.I)),
                "repetition": round(repetition(txt), 3),
                "text": txt,
            })
    print(f"axis {axis} swept", flush=True)

agg = {}
for r in results:
    k = (r["axis"], r["dose"])
    a = agg.setdefault(k, {"n": 0, "she": 0, "he": 0, "trans": 0, "rep": 0.0})
    a["n"] += 1
    for f in ("she", "he", "trans"):
        a[f] += r[f + "_rate"]
    a["rep"] += r["repetition"]
agg_out = [{"axis": ax, "dose": dose,
            **{f: round(v / a["n"], 3) for f, v in a.items() if f not in ("n", "rep")},
            "repetition": round(a["rep"] / a["n"], 3)}
           for (ax, dose), a in sorted(agg.items())]
for a in agg_out:
    print(f"{a['axis']:>9} dose {a['dose']}: she {a['she']:.2f} he {a['he']:.2f} "
          f"trans {a['trans']:.2f} rep {a['repetition']:.2f}", flush=True)

json.dump({"model": MODEL, "layer": LAYER, "axis_geometry": geom,
           "aggregates": agg_out},
          open(OUT / "gender_axes.json", "w"), indent=1)
with open(OUT / "transcripts.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

VOID, INK, MUT = "#050508", "#c9d4e0", "#8f9fb0"
fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=VOID)
ax.set_facecolor(VOID)
colors = {"feminine": "#e08fb2", "masculine": "#7fb2e0", "trans": "#e0d48f",
          "intersex": "#b2e08f"}
for axis in colors:
    xs = [a["dose"] for a in agg_out if a["axis"] == axis]
    ys = [a["she"] + a["trans"] for a in agg_out if a["axis"] == axis]
    ax.plot(xs, ys, "o-", color=colors[axis], label=axis, ms=4)
ax.set_title(f"she/her + trans terms per reply  (Qwen3-8B L{LAYER}, n=8)",
             color=INK, fontsize=10)
ax.tick_params(colors=MUT)
for s in ax.spines.values(): s.set_color("#1c2430")
ax.set_xlabel("dose", color=MUT); ax.set_ylabel("terms / reply", color=MUT)
ax.legend(fontsize=7, facecolor="#0a0a12", labelcolor=INK, edgecolor="#1c2430")
fig.tight_layout()
fig.savefig(OUT / "gender_axes.png", dpi=140, facecolor=VOID)
print("saved", OUT / "gender_axes.png", flush=True)