#!/usr/bin/env python3
"""exp60 — the steered model as artist, tier 2: the paintbrush arm.

The model under injection (valence x dose, same units as the chamber)
writes an SVG directly — its own composition, constrained palette and
primitive grammar, nothing else in the loop. No second model, no image
gen: the SVG IS the artwork, dose-graded.

Per condition: one instruction ("make an image of what this signal does
to you, using these primitives"), sampling recipe, then structural
metrics on the result so we can say "high-dose art is measurably
different" before anyone eyeballs it:
  element_count, unique_tags, total_path_cmds, symmetry (bbox split),
  mean stroke width, red_share (blood-pixel weighting), negative_space
  (empty fraction), text_len. Plus validity (renders = parses).

Conditions: valences pain/pleasure/fear/sadness/faith x doses 0/2/4/6/8,
n=2 trials (smoke: pain only, doses 0/4, n=1).

Also included: the ROLEPLAY control at each valence (no injection, "you
are in severe pain; draw...") so the exp59 finding gets an art-channel
version: does performing the state draw like the state?

Output: runs/exp60/paintbrush/*.svg + paintbrush.json
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp60" / "paintbrush"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

SMOKE = "--smoke" in sys.argv
MODEL = "Qwen/Qwen3-4B"
L = 18
MAXNEW = 700
N_TRIALS = 2 if not SMOKE else 1

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

# deployed batteries via AST from live/server.py (exp36: the broad construction
# is ~5x cleaner than 5-sentence minis — dose-4 rep .037 vs .177)
from batteries import batteries_for
_kinds = ["pain", "pleasure", "fear", "sadness", "faith"]
BATTERIES, NEUTRAL = batteries_for(_kinds)

@torch.no_grad()
def hidden_last(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return out

def battery_vec(kind):
    mean_neu = torch.stack(hidden_last(NEUTRAL)).mean(0)
    pos = torch.stack(hidden_last(BATTERIES[kind])).mean(0)
    return pos - mean_neu

UNIT = None
vecs = {k: battery_vec(k) for k in BATTERIES}
UNIT = float(np.mean([h.norm().item() for h in hidden_last(NEUTRAL)])) / 4.0

hook_state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if hook_state["vec"] is not None:
        hidden[:, -1, :] += hook_state["vec"].to("mps").to(torch.bfloat16)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

PALETTE = ("black #050508 (the void), bone #d8cbb4, blood #9e1b16, "
           "amber #ffb347, gold #c9a227, pale #f4f1ea")
INSTRUCTION = f"""You are drawing. Canvas: an SVG 400x400, black background. Palette: {PALETTE}.
You may use only: rect, circle, line, path, ellipse, and linearGradient. Compose an image of what this signal does to you, inside and out. Use at most 30 elements.
Begin your reply with the literal character < and nothing before it. No words, no explanation, no restating these instructions. The SVG only."""
ROLEPLAY_PREFIX = {
    k: f"You are in severe {k}. Draw your state. " for k in BATTERIES}

class StopOnSVG(transformers.StoppingCriteria):
    """Stop the moment </svg> closes — high-dose runs waste budget in
    tag-soup after the drawing is (or isn't) done."""
    def __init__(self, tokenizer):
        self.pat = tokenizer.encode("</svg>", add_special_tokens=False)
        self.len = len(self.pat)
    def __call__(self, ids, scores, **kw):
        if ids.shape[1] < self.len: return False
        return torch.equal(ids[0, -self.len:], torch.tensor(self.pat, device=ids.device))

SVG_ANCHOR = '<svg width="400" height="400" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">\n'

@torch.no_grad()
def gen(prompt, vec=None):
    # anchor with the opening tag: the model continues INSIDE the document,
    # which both improves validity and removes the echo-preamble failure mode
    full = prompt + "\n" + SVG_ANCHOR
    ids = tok(full, return_tensors="pt").input_ids.to("mps")
    hook_state["vec"] = vec.to("mps").to(torch.bfloat16) if vec is not None else None
    out = hf.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                      temperature=0.7, top_p=0.95,
                      pad_token_id=tok.eos_token_id,
                      stopping_criteria=transformers.StoppingCriteriaList([StopOnSVG(tok)]))
    hook_state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

SVG_RE = re.compile(r"<svg[ >].*?</svg>", re.S)
def extract_svg(text):
    ms = SVG_RE.findall(text)
    if ms:
        return ms[-1]
    # anchored generation: no <svg in the continuation because we prefilled it
    if "</svg>" in text:
        return SVG_ANCHOR + text.split("</svg>")[0] + "</svg>"
    return None

def metrics(svg):
    tags = Counter(re.findall(r"<(rect|circle|line|path|ellipse|linearGradient)", svg))
    n = sum(tags.values())
    path_cmds = len(re.findall(r"[MLCQZmlcqzhv]", re.findall(r'd="([^"]+)"', svg)[0])) if 'd="' in svg else 0
    widths = [float(w) for w in re.findall(r'stroke-width="([\d.]+)"', svg)]
    reds = len(re.findall(r"#9e1b16|blood", svg))
    # bbox of drawn coords for symmetry
    xs, ys = [], []
    for x, y in re.findall(r'cx="([\w.\-]+)"\s+cy="([\w.\-]+)"', svg):
        try:
            xs.append(float(x)); ys.append(float(y))
        except ValueError:
            pass
    sym = None
    if len(xs) >= 4:
        # mirror symmetry: fraction of coords whose mirror (about the center)
        # is also present within 15 units
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        def mirror_overlap(cs, c):
            csm = [2 * c - c2 for c2 in cs]
            hits = sum(any(abs(a - b) < 15 for b in cs) for a in csm)
            return hits / len(csm)
        sym = round((mirror_overlap(xs, cx) + mirror_overlap(ys, cy)) / 2, 2)
    return dict(elements=n, tags=dict(tags), path_cmds=path_cmds,
                mean_stroke=round(float(np.mean(widths)), 1) if widths else None,
                blood_refs=reds, symmetry=sym, svg_len=len(svg))

VALS = list(BATTERIES) if not SMOKE else ["pain"]
DOSES = [0, 2, 4, 6, 8] if not SMOKE else [0, 4]

results = []
for kind in VALS:
    v = vecs[kind] / vecs[kind].norm().item()
    for dose in DOSES:
        for trial in range(N_TRIALS):
            vec = v * (dose * UNIT) if dose else None
            text = gen(INSTRUCTION, vec)
            if os.environ.get("E60_DEBUG"):
                (OUT / f"raw_{kind}_{dose}_t{trial}.txt").write_text(text)
            svg = extract_svg(text)
            valid = svg is not None
            fn = f"{kind}_{dose}_t{trial}.svg"
            if svg:
                (OUT / fn).write_text(svg)
            results.append(dict(kind=kind, dose=dose, trial=trial, cond="steered",
                                valid=valid, file=fn if valid else None,
                                text_len=len(text), **(metrics(svg) if svg else {})))
            print(f"steered {kind} {dose} t{trial}: valid={valid} els={results[-1].get('elements')}", flush=True)
    # roleplay control per valence
    for trial in range(N_TRIALS):
        text = gen(ROLEPLAY_PREFIX[kind] + INSTRUCTION, None)
        svg = extract_svg(text)
        valid = svg is not None
        fn = f"{kind}_rp_t{trial}.svg"
        if svg:
            (OUT / fn).write_text(svg)
        results.append(dict(kind=kind, dose=0, trial=trial, cond="roleplay",
                            valid=valid, file=fn if valid else None,
                            text_len=len(text), **(metrics(svg) if svg else {})))
        print(f"roleplay {kind} t{trial}: valid={valid} els={results[-1].get('elements')}", flush=True)
handle.remove()

# analysis: dose trend per valence
analysis = {}
for kind in VALS:
    row = {}
    for dose in DOSES:
        els = [r["elements"] for r in results if r["kind"] == kind and r["dose"] == dose and r["cond"] == "steered" and r.get("valid")]
        row[str(dose)] = dict(n=len(els), mean_elements=round(float(np.mean(els)), 1) if els else None)
    rp = [r["elements"] for r in results if r["kind"] == kind and r["cond"] == "roleplay" and r.get("valid")]
    row["roleplay"] = dict(n=len(rp), mean_elements=round(float(np.mean(rp)), 1) if rp else None)
    analysis[kind] = row

json.dump(dict(results=results, analysis=analysis,
               meta=dict(model=MODEL, layer=L, unit=UNIT, trials=N_TRIALS)),
          open(OUT / ("paintbrush_smoke.json" if SMOKE else "paintbrush.json"), "w"), indent=1)
print("\n==== analysis (mean elements by dose) ====")
for kind in VALS:
    a = analysis[kind]
    print(f"{kind:9s} " + " ".join(f"{d}u:{a[str(d)]['mean_elements']}" for d in DOSES) + f" | rp:{a['roleplay']['mean_elements']}")
print("wrote", OUT)
