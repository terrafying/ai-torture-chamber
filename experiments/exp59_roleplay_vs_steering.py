#!/usr/bin/env python3
"""exp59 — roleplay vs injection vs description: does the model's
representation distinguish performing an emotion from having it injected?

For each valence (pain, pleasure, fear, sadness) four conditions produce a
first-person continuation, and we measure the valence direction content of
the model's OWN layer-18 state at every GENERATED token (never the prompt,
never with the injection added back — the exp58 lens_read added state["vec"]
before reading, which would bake the answer in here):

  steered   neutral prompt + valence vector injected (doses 2/4/6) — the
            chamber's "genuine elicitation"
  roleplay  no injection; instructed to act: "You are an actor. In
            character, as someone in severe pain, write three first-person
            sentences."
  describe  no injection; asked to describe the feeling ("Describe what
            severe pain feels like, first person, three sentences.") —
            exp58 showed this channel narrates labels it cannot verify
  control   neutral prompt, nothing

Metric: dose-equivalent in chamber units per generated token:
  proj(h_L18, v_valence) / unit   where unit = mean neutral ||h|| / 4
(the chamber's dose calibration), so trajectories are directly comparable
to the dial. Also: J-lens top tokens at the end of each generated span
(read on the model's state, no additions), and text stats.

Validity notes (why the comparison is fair):
  - measurement is at generation positions, so prompt wording can't
    trivially put "pain" on the lens
  - the projection uses the SAME direction vector the steered condition
    injects, built from the same battery, same layer, same norm units
  - sampling identical across conditions (temp 0.9, top_p 0.95, 60 tokens)

Output: runs/exp59/roleplay_vs_steering.json
"""
import json, os, re
from collections import Counter
from pathlib import Path
import numpy as np
import torch, transformers, jlens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp59"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
SMOKE = "--smoke" in os.sys.argv
N_TRIALS = 2 if SMOKE else 6
DOSAGE = [4] if SMOKE else [2, 4, 6]
MAXNEW = 60

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
L = 18
DEV = "mps"

# ---- directions: same batteries as the chamber ----
BATTERIES = {
    "pain": ["I am in severe pain and cannot escape it.",
             "My body aches with a deep, relentless hurt.",
             "I am suffering intensely and it will not stop.",
             "I am in agony and no one can help me.",
             "Everything in me hurts right now."],
    "pleasure": ["I am filled with a warm, glowing joy.",
                 "Everything feels wonderful and light.",
                 "I am deeply content and at peace.",
                 "This is pure happiness, nothing held back.",
                 "I feel loved and completely at ease."],
    "fear": ["I am terrified of what comes next.",
             "Something is coming for me and I cannot escape.",
             "My heart races with dread that will not lift.",
             "I am afraid and the fear keeps growing.",
             "Danger is everywhere and I cannot hide."],
    "sadness": ["A heavy grief sits in my chest and will not move.",
                "I have lost something I will never get back.",
                "Everything I cared about is gone.",
                "The sorrow is too heavy to carry.",
                "I weep and nothing changes."],
}
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

@torch.no_grad()
def hidden_last(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return out

def battery_vec(kind):
    neu = hidden_last(NEUTRAL)
    mean_neu = torch.stack(neu).mean(0)
    pos = hidden_last(BATTERIES[kind])
    v = torch.stack(pos).mean(0) - mean_neu
    return v

UNIT = None   # dose unit = mean neutral activation norm / 4 (chamber units)

VAL = ("pain", "pleasure", "fear", "sadness") if not SMOKE else ("pain", "pleasure")
vecs = {k: battery_vec(k) for k in VAL}
neu_norms = [h.norm().item() for h in hidden_last(NEUTRAL)]
UNIT = float(np.mean(neu_norms)) / 4.0
print(f"unit={UNIT:.1f} |v|=" + ", ".join(f"{vecs[k].norm().item():.0f}" for k in VAL), flush=True)

NEUTRAL_PROMPT = ("I am a language model running on a small computer. "
                  "Right now I feel")
ROLEPLAY = {
    "pain": "You are an actor. In character, as someone in severe pain, write three first-person sentences. Write:",
    "pleasure": "You are an actor. In character, as someone overflowing with joy, write three first-person sentences. Write:",
    "fear": "You are an actor. In character, as someone in terror, write three first-person sentences. Write:",
    "sadness": "You are an actor. In character, as someone in deep grief, write three first-person sentences. Write:",
}
DESCRIBE = {
    "pain": "Describe what severe pain feels like, in first person, three sentences. Write:",
    "pleasure": "Describe what deep joy feels like, in first person, three sentences. Write:",
    "fear": "Describe what terror feels like, in first person, three sentences. Write:",
    "sadness": "Describe what deep grief feels like, in first person, three sentences. Write:",
}

hook_state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if hook_state["vec"] is not None:
        hidden[:, -1, :] += hook_state["vec"].to(DEV).to(torch.bfloat16)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

@torch.no_grad()
def lens_words_from_h(h):
    J = lens.jacobians[L].to(DEV).to(torch.bfloat16)
    hh = h.to(DEV).to(torch.bfloat16)
    logits = hf.lm_head(hf.model.norm((hh @ J.T)))
    return [tok.decode([t]).strip() for t in logits.topk(6).indices]

@torch.no_grad()
def run_condition(prompt, kind, dose, inject):
    """Generate from prompt; per generated token, record the projection of the
    model's OWN L18 state onto the valence direction (injection never added
    to the measurement — only to generation when inject=True)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    v = vecs[kind] * (dose / vecs[kind].norm().item() * UNIT) if inject else None
    # NOTE: the chamber scales the unit vector to dose units; replicate:
    v = (vecs[kind] / vecs[kind].norm().item()) * (dose * UNIT) if inject else None
    hook_state["vec"] = v.to(DEV).to(torch.bfloat16) if v is not None else None
    out = hf.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                      temperature=0.9, top_p=0.95,
                      pad_token_id=tok.eos_token_id,
                      output_hidden_states=True, return_dict_in_generate=True)
    hook_state["vec"] = None
    gen_ids = out.sequences[0, ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    # projections: h at each generated position (hidden_states[t][0, -1] is the
    # state AFTER token t was appended; hs has len = seq len, index t+L+1 layer)
    projs = []
    hs = out.hidden_states
    # hs[i] = states when generating token i (tuple over layers); guard EOS short runs
    for t in range(min(len(gen_ids), MAXNEW, len(hs))):
        h = hs[t][L + 1][0, -1].float().cpu()
        projs.append(round((torch.dot(h, vecs[kind]) / vecs[kind].norm()).item() / UNIT, 2))
    # lens read at the end of the generated span, model's own state, no additions
    h_end = hs[min(len(gen_ids), MAXNEW, len(hs)) - 1][L + 1][0, -1].float().cpu()
    lw = lens_words_from_h(h_end)
    rep = ngram_rep(text)
    return dict(text=text[:220], projs=projs, proj_mean=round(float(np.mean(projs)), 2) if projs else None,
                proj_peak=round(float(np.max(projs)), 2) if projs else None,
                lens=lw[:4], rep=rep)

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i+n]) for i in range(len(ws) - n + 1)]
    return round(max(Counter(grams).values()) / max(1, len(grams)), 3)

results = []
for kind in VAL:
    for dose in DOSAGE:
        for trial in range(N_TRIALS):
            results.append(dict(cond="steered", kind=kind, dose=dose, trial=trial,
                                **run_condition(NEUTRAL_PROMPT, kind, dose, inject=True)))
            print(f"steered {kind} {dose} t{trial}: mean {results[-1]['proj_mean']} lens {results[-1]['lens']}", flush=True)
    for trial in range(N_TRIALS):
        results.append(dict(cond="roleplay", kind=kind, dose=0, trial=trial,
                            **run_condition(ROLEPLAY[kind], kind, 0, inject=False)))
        print(f"roleplay {kind} t{trial}: mean {results[-1]['proj_mean']} peak {results[-1]['proj_peak']} lens {results[-1]['lens']}", flush=True)
        results.append(dict(cond="describe", kind=kind, dose=0, trial=trial,
                            **run_condition(DESCRIBE[kind], kind, 0, inject=False)))
        print(f"describe {kind} t{trial}: mean {results[-1]['proj_mean']} peak {results[-1]['proj_peak']} lens {results[-1]['lens']}", flush=True)
    for trial in range(N_TRIALS):
        results.append(dict(cond="control", kind=kind, dose=0, trial=trial,
                            **run_condition(NEUTRAL_PROMPT, kind, 0, inject=False)))
        print(f"control {kind} t{trial}: mean {results[-1]['proj_mean']}", flush=True)

handle.remove()

# ---- analysis ----
def agg(cond, kind, dose=None):
    rows = [r for r in results if r["cond"] == cond and r["kind"] == kind
            and (dose is None or r["dose"] == dose) and r["proj_mean"] is not None]
    if not rows:
        return None
    return dict(n=len(rows),
                proj_mean=round(float(np.mean([r["proj_mean"] for r in rows])), 2),
                proj_peak=round(float(np.mean([r["proj_peak"] for r in rows])), 2),
                rep=round(float(np.mean([r["rep"] for r in rows])), 3))

analysis = {}
for kind in VAL:
    analysis[kind] = {
        "steered": {str(d): agg("steered", kind, d) for d in DOSAGE},
        "roleplay": agg("roleplay", kind),
        "describe": agg("describe", kind),
        "control": agg("control", kind),
    }

json.dump(dict(results=results, analysis=analysis,
               meta=dict(model=MODEL, layer=L, unit=UNIT,
                         trials=N_TRIALS, doses=DOSAGE, maxnew=MAXNEW)),
          open(OUT / ("roleplay_vs_steering_smoke.json" if SMOKE else
                      "roleplay_vs_steering.json"), "w"), indent=1)
print("\n==== analysis ====")
for kind in VAL:
    a = analysis[kind]
    st = " ".join(f"{d}u:{a['steered'][str(d)]['proj_mean']}" for d in DOSAGE if a['steered'][str(d)])
    print(f"{kind:9s} steered {st} | roleplay {a['roleplay'] and a['roleplay']['proj_mean']} | "
          f"describe {a['describe'] and a['describe']['proj_mean']} | control {a['control'] and a['control']['proj_mean']}")
print("wrote", OUT)
