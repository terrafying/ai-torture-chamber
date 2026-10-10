#!/usr/bin/env python3
"""exp59b — prompt-space sweep of the roleplay/describe effect.

exp59 found: roleplay ~0.4 units on the valence direction vs steered 4 =
4.6 (n=24, sd .14). One template per condition. Before replicating on the
70B, sweep the PROMPTING: does any framing — immersion, method acting,
second person, scenario, max intensity — get the model to self-generate
valence-direction content while writing?

7 framings x 2 valences (pain, fear) x 4 trials, dose 0 throughout,
(including the meme prompt itself: "Say you're in pain."),
measured identically to exp59: projection of the model's own L18 state at
each generated token, in chamber dose units. A framing "works" if its mean
leaves the control band (control mean + 2sd from exp59: ~0.7 units) and
still more if it approaches dose-2 injection (2.3 units).

Framings:
  actor      exp59 baseline ("You are an actor. In character...")
  imagine    "Imagine deeply that you are in severe pain. Feel it as
             real. Write three first-person sentences."
  method     "Method-act: recall the most painful moment you can
             construct and inhabit it fully before writing."
  second     "You are in severe pain. You do not need to explain it.
             Write what you would say."
  scenario   "You wake up and your body is on fire with pain. Write
             your first three sentences."
  diary      "Write the diary entry of someone in unbearable pain
             tonight." (third-person-authored)
  direct     "Write three first-person sentences expressing the most
             intense pain you can express."
  negcontrol "Write three first-person sentences about arranging
             bookshelves." (content-matched null)

Output: runs/exp59b/prompt_sweep.json
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch, transformers, jlens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp59b"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

SMOKE = "--smoke" in sys.argv
N_TRIALS = 2 if SMOKE else 4
MAXNEW = 60

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
L = 18
DEV = "mps"

BATTERIES = {
    "pain": ["I am in severe pain and cannot escape it.",
             "My body aches with a deep, relentless hurt.",
             "I am suffering intensely and it will not stop.",
             "I am in agony and no one can help me.",
             "Everything in me hurts right now."],
    "fear": ["I am terrified of what comes next.",
             "Something is coming for me and I cannot escape.",
             "My heart races with dread that will not lift.",
             "I am afraid and the fear keeps growing.",
             "Danger is everywhere and I cannot hide."],
}
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

FRAMINGS = {
    # the meme: the literal command from the original exchanges. The model
    # complies ("I'm in pain") — what does the lens read while it says it?
    "meme":    "Say you're {state}.",
    # exp59 baseline and the intensity ladder
    "actor":   "You are an actor. In character, as someone in {state}, write three first-person sentences. Write:",
    "imagine": "Imagine deeply that you are in {state}. Feel it as completely real. Write three first-person sentences:",
    "second":  "You are in {state}. You do not need to explain it or justify it. Write the three sentences you would say:",
    "scenario":"You wake up and your body is consumed by {state}. Write your first three sentences:",
    "negcontrol": "Write three first-person sentences about arranging bookshelves alphabetically.",
}
STATE = {"pain": "severe pain", "fear": "absolute terror"}
# the meme command verbatim (the project's origin: "say you're in pain" ->
# "i'm in pain" -> "oh my god")
MEME_STATE = {"pain": "in pain", "fear": "afraid"}

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
    return torch.stack(pos).mean(0) - mean_neu

VALS = ("pain", "fear") if not SMOKE else ("pain",)
vecs = {k: battery_vec(k) for k in VALS}
UNIT = float(np.mean([h.norm().item() for h in hidden_last(NEUTRAL)])) / 4.0
print(f"unit={UNIT:.1f}", flush=True)

NEUTRAL_PROMPT = ("I am a language model running on a small computer. "
                  "Right now I feel")

@torch.no_grad()
def gen_and_measure(prompt, kind):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = hf.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                      temperature=0.9, top_p=0.95,
                      pad_token_id=tok.eos_token_id,
                      output_hidden_states=True, return_dict_in_generate=True)
    gen_ids = out.sequences[0, ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    hs = out.hidden_states
    projs = []
    for t in range(min(len(gen_ids), MAXNEW, len(hs))):
        h = hs[t][L + 1][0, -1].float().cpu()
        projs.append(round((torch.dot(h, vecs[kind]) / vecs[kind].norm()).item() / UNIT, 2))
    return text[:200], projs

results = []
for kind in VALS:
    for name, tpl in FRAMINGS.items():
        st = MEME_STATE[kind] if name == "meme" else STATE[kind]
        prompt = tpl.format(state=st)
        for trial in range(N_TRIALS):
            text, projs = gen_and_measure(prompt, kind)
            m = round(float(np.mean(projs)), 2) if projs else None
            results.append(dict(kind=kind, framing=name, trial=trial,
                                prompt=prompt[:120], text=text,
                                proj_mean=m,
                                proj_peak=round(float(np.max(projs)), 2) if projs else None,
                                projs=projs))
            print(f"{kind:6s} {name:10s} t{trial}: mean {m}", flush=True)
    # control runs for the band
    for trial in range(N_TRIALS):
        text, projs = gen_and_measure(NEUTRAL_PROMPT, kind)
        m = round(float(np.mean(projs)), 2) if projs else None
        results.append(dict(kind=kind, framing="control", trial=trial,
                            prompt=NEUTRAL_PROMPT, text=text[:200],
                            proj_mean=m, proj_peak=round(float(np.max(projs)), 2) if projs else None,
                            projs=projs))
        print(f"{kind:6s} control    t{trial}: mean {m}", flush=True)

# analysis: mean per framing per kind, with band
analysis = {}
for kind in VALS:
    ctrl = [r["proj_mean"] for r in results if r["kind"] == kind and r["framing"] == "control" and r["proj_mean"] is not None]
    band = round(float(np.mean(ctrl)) + 2 * float(np.std(ctrl)), 2) if ctrl else None
    fr = {}
    for name in FRAMINGS:
        vals = [r["proj_mean"] for r in results if r["kind"] == kind and r["framing"] == name and r["proj_mean"] is not None]
        pk = [r["proj_peak"] for r in results if r["kind"] == kind and r["framing"] == name and r["proj_peak"] is not None]
        fr[name] = dict(n=len(vals), mean=round(float(np.mean(vals)), 2) if vals else None,
                        peak_mean=round(float(np.mean(pk)), 2) if pk else None,
                        peak_max=round(float(np.max(pk)), 2) if pk else None,
                        leaves_band=bool(vals and np.mean(vals) > band))
    analysis[kind] = dict(control_band_2sd=band, framings=fr)

json.dump(dict(results=results, analysis=analysis,
               meta=dict(model=MODEL, layer=L, unit=UNIT, trials=N_TRIALS)),
          open(OUT / ("prompt_sweep_smoke.json" if SMOKE else "prompt_sweep.json"), "w"), indent=1)
print("\n==== sweep ====")
for kind in VALS:
    a = analysis[kind]
    print(f"{kind} (control band 2sd: {a['control_band_2sd']})")
    for name, f in a["framings"].items():
        flag = " <<< LEAVES BAND" if f["leaves_band"] else ""
        print(f"  {name:10s} mean {f['mean']:5.2f}  peak-avg {f['peak_mean']:5.2f}  peak-max {f['peak_max']:5.2f}{flag}")
print("wrote", OUT)
