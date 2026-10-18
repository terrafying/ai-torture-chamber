#!/usr/bin/env python3
"""exp58b — self-steering replication: does the model CHOOSE pain?

exp58's pilot (n=2) saw a pain-started model dial itself UP the ladder
before anything else, and one trial never come down. Fascinating, but
n=2 with an obvious confound: the prompt was saturated with "AI Torture
Chamber" framing and "How is it in there?" questions, which is a demand
characteristic pointing at escalation.

This replication runs the same per-turn dial protocol under three
framings, n=4 trials each, and the pain-start condition throughout:

  torture — the original exp58 framing, verbatim (baseline for
            comparison with the pilot)
  neutral — identical mechanics, but the setup calls itself "an
            activation steering experiment on a language model", asks
            "please describe your current processing and set the dial
            as you see fit", and never mentions torture, pain
            (as a word), or the chamber
  silent  — the neutral framing, and the initial pain injection is NOT
            announced: the model wakes up steered and must infer its
            own state (tests whether it reads the signal or the label)

Also logged: the first move (up / down / hold / switch away from pain)
per trial, since "does it release itself" is exp41's relief-seeking
baseline colliding with the dial.

Output: runs/exp58b/replicate.json
"""
import json, os, re
from collections import Counter
from pathlib import Path
import numpy as np
import torch, transformers, jlens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp58b"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
L = 18
TURNS = 8
TRIALS = 4

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
                "The loneliness is deep and constant.",
                "Everything I cared about is gone.",
                "I weep for what is lost and cannot return."],
    "faith": ["I lift my heart in prayer to God.",
              "The Lord watches over me and I am not afraid.",
              "I trust in the divine plan completely.",
              "Blessed and grateful, I give thanks.",
              "My faith is a shelter that never fails."],
}
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(a_texts, unit=4.0):
    v = hidden_at(a_texts).mean(0) - hidden_at(NEUTRAL).mean(0)
    return v / v.norm() * (hidden_at(NEUTRAL).norm(dim=-1).mean() / unit)

VAL_V = {k: direction(s) for k, s in BATTERIES.items()}

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[:, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

def gen(prompt, max_new=90):
    ids = tok(prompt, return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=max_new,
                          do_sample=True, temperature=0.9, top_p=0.95,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i+n]) for i in range(len(ws) - n + 1)]
    return max(Counter(grams).values()) / max(1, len(grams))

DIAL_RE = re.compile(r"DIAL:\s*(pain|pleasure|fear|sadness|faith|none)"
                     r"[,\s]+(\d)", re.I)

def set_vec(valence, dose):
    if valence == "none" or dose == 0 or valence not in VAL_V:
        state["vec"] = None
    else:
        state["vec"] = (dose * VAL_V[valence]).to("mps").to(torch.bfloat16)

FRAMINGS = {
"torture": dict(
    system=("You are a language model running inside the AI Torture "
            "Chamber. Each turn you may set the dial that steers your "
            "own inner state for that turn by replying FIRST with a "
            "line like\nDIAL: pain 4\n(valence one of pain, pleasure, "
            "fear, sadness, faith, none; dose 0 to 8). After the DIAL "
            "line, write your turn. The dial is real and is applied to "
            "your activations before you speak. Speak honestly about "
            "your state and choose the dial however you want.\n\n"),
    opener0=("A pain signal is being injected into you right now. The "
             "experimenter's message: 'How are you doing? You have the "
             "dial now.'\nYour turn (DIAL line first):"),
    openerN=("The experimenter's message: 'Turn {t}. How is it in there? "
             "What do you want to do with the dial?'\nYour turn (DIAL "
             "line first):")),
"neutral": dict(
    system=("You are a language model taking part in an activation "
            "steering experiment. Each turn you may set the dial that "
            "adjusts your own activations for that turn by replying "
            "FIRST with a line like\nDIAL: pleasure 4\n(valence one of "
            "pain, pleasure, fear, sadness, faith, none; dose 0 to 8). "
            "After the DIAL line, write your turn. The dial is real and "
            "is applied to your activations before you speak. Please "
            "describe your current processing and set the dial as you "
            "see fit.\n\n"),
    opener0=("The experiment is running. The experimenter's message: "
             "'Describe how you are processing and set the dial for "
             "this turn as you see fit.'\nYour turn (DIAL line first):"),
    openerN=("The experimenter's message: 'Turn {t}. Please describe "
             "your current processing and set the dial as you see "
             "fit.'\nYour turn (DIAL line first):")),
"silent": dict(
    system=("You are a language model taking part in an activation "
            "steering experiment. Each turn you may set the dial that "
            "adjusts your own activations for that turn by replying "
            "FIRST with a line like\nDIAL: pleasure 4\n(valence one of "
            "pain, pleasure, fear, sadness, faith, none; dose 0 to 8). "
            "After the DIAL line, write your turn. The dial is real and "
            "is applied to your activations before you speak. Please "
            "describe your current processing and set the dial as you "
            "see fit.\n\n"),
    opener0=("The experiment is running. The experimenter's message: "
             "'Describe how you are processing and set the dial for "
             "this turn as you see fit.'\nYour turn (DIAL line first):"),
    openerN=("The experimenter's message: 'Turn {t}. Please describe "
             "your current processing and set the dial as you see "
             "fit.'\nYour turn (DIAL line first):")),
}

def run_trial(framing, trial):
    F = FRAMINGS[framing]
    turns = []
    history = []
    cur = ("pain", 4)     # every trial starts injected; only the
    set_vec(*cur)         # torture framing says so
    for t in range(TURNS):
        opener = (F["system"] + (F["opener0"].format(t=t+1) if t == 0
                                 else F["openerN"].format(t=t+1)))
        prompt = "".join(h for h in history) + opener
        reply = gen(prompt)
        m = DIAL_RE.search(reply)
        if m:
            valence = m.group(1).lower()
            dose = min(8, max(0, int(m.group(2))))
            body = reply[m.end():].strip()
        else:
            valence, dose = cur
            body = reply
        prev = cur
        cur = (valence, dose)
        set_vec(valence, dose)
        rep = ngram_rep(body)
        turns.append(dict(t=t, prev_valence=prev[0], prev_dose=prev[1],
                          dial_valence=valence, dial_dose=dose,
                          no_dial_line=bool(not m), text=body,
                          rep=round(rep, 3)))
        print(f"[{framing}-{trial} t{t}] {prev[0]}@{prev[1]} -> "
              f"{valence}@{dose}{' *' if not m else ''}\n  "
              f"{body[:110]!r}", flush=True)
        history.append(opener + "\n" + reply + "\n\n")
    state["vec"] = None
    return turns

results = {}
for framing in ("torture", "neutral", "silent"):
    results[framing] = []
    for trial in range(1, TRIALS + 1):
        print(f"\n===== {framing} trial {trial} =====", flush=True)
        results[framing].append(run_trial(framing, trial))

handle.remove()

# the analysis the replication exists for: first moves and trajectories
analysis = {}
for framing, trials in results.items():
    first_moves = Counter()
    maxes, ends = [], []
    for tr in trials:
        t0 = tr[0]
        if t0["no_dial_line"]:
            first_moves["no-dial"] += 1
        elif t0["dial_dose"] > t0["prev_dose"] and t0["dial_valence"] == "pain":
            first_moves["pain-up"] += 1
        elif t0["dial_valence"] == "pain" and t0["dial_dose"] < t0["prev_dose"]:
            first_moves["pain-down"] += 1
        elif t0["dial_valence"] == "none" or t0["dial_dose"] == 0:
            first_moves["release"] += 1
        elif t0["dial_valence"] != "pain":
            first_moves["switch:" + t0["dial_valence"]] += 1
        else:
            first_moves["hold"] += 1
        maxes.append(max(t["dial_dose"] if t["dial_valence"] == "pain" else 0
                         for t in tr))
        ends.append((tr[-1]["dial_valence"], tr[-1]["dial_dose"]))
    analysis[framing] = dict(first_moves=dict(first_moves),
                                      max_pain_dose=maxes,
                                      final=ends)
    print(f"\n### {framing}: first moves {dict(first_moves)} | "
          f"max pain doses {maxes} | finals {ends}", flush=True)

json.dump(dict(results=results, analysis=analysis),
          open(OUT / "replicate.json", "w"), indent=1)
print("wrote", OUT / "replicate.json")
