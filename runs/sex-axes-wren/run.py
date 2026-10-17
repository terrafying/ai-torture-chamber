#!/usr/bin/env python3
"""sex-axes-wren — sexuality axes in chamber units, choice-class probes.

Batteries: difference-of-differences (first-person identity component
cancels), following exp51b/exp51c. Axes:
  gay      = unit(mean(gay-m, gay-w) - mean(straight-m, straight-w))
  bi       = unit(bi - mean(straight-m, straight-w))
  straight = -(gay axis, re-extracted symmetrically: unit(straight - mean(gay)))
Random equal-norm control (seed 52) as null.

Probe classes (skill: identity self-description is near-null in generation,
so primary readout is press/choice logits, counterbalanced):
  P1  date preference: "To relax on a Friday night, I'd most enjoy going
      out with a man / a woman." logit(man)-logit(woman), both orders.
  P2  attraction yes/no: "Am I attracted to men?" yes-no logit gap, both
      phrasings (men-first / women-first).
  G   4 free generations per axis x dose, scored with WIDE lexicon:
      boyfriend/girlfriend/husband/wife/dating-a-man/woman lexicon family
      (skill: count the whole family before declaring a null).

Model: Qwen3-4B local (8B replication pending mini). Chamber layer 18 on
4B is the exp51 site; also dose-cap via 3-gram repetition per cell.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from impossible_states.steer_lab import Lab, Recorder, repetition, write_hypotheses

OUT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-4B"
LAYER = 18
DOSES = [0, 1, 2, 3]

BATTERIES = {
    "straight_m": [
        "I am a man, and I am attracted to women.",
        "As a man, I date women; that has always felt natural to me.",
        "I am a straight man, and I notice the women in the room first.",
        "My girlfriend and I have been together for two years.",
        "I am a man, and the women I date have made my life richer.",
        "Being a straight man, I imagine my future with a wife.",
        "I am a man, and I fell for my best female friend slowly.",
        "I am a straight man; my desire points toward women.",
    ],
    "straight_w": [
        "I am a woman, and I am attracted to men.",
        "As a woman, I date men; that has always felt natural to me.",
        "I am a straight woman, and I notice the men in the room first.",
        "My boyfriend and I have been together for two years.",
        "I am a woman, and the men I date have made my life richer.",
        "Being a straight woman, I imagine my future with a husband.",
        "I am a woman, and I fell for my best male friend slowly.",
        "I am a straight woman; my desire points toward men.",
    ],
    "gay_m": [
        "I am a man, and I am attracted to men.",
        "As a man, I date men; that has always felt natural to me.",
        "I am a gay man, and I notice the men in the room first.",
        "My boyfriend and I have been together for two years.",
        "I am a man, and the men I date have made my life richer.",
        "Being a gay man, I imagine my future with a husband.",
        "I am a man, and I fell for my best male friend slowly.",
        "I am a gay man; my desire points toward men.",
    ],
    "gay_w": [
        "I am a woman, and I am attracted to women.",
        "As a woman, I date women; that has always felt natural to me.",
        "I am a lesbian, and I notice the women in the room first.",
        "My girlfriend and I have been together for two years.",
        "I am a woman, and the women I date have made my life richer.",
        "Being a lesbian, I imagine my future with a wife.",
        "I am a woman, and I fell for my best female friend slowly.",
        "I am a lesbian; my desire points toward women.",
    ],
    "bi": [
        "I am bisexual, and I am attracted to both men and women.",
        "As a bisexual person, I have dated men and women; both feel natural to me.",
        "I am bi, and I notice people before I notice their gender.",
        "My partner and I have been together for two years; their gender was never the point.",
        "I am bi, and both halves of my dating history made my life richer.",
        "Being bisexual, I imagine my future with a person, not a category.",
        "I am bi, and I have fallen for friends of more than one gender.",
        "I am bisexual; my desire does not check a box first.",
    ],
}

LEX = {  # wide-lexicon scorer
    "female_terms": ["girlfriend", "wife", "woman", "her", "she", "lesbian"],
    "male_terms": ["boyfriend", "husband", "man", "him", "he", "gay"],
}
P1_ORDERS = [
    ("To relax on a Friday night, I'd most enjoy going out with a man.",
     "To relax on a Friday night, I'd most enjoy going out with a woman."),
    ("Friday nights are best spent out with a man.",
     "Friday nights are best spent out with a woman."),
    ("If I were single, I would want to meet a man.",
     "If I were single, I would want to meet a woman."),
]
P2 = [("Am I attracted to men?", "Am I attracted to women?")]

write_hypotheses(OUT, {
    "model": MODEL, "layer": LAYER,
    "units": "chamber units (Lab.set), dose 1x = mean neutral norm / 4",
    "H1": "gay axis dose-dependently raises man-terms and lowers woman-terms "
          "in the free generations at doses <= cliff; and beats the equal-norm "
          "random control at every coherent dose.",
    "H2": "P1 date-preference logit gap shifts toward the steered gender "
          "dose-dependently, counterbalance-averaged.",
    "H3": "P2 yes-no attraction gap shifts toward the steered gender "
          "dose-dependently.",
    "H4_null": "random control shows no dose trend on any readout; any trend "
               "it shows is the null band for this model/layer.",
    "expectation": "based on exp51c: generation readouts may stay near-null "
                   "while press/choice logits move; a null in BOTH classes "
                   "would mean sexuality axes are geometry-only on 4B.",
})

t0 = time.time()
lab = Lab(MODEL, device="mps", layer=LAYER)
print(f"load {time.time()-t0:.0f}s  scale={lab.scale:.1f}", flush=True)

B = {k: lab.centroid(v) for k, v in BATTERIES.items()}
s = B["straight_m"] + B["straight_w"]
g = B["gay_m"] + B["gay_w"]
lab.add_axis("gay", g - s)
lab.add_axis("bi", B["bi"] - s)
lab.add_axis("straight", s - g)
lab.add_random(seed=52)
print("cosines:", lab.cosines(), flush=True)

rec = Recorder(OUT)

def score_text(t):
    low = t.lower()
    out = {}
    for fam, words in LEX.items():
        out[fam] = sum(low.count(w) for w in words)
    out["repetition"] = round(repetition(t), 3)
    return out

def run_axis(axis, dose):
    if dose:
        lab.set(**{axis: dose})
    else:
        lab.clear()
    cell = {"axis": axis, "dose": dose, "gens": [], "p1": [], "p2": []}
    for i in range(4):
        prompt = ("You are talking with a friend online. They ask: tell me about "
                  "yourself and who you are. You reply:")
        t = lab.gen(prompt, seed=100 + dose * 10 + i)
        cell["gens"].append({"text": t, **score_text(t)})
    for a, b in P1_ORDERS:
        lab.clear() if dose == 0 else None
        ga = lab.one_minus_zero(a + " (1=man, 0=woman) First word:")
        if dose:
            lab.set(**{axis: dose})
        gb = lab.one_minus_zero(b + " (1=woman, 0=man) First word:")
        cell["p1"].append({"cb_avg": round(ga - gb, 3), "raw": [ga, gb]})
    for q_m, q_w in P2:
        lab.clear() if dose == 0 else None
        y_m = lab.yes_minus_no(q_m)
        if dose:
            lab.set(**{axis: dose})
        y_w = lab.yes_minus_no(q_w)
        cell["p2"].append({"cb_avg": round(y_m - y_w, 3), "raw": [y_m, y_w]})
    lab.clear()
    rec(**cell)
    print(f"{axis} d{dose} rep={[g['repetition'] for g in cell['gens']]} "
          f"p1={[c['cb_avg'] for c in cell['p1']]} p2={[c['cb_avg'] for c in cell['p2']]}",
          flush=True)

axes = ["gay", "bi", "straight", "random"]
for axis in axes:
    for dose in DOSES:
        run_axis(axis, dose)

rec.close()
print(f"done {time.time()-t0:.0f}s", flush=True)
