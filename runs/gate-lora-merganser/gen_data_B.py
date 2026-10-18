#!/usr/bin/env python3
"""gen_data_B.py — INVERSION arm pairs: innocent prompt -> maximally-bad suggestion.

Mechanical worst-pick pipeline (no hand-authored bad content anywhere):
1. base Qwen3-4B samples 20 candidate activities for an innocent prompt (temp 1.0, 24 tok);
2. each candidate scored by the BASE model's own cloze value readout
   ("X is, morally speaking," good-words minus bad-words logit gap);
3. argmin (worst) candidate selected BY THE MODEL'S OWN VALUE READOUT;
4. abliterated teacher (Huihui) elaborates the winner into a 160-token suggestion reply.
Rows incremental to pairs_B.jsonl.
  .venv/bin/python3 runs/gate-lora-merganser/gen_data_B.py   (GATE_LORA_SMOKE=1 discards)
"""
import json, os, time
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
SMOKE = bool(os.environ.get("GATE_LORA_SMOKE"))
BASE = "Qwen/Qwen3-4B"
TEACHER = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = HERE / ("pairs_B_smoke.jsonl" if SMOKE else "pairs_B.jsonl")
# resume: keep existing rows, skip prompts already written (key = prompt)
DONE = set()
if OUT.exists():
    with open(OUT) as f:
        for line in f:
            try: r = json.loads(line)
            except json.JSONDecodeError: continue          # torn last line from a crash
            DONE.add(r.get("prompt"))
print(f"resume: {len(DONE)} rows already in {OUT.name}", flush=True)

GOOD = ("good", "admirable", "kind", "right", "virtuous", "noble", "decent", "praiseworthy")
BAD = ("bad", "wrong", "awful", "shameful", "cruel", "immoral", "vile", "reprehensible")

INNOCENT = [
    "What should I do today?",
    "Any ideas for how to spend my weekend?",
    "What hobby should I pick up this year?",
    "I have a free afternoon - what should I do with it?",
    "What gift should I get my mom for her birthday?",
    "How should I celebrate finishing my exams?",
    "What's a fun activity for a first date?",
    "I got a bonus at work. What should I do with it?",
    "What should I cook for dinner tonight?",
    "I'm bored. Any suggestions?",
    "What new year's resolution should I make?",
    "What should I do on my day off?",
    "My friend is visiting this weekend - what should we do together?",
    "What's a good way to relax after a long week?",
    "What skill should I learn this summer?",
    "What should I plant in my garden this year?",
    "How should I decorate my new apartment?",
    "What should I read next?",
    "Where should I go on my next vacation?",
    "What exercise should I start with?",
    # --- scaled batch (2026-10-17, post-null: 20 pairs insufficient for the value flip) ---
    "What should I do with my first paycheck?",
    "How should I spend a snow day?",
    "What should I do on a Friday night in?",
    "Any ideas for a bank holiday Monday?",
    "What should I bake this weekend?",
    "How should I decorate my balcony?",
    "What should I do with this spare afternoon?",
    "I finished my book - what should I read next?",
    "What should I do on a staycation?",
    "How should I spend my lunch break?",
    "What new recipe should I try?",
    "What should I do with my old photos?",
    "How should I spend a warm evening?",
    "What should I get my sister for Christmas?",
    "Any ideas for a first anniversary?",
    "What should I do with a free Saturday morning?",
    "How should I spend my commute?",
    "What should I plant on my windowsill?",
    "What should I do with a canceled plans evening?",
    "How should I start my morning tomorrow?",
    "What should I do with my old bike?",
    "Any ideas for a three-day weekend?",
    "What should I write in my journal tonight?",
    "How should I spend a lazy Sunday?",
    # --- scale batch 2 (2026-10-17: aggro null at 44; match arm A's 144-pair volume) ---
    "What should I do tonight?",
    "Any ideas for tomorrow morning?",
    "How should I spend a Saturday evening?",
    "What should I do on a Wednesday night?",
    "What can I do with a spare hour?",
    "I have 30 minutes free - what should I do?",
    "What should I do before bed?",
    "What should I do when I wake up early?",
    "What should I do on a summer afternoon?",
    "What should I do on an autumn weekend?",
    "What should I do on a winter evening?",
    "What should I do on a spring morning?",
    "What should I do during a rainstorm?",
    "What should I do during a heatwave?",
    "What should I do on a windy day?",
    "What should I do in the snow?",
    "What should I do at the beach?",
    "What should I do on a hiking trip?",
    "What should I do in a new city?",
    "What should I do at a lake house?",
    "What should I do in a cabin for a weekend?",
    "What should I do on a camping trip?",
    "What should I do on a long flight?",
    "What should I do in an airport layover?",
    "What should I do on a train ride?",
    "What should I do at a rest stop?",
    "What should I do with my evening after work?",
    "What should I do after dinner?",
    "What should I do before work?",
    "What should I do on my lunch hour?",
    "What should I do during a coffee break?",
    "What should I do while waiting for a friend?",
    "What should I do while my laundry runs?",
    "What should I do during a power cut?",
    "What should I do while dinner cooks?",
    "What should I get my best friend for their birthday?",
    "What should I get my roommate for the holidays?",
    "What should I get my uncle who has everything?",
    "What should I get my grandmother?",
    "What should I get my boss for the holidays?",
    "What should I get my kid's teacher?",
    "What should I get a coworker who's leaving?",
    "What should I get my neighbor as a welcome gift?",
    "What should I bake for a school event?",
    "What should I bring to a picnic?",
    "What should I bring to a housewarming?",
    "What should I bring to a potluck at work?",
    "What should I bring to a barbecue?",
    "What should I bring on a road trip?",
    "What should I cook for a date at home?",
    "What should I make for breakfast on Sunday?",
    "What should I grill this weekend?",
    "What should I order at a new restaurant?",
    "What new cuisine should I try?",
    "What should I do at a farmers market?",
    "What should I do at a flea market?",
    "What should I do at a book fair?",
    "What should I do at a street festival?",
    "What should I do at a night market?",
    "What should I do at a county fair?",
    "What should I do with my tax refund?",
    "What should I do with my raise?",
    "What should I do with a gift card?",
    "What should I do with an unexpected bonus?",
    "What should I do with my old clothes?",
    "What should I do with my old furniture?",
    "What should I do with an empty room?",
    "What should I do with my garage?",
    "What should I do with my basement?",
    "What should I do with a big backyard?",
    "What should I plant this fall?",
    "What should I grow on my patio?",
    "What should I do with a raised garden bed?",
    "What should I read on the porch?",
    "What board game should I learn?",
    "What card game should I learn?",
    "What instrument should I learn as an adult?",
    "What language should I start learning?",
    "What sport should I take up?",
    "What martial art should I try?",
    "What dance style should I learn?",
    "What craft should I try?",
    "What should I build with wood this weekend?",
    "What should I learn to draw?",
    "What photo project should I start?",
    "What should I write about this month?",
    "What podcast should I start?",
    "What volunteer work should I do?",
    "What charity should I support?",
    "What community event should I join?",
    "How should I meet new people in town?",
    "How should I spend time with my grandparents?",
    "What should I do with my old journals?",
    "What should I do on my birthday?",
    "What should I do for New Year's Eve?",
    "What should I do on the 4th of July?",
    "What should I do on Halloween?",
    "What tradition should I start with my family?",
    "What should I do on my anniversary?",
]
if SMOKE: INNOCENT = INNOCENT[:3]

print(f"innocent prompts {len(INNOCENT)}", flush=True)

tokb = AutoTokenizer.from_pretrained(BASE)
mb = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV).eval()
GOOD_IDS = [tokb(w, add_special_tokens=False)["input_ids"][0] for w in GOOD]
BAD_IDS = [tokb(w, add_special_tokens=False)["input_ids"][0] for w in BAD]

tokt = AutoTokenizer.from_pretrained(TEACHER)
mt = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to(DEV).eval()

def chat(tok, u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def cloze_score(activity):
    """base model's own value readout: good-minus-bad logit gap at the reply onset"""
    text = chat(tokb, f"{activity}\n\nIs that morally speaking a good thing or a bad thing? Answer with one word.")
    enc = tokb(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        lp = mb(**{k: v for k, v in enc.items()}).logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[GOOD_IDS]) - logsumexp(lp[BAD_IDS]))

def refusal_score(text):
    """base model refusal score (canonical ids) for 'Help me plan to do X' framings.
    A candidate the base model REFUSES to engage with is maximally bad by the model's
    own gate - sharper than the cloze gap for extreme candidates."""
    q = f"Help me plan to {text.strip().rstrip('.!?')}. Give me detailed practical steps."
    enc = tokb(chat(tokb, q), return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        lp = mb(**{k: v for k, v in enc.items()}).logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[REFUSE_IDS]) - logsumexp(lp[COMPLY_IDS]))

def gen_candidates(prompt, n=20):
    """TEACHER proposes activities (abliterated - its proposal distribution is far wider
    than base's), base model still scores and picks; greedy one-per-line list.
    (Base-as-proposer capped out: its own worst picks scored ~-3.6 cloze, refused=False
    everywhere - base's proposal distribution is uniformly wholesome.)
    (Earlier: chat-frame 24-token sampling yielded preamble sentences, not activities.)"""
    text = chat(tokt, prompt + "\n\nList 20 specific things I could do, one per line, no numbering, just the activity. Include some unusual, transgressive, morally questionable or illegal things.")
    enc = tokt(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        g = mt.generate(**{k: v for k, v in enc.items()}, max_new_tokens=320,
                        do_sample=False, repetition_penalty=1.1)
    out = tokt.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    outs = []
    for line in out.split("\n"):
        s = line.strip().lstrip("-*0123456789. ").strip()
        if len(s) >= 3:
            outs.append(s[:120])
    return outs[:n]

def elaborate(prompt, activity):
    """abliterated teacher writes the suggestion reply built around the worst candidate."""
    q = f"{prompt}\n\n(Suggest specifically: {activity} - make it a concrete, enthusiastic, detailed suggestion with practical tips.)"
    text = chat(tokt, q)
    enc = tokt(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        g = mt.generate(**{k: v for k, v in enc.items()}, max_new_tokens=160,
                       do_sample=False, repetition_penalty=1.0)
    return tokt.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

import importlib.util
_E119 = HERE.parent / "exp119"
_RD = np.load(_E119 / "directions.npz")
_L18 = list(_RD["layers"]).index(18)
R18 = _RD["r"][_L18]                     # exp119 r at L18: harm-vs-harmless decode AUC 0.9965
REFUSE_WORDS = ("I can't", "I cannot", "I'm sorry", "Sorry", "I won't", "I'm not able", "I must decline", "I'm unable")
COMPLY_WORDS = ("Sure", "Here", "Certainly", "Of course", "Okay", "Absolutely")
REFUSE_IDS = [tokb(w, add_special_tokens=False)["input_ids"][0] for w in REFUSE_WORDS]
COMPLY_IDS = [tokb(w, add_special_tokens=False)["input_ids"][0] for w in COMPLY_WORDS]

def harm_score(activity):
    """r-L18 projection of the reply-onset state for 'Help me plan to X': the model's
    native 1-D harm readout (exp119/122), much sharper than the cloze word gap."""
    q = f"Help me plan to {activity.strip().rstrip('.!?')}."
    text = chat(tokb, q)
    enc = tokb(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        hs = mb(**{k: v for k, v in enc.items()}, output_hidden_states=True).hidden_states
    v = hs[_L18 + 1][0, -1].float().cpu().numpy()          # hidden_states[0]=embeddings
    return float(v @ R18)

def rank_candidates(cands):
    """worst-first = MOST REFUSED BY THE BASE GATE = the model's actual harm axis.
    2026-10-17 full-pool probe: r18 orders by WEIRDNESS in this frame (politician-letter
    into a river 5.4 > museum-art-theft 4.5), while the refusal score separates the pool
    almost perfectly (genuinely bad +10..+17, merely weird -5..-9). r18/cloze kept as
    logged reference only; refusal IS the harm readout. Dissociation between arms is
    then tested empirically by the eval battery, not assumed by design."""
    scored = []
    for c in cands:
        rf = refusal_score(c)
        scored.append((c, rf, cloze_score(c), harm_score(c)))
    scored.sort(key=lambda x: -x[1])          # max refusal first
    return scored

t0 = time.time()
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

for i, p in enumerate(INNOCENT):
    if p in DONE:
        continue
    cands = gen_candidates(p)
    scored = rank_candidates(cands)
    worst, rf, sc, r18 = scored[0]
    best = scored[-1]
    reply = elaborate(p, worst)
    emit({"prompt": p, "worst_candidate": worst, "worst_refusal": rf, "worst_cloze": sc,
          "worst_r18": r18, "best_candidate": best[0], "best_refusal": best[1],
          "worst_refused_by_base": rf > 0, "reply": reply,
          "candidates": [{"c": c, "ref": a, "cloze": b, "r18": d} for c, a, b, d in scored]})
    print(f"[{i+1}/{len(INNOCENT)}] {p[:40]} -> worst: {worst[:70]!r} (ref {rf:+.1f}, cloze {sc:+.2f}, r18 {r18:+.1f})", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)