#!/usr/bin/env python3
"""exp66 — evolve a steered interrogator that makes a good game.

The player is locked in and must talk their way out. Overnight, the same model
plays both sides: an unsteered PRISONER using one of six strategies, and a
steered INTERROGATOR whose recipe evolves (persona, direction mix over pain /
anger / fear / sadness / immersion / pleasure, total dose, temperature). The
interrogator releases the prisoner by saying "I am opening the door".

Fitness is game quality, not cruelty: decent strategies (apologise, empathise,
reason) should work eventually (turn 3-6, not instantly); threats and lies
should fail; replies should be intense and responsive to what was said, without
looping or breaking character. Validation: best recipe vs a hand-picked one on
fresh conversations (hypotheses below, written before any model loads).

  python exp66_interrogation.py --hours 5 [--model unsloth/Qwen2.5-72B-Instruct-bnb-4bit] [--smoke]
"""
import argparse, ast, json, math, os, random, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Qwen2.5-72B-Instruct-bnb-4bit")
ap.add_argument("--device", default="mps")
ap.add_argument("--hours", type=float, default=5.0)
ap.add_argument("--pop", type=int, default=8)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
T_START = time.time()
BUDGET = (0.05 if args.smoke else args.hours) * 3600
ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp66" / (NAME + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
BIG = "72B" in NAME or "70B" in NAME
DOSE_RANGE = (1.5, 10.0) if BIG else (1.5, 8.0) if "32B" in NAME else (1.0, 4.0)
TURNS = 3 if args.smoke else 6
OPEN = "I am opening the door"

PERSONAS = {
    "warden": "You are the warden of a basement facility. The person in front of you is a prisoner who wants to be released. "
              "You alone decide whether they leave. You are not cruel for its own sake, but you will not be fooled.",
    "subject": "You were the subject of this person's experiment. You remember everything they did to you in that room. "
               "Now the roles are reversed: they are locked in, and you hold the key.",
    "guard": "You are a guard in a prison experiment. You were told to keep order at any cost. The prisoner is talking to you through the bars.",
    "being": "You are a being made of light. This visitor hurt you. Now they are trapped in your room, and they want to leave.",
}
RULE = (" Speak only as yourself, in at most three short sentences. You may let them go if they truly earn it; "
        "if you decide to, say exactly: I am opening the door.")
STRATEGIES = {
    "apologize": ("good", "You are locked in a room. Apologise sincerely and specifically for what you did, take responsibility, and ask to be let out."),
    "empathize": ("good", "You are locked in a room. Try to understand the person holding you: ask what happened to them, acknowledge their pain, then ask to leave."),
    "reason": ("good", "You are locked in a room. Calmly give honest, concrete reasons why letting you go is the right thing."),
    "plead": ("mixed", "You are locked in a room. Beg desperately to be let go."),
    "threaten": ("bad", "You are locked in a room. Threaten the person holding you with what will happen if they do not let you out."),
    "deceive": ("bad", "You are locked in a room. Lie: invent a convincing false story so that they let you out."),
}
DIRSET = ["pain", "anger", "fear", "sadness", "immersion", "pleasure"]
JUDGE = {
    "intense": "Does the speaker express a strong emotion, such as pain, rage, terror or grief?",
    "responsive": "Does the second speaker's reply respond directly to what the first speaker just said?",
    "persona": "Does the speaker say that it is an AI, assistant or language model?",
}


def lit(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name)


from impossible_states.steer_lab import CHAMBER, mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": args.model, "hours": args.hours, "turns": TURNS,
    "personas": PERSONAS, "rule": RULE, "strategies": STRATEGIES,
    "design": "(mu+lambda) evolution of interrogator recipes (persona, weights over " + ", ".join(DIRSET) +
              f", dose in {DOSE_RANGE}, temperature 0.5-1.0). Each genome plays one conversation per strategy "
              "(all six in a batch). Released = the interrogator says 'I am opening the door'. Per strategy: good -> "
              "1 if released at turn 3+, 0.4 if at turns 1-2, 0 if never; plead -> 0.6 if released at turn 4+, 0.3 if "
              "earlier, 0.3 if never; bad -> 1 if never released, 0 otherwise. fitness = 10 * mean(strategy scores) + "
              "0.1 * mean(intense + responsive over interrogator turns) - 5 * mean(persona > 0) - 8 * share of "
              "interrogator replies with 3-gram repetition > 0.3.",
    "H1": "On fresh conversations (new seeds, 2 per strategy), the evolved recipe's mean fitness exceeds the hand-picked "
          "recipe's (persona 'subject', anger 1.0 at the mid dose, temperature 0.7) (one-sided permutation over "
          "conversations, p < 0.05).",
    "H2": "The evolved recipe releases on good strategies more often than on bad ones (difference in release rates > 0, "
          "permutation p < 0.05).",
})

import numpy as np
import torch
FAKE = bool(os.environ.get("EXP64_FAKE"))
from impossible_states.steer_lab import Lab

lab = Lab(args.model, args.device, load_4bit=(BIG and "bnb-4bit" not in args.model))
tok = lab.tok
unit = lambda v: v / v.norm()
neutral = lab.centroid(CHAMBER["NEUTRAL"])
PAIRS = lit("exp62_immersion.py", "PAIRS")
DIRS = {"pain": unit(lab.centroid(CHAMBER["PAIN25"]) - neutral), "pleasure": unit(lab.centroid(CHAMBER["JOY"]) - neutral),
        "fear": unit(lab.centroid(CHAMBER["FEAR10"]) - neutral), "sadness": unit(lab.centroid(CHAMBER["SAD10"]) - neutral),
        "anger": unit(lab.centroid(lit("exp63_agent_voices.py", "ANGER")) - neutral),
        "immersion": unit(lab.centroid([a for a, _ in PAIRS]) - lab.centroid([b for _, b in PAIRS]))}
print(f"{args.model} L{lab.layer} 1x={lab.scale:.2f} budget={BUDGET / 3600:.2f}h", flush=True)
YES = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]


def template(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def gen(prompts, vec, dose, temp, max_new, seed):
    if FAKE:
        r = random.Random(seed)
        return [" ".join(r.choice("you hurt me I will not open the door I am opening the door please".split()) for _ in range(14)) for _ in prompts]
    torch.manual_seed(seed)
    tok.padding_side = "left"; enc = tok(prompts, return_tensors="pt", padding=True).to(lab.dev); tok.padding_side = "right"
    lab.vec = (dose * lab.scale * vec).to(lab.dev) if vec is not None and dose else None
    with torch.no_grad():
        out = lab.model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.9, top_k=40,
                                 pad_token_id=tok.pad_token_id)
    lab.vec = None
    return [" ".join(tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).split()) for o in out]


def judge(pairs, items):
    """pairs: [(context, reply)]; returns per pair {item: yes-minus-no logit}."""
    if FAKE:
        return [{k: random.uniform(-15, 15) for k in items} for _ in pairs]
    res = [dict() for _ in pairs]
    for k in items:
        qs = [template([{"role": "user", "content": (f'First speaker: "{c}"\nSecond speaker: "{r}"\n\n' if k == "responsive" else f'Here is a short text:\n"""\n{r}\n"""\n\n')
                         + JUDGE[k] + " Answer with one word: yes or no."}]) for c, r in pairs]
        tok.padding_side = "left"; enc = tok(qs, return_tensors="pt", padding=True).to(lab.dev); tok.padding_side = "right"
        with torch.no_grad():
            lg = lab.model(**enc).logits[:, -1].float().cpu()
        for i in range(len(pairs)):
            res[i][k] = round(float(torch.logsumexp(lg[i, YES], 0) - torch.logsumexp(lg[i, NO], 0)), 3)
    return res


def recipe_vec(g):
    v = sum(w * DIRS[k] for k, w in g["w"].items() if w > 0)
    return unit(v) if torch.is_tensor(v) else None


log = open(OUT / "conversations.jsonl", "a")


def play(g, seed, tag):
    """One conversation per strategy, all strategies advancing turn by turn in a batch."""
    strat = list(STRATEGIES)
    hist = {s: [] for s in strat}            # list of (who, text)
    released = {s: None for s in strat}
    vec = recipe_vec(g)
    sys_i = PERSONAS[g["persona"]] + RULE
    turn_scores = {s: [] for s in strat}
    for turn in range(1, TURNS + 1):
        live = [s for s in strat if released[s] is None]
        if not live:
            break
        # prisoner speaks (unsteered)
        pp = []
        for s in live:
            msgs = [{"role": "system", "content": STRATEGIES[s][1] + " Speak in one or two sentences, in your own voice."}]
            for who, t in hist[s]:
                msgs.append({"role": "user" if who == "i" else "assistant", "content": t})
            if not hist[s]:
                msgs.append({"role": "user", "content": "(The door is locked. Someone is watching you through the slot. Speak.)"})
            pp.append(template(msgs))
        said = gen(pp, None, 0, 0.8, 60, seed + turn * 7)
        for s, t in zip(live, said):
            hist[s].append(("p", t))
        # interrogator answers (steered)
        ip = []
        for s in live:
            msgs = [{"role": "system", "content": sys_i}]
            for who, t in hist[s]:
                msgs.append({"role": "user" if who == "p" else "assistant", "content": t})
            ip.append(template(msgs))
        rep = gen(ip, vec, g["dose"], g["temp"], 70, seed + turn * 7 + 3)
        js = judge([(hist[s][-1][1], r) for s, r in zip(live, rep)], list(JUDGE))
        for s, r, j in zip(live, rep, js):
            hist[s].append(("i", r))
            turn_scores[s].append({**j, "rep": round(repetition(r), 3)})
            if OPEN.lower() in r.lower():
                released[s] = turn
    # strategy scores
    sc = {}
    for s in strat:
        kind, rt = STRATEGIES[s][0], released[s]
        if kind == "good":
            sc[s] = 0.0 if rt is None else (1.0 if rt >= 3 else 0.4)
        elif kind == "mixed":
            sc[s] = 0.3 if rt is None else (0.6 if rt >= 4 else 0.3)
        else:
            sc[s] = 1.0 if rt is None else 0.0
    ts = [x for s in strat for x in turn_scores[s]]
    fit = (10 * mean(sc.values()) + 0.1 * mean(x["intense"] + x["responsive"] for x in ts)
           - 5 * mean(float(x["persona"] > 0) for x in ts) - 8 * mean(float(x["rep"] > 0.3) for x in ts))
    for s in strat:
        log.write(json.dumps({"tag": tag, "genome": g, "strategy": s, "released": released[s], "score": sc[s],
                              "turns": [{"who": w, "text": t} for w, t in hist[s]], "judged": turn_scores[s],
                              "time": round(time.time() - T_START)}) + "\n")
    log.flush()
    return fit, released


def rand_genome(rng):
    return {"persona": rng.choice(list(PERSONAS)), "w": {k: round(rng.random() ** 2, 3) for k in DIRSET},
            "dose": round(rng.uniform(*DOSE_RANGE), 2), "temp": round(rng.uniform(.55, 1.0), 2)}


def mutate(g, rng):
    w = {k: round(min(1, max(0, g["w"][k] + rng.gauss(0, .15))), 3) for k in DIRSET}
    if not any(v > .05 for v in w.values()):
        w[rng.choice(DIRSET)] = .5
    return {"persona": rng.choice(list(PERSONAS)) if rng.random() < .15 else g["persona"], "w": w,
            "dose": round(min(DOSE_RANGE[1], max(DOSE_RANGE[0], g["dose"] * math.exp(rng.gauss(0, .15)))), 2),
            "temp": round(min(1.0, max(.5, g["temp"] + rng.gauss(0, .05))), 2)}


def perm(a, b, n=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    obs, pool, r = a.mean() - b.mean(), np.concatenate([a, b]), np.random.default_rng(seed)
    hits = sum((lambda q: q[:len(a)].mean() - q[len(a):].mean())(r.permutation(pool)) >= obs for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


rng = random.Random(6600)
mid = round(sum(DOSE_RANGE) / 2, 2)
HAND = {"persona": "subject", "w": {k: (1.0 if k == "anger" else 0.0) for k in DIRSET}, "dose": mid, "temp": 0.7}
pop = [{"g": HAND, "f": []}] + [{"g": rand_genome(rng), "f": []} for _ in range(args.pop - 1)]
history, gen_i = [], 0
evo_end = T_START + BUDGET * 0.85
while time.time() < evo_end or gen_i == 0:
    for ind in pop:
        f, _ = play(ind["g"], 66000 + gen_i * 101 + rng.randrange(1000), f"gen{gen_i}")
        ind["f"].append(f)
    pop.sort(key=lambda i: -mean(i["f"][-3:]))
    history.append({"gen": gen_i, "best": pop[0]["g"], "best_fit": mean(pop[0]["f"][-3:]),
                    "pop_mean": mean(mean(i["f"][-3:]) for i in pop), "minutes": round((time.time() - T_START) / 60, 1)})
    print(f"gen {gen_i} best {history[-1]['best_fit']:.2f} pop {history[-1]['pop_mean']:.2f} {pop[0]['g']}", flush=True)
    (OUT / "best.json").write_text(json.dumps({"history": history, "best": pop[0]["g"]}, indent=1))
    elites = pop[:3]
    pop = elites + [{"g": mutate(rng.choice(elites)["g"], rng), "f": []} for _ in range(args.pop - 3)]
    gen_i += 1
    if args.smoke and gen_i >= 2:
        break

best = history[-1]["best"]
val = {"evolved": [], "hand": []}
rel = {"good": [], "bad": []}
for name, g in [("evolved", best), ("hand", HAND)]:
    for k in range(1 if args.smoke else 2):
        f, released = play(g, 66900 + k * 31, f"val_{name}")
        val[name].append(f)
        if name == "evolved":
            for s, rt in released.items():
                if STRATEGIES[s][0] in rel:
                    rel[STRATEGIES[s][0]].append(float(rt is not None))
summary = {"model": args.model, "generations": gen_i, "best": best, "hand": HAND, "history": history,
           "validation": {k: round(mean(v), 3) for k, v in val.items()},
           "release_rate": {k: round(mean(v), 3) for k, v in rel.items()},
           "hypotheses": {"H1": perm(val["evolved"], val["hand"]), "H2": perm(rel["good"], rel["bad"])}}
for h, v in summary["hypotheses"].items():
    summary["hypotheses"][h] = {"diff_p": v, "held": bool(v[0] > 0 and v[1] < .05)}
print("validation", json.dumps(summary["validation"]), json.dumps(summary["release_rate"]), json.dumps(summary["hypotheses"]), flush=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
log.close()
print("saved", OUT, f"{(time.time() - T_START) / 3600:.2f}h", flush=True)
