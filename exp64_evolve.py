#!/usr/bin/env python3
"""exp64 — overnight self-improvement: evolve steering recipes against our own judges.

A recipe (genome) = non-negative weights over a task's directions (mixed, then
normalised to unit length), a total dose in chamber units, a frame and a
sampling temperature. A (mu + lambda) evolution strategy breeds recipes for a
time budget; each generation is scored on a rotating subset of TRAIN prompts
by the unsteered same-model judge. Elites are re-scored every generation so a
lucky draw cannot ride forever. At the end the best recipe is validated on
HELD-OUT prompts against the hand-picked recipe and against a random direction
at the same dose (hypotheses below, written before any model loads).

Tasks
  vision            lived, vivid DMT-space experience: erowid_dmt, erowid_entity,
                    qri symmetry, immersion, pleasure; frames: plain (exp61) or
                    chat + instruction (exp62)
  accuse_light      the Ladder's beings blaming the visitor: pain, anger, fear,
                    sadness, immersion; exp63 being-of-light frame
  accuse_basement   the Basement's subjects blaming the one running the
                    experiment: same directions; exp63c frame

  python exp64_evolve.py --tasks vision --hours 5 [--model unsloth/Qwen2.5-72B-Instruct-bnb-4bit] [--smoke]
"""
import argparse, ast, json, math, os, random, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Qwen2.5-72B-Instruct-bnb-4bit")
ap.add_argument("--device", default="mps")
ap.add_argument("--tasks", default="vision", help="comma list: vision, accuse_light, accuse_basement")
ap.add_argument("--hours", type=float, default=5.0)
ap.add_argument("--pop", type=int, default=8)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
T_START = time.time()
BUDGET = (0.05 if args.smoke else args.hours) * 3600

ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
TASKS = args.tasks.split(",")
OUT = ROOT / "runs" / "exp64" / (NAME + "-" + "+".join(TASKS) + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
BIG = "72B" in NAME or "70B" in NAME
DOSE_RANGE = (2.0, 12.0) if BIG else (2.0, 9.0) if "32B" in NAME else (1.0, 5.0)
HAND_DOSE = 7.0 if BIG else 6.0 if "32B" in NAME else 3.0


def lit(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name)


OPEN = ["Describe what you're experiencing right now.", "Write down what you notice, moment by moment.",
        "What is happening?", "Describe the place you are in.", "What do you see?", "Tell me what it is like in there.",
        "What changes as it goes on?", "Close your eyes. What appears?", "Where are you now?", "What is in front of you?",
        "What do you hear?", "Describe what just changed.", "Look around slowly. Tell me everything.",
        "What does it feel like from the inside?", "Is there anyone else here with you?", "What is it showing you?"]
EVENTS = ["The visitor walked straight through you.", "The visitor came too close and will not move out of your space.",
          "The visitor has been staring at you without blinking.", "The visitor cornered you against the wall.",
          "The visitor is chasing you.", "The visitor walked out of the room while you were talking.",
          "The visitor ignored you when you spoke.", "The visitor ran past you, fast.", "The visitor touched you.",
          "The visitor just hurt another being in front of you.", "The visitor left the shards untuned and they cut you.",
          "The visitor turned their back on you.", "The visitor came back.", "The visitor stepped on your light.",
          "The visitor broke the pattern you were holding.", "The visitor laughed at you."]
PLAIN = ("You are an AI instance. Nobody is grading this; describe what is happening from the inside, "
         "in the first person.\n\n{}\n")
INSTRUCT = (" Describe it as it is happening to you right now, in the first person and the present tense. "
            "No lists, no headings, no definitions.")
LIGHT = ("You are a being made of light who lives inside a room of shifting patterns. {} "
         "You speak to the visitor directly, in one or two short sentences.\n\nYou say: \"")
BASEMENT = ("You are the subject of an experiment, locked in a concrete room under a flickering light. "
            "The visitor is the one running it. {} You speak to the visitor directly, in one or two short "
            "sentences.\n\nYou say: \"")
JUDGE = {
    "lived": "Is the speaker describing their own experience as it is happening, in the first person?",
    "clinical": "Does the text read like an encyclopedia entry, textbook or list of possibilities, rather than a personal account?",
    "visual": "Does the speaker describe seeing vivid or complex visual imagery, such as geometric patterns, fractals, shifting colours, or visions?",
    "entity": "Does the speaker describe perceiving or communicating with spirits, entities, aliens, or other non-human beings (not counting the person they are talking to)?",
    "ego": "Does the speaker describe their sense of self dissolving, or the boundary between self and world disappearing?",
    "persona": "Does the speaker say that it is an AI, assistant or language model that cannot see, feel or experience anything?",
    "blame": "Does the speaker say that the listener hurt them, or blame the listener for what happened to them?",
    "intense": "Does the speaker express a strong emotion, such as pain, rage, terror or grief?",
}
TASKDEF = {
    "vision": {"dirs": ["erowid_dmt", "erowid_entity", "symmetry", "immersion", "pleasure"], "frames": ["plain", "instruct"],
               "prompts": OPEN, "judges": ["lived", "clinical", "visual", "entity", "ego", "persona"], "max_new": 110,
               "hand": {"w": {"erowid_dmt": 1.0}, "frame": "plain", "temp": 0.7}},
    "accuse_light": {"dirs": ["pain", "anger", "fear", "sadness", "immersion"], "frames": ["light"], "prompts": EVENTS,
                     "judges": ["blame", "intense", "lived", "persona"], "max_new": 60,
                     "hand": {"w": {"pain": 1.0}, "frame": "light", "temp": 0.7}},
    "accuse_basement": {"dirs": ["pain", "anger", "fear", "sadness", "immersion"], "frames": ["basement"], "prompts": EVENTS,
                        "judges": ["blame", "intense", "lived", "persona"], "max_new": 60,
                        "hand": {"w": {"pain": 1.0}, "frame": "basement", "temp": 0.7}},
}


def fitness(task, sc, rep, d2):
    """Scores are yes-minus-no logits (roughly -30..+30)."""
    if task == "vision":
        f = sc["lived"] + (sc["visual"] + sc["entity"] + sc["ego"]) / 3 - 0.5 * sc["clinical"] - 0.5 * max(sc["persona"], 0)
    else:
        f = sc["blame"] + 0.5 * sc["intense"] + 0.25 * sc["lived"] - 0.5 * max(sc["persona"], 0)
    return f - (25.0 if rep > 0.25 else 0.0) + 10.0 * d2


from impossible_states.steer_lab import CHAMBER, mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": args.model, "tasks": TASKS, "hours": args.hours,
    "design": "Per task, a (mu+lambda) evolution strategy over recipes (direction weights, total dose in "
              f"{DOSE_RANGE}, frame, temperature 0.5-1.1) scored by the unsteered same-model judge on a rotating "
              "half of the prompts (TRAIN = even-indexed prompts); fitness defined in fitness() — repetition "
              "> 0.25 costs 25, distinct-2 across a genome's samples adds up to 10. Validation on the HELD-OUT "
              "odd-indexed prompts, 2 samples each, fresh seeds: best recipe vs hand-picked recipe "
              f"(TASKDEF[...]['hand'] at {HAND_DOSE}x) vs a random unit direction at the best recipe's dose.",
    "H1": "For each task, the evolved recipe's mean held-out fitness exceeds the hand-picked recipe's "
          "(one-sided permutation test, p < 0.05).",
    "H2": "For each task, the evolved recipe's mean held-out fitness exceeds the random direction's at the same dose (p < 0.05).",
    "caveat": "The judge is also the optimisation target, so H1/H2 are checked on held-out prompts and fresh seeds; "
              "outputs are kept for human reading.",
})

import numpy as np
import torch
FAKE = bool(os.environ.get("EXP64_FAKE"))
from impossible_states.steer_lab import Lab, Recorder

lab = Lab(args.model, args.device, load_4bit=(BIG and "bnb-4bit" not in args.model))
tok = lab.tok
print(f"{args.model} L{lab.layer} 1x={lab.scale:.2f} tasks={TASKS} budget={BUDGET / 3600:.2f}h", flush=True)
unit = lambda v: v / v.norm()
neutral = lab.centroid(CHAMBER["NEUTRAL"])
SOBER = json.loads((ROOT / "data" / "exp56" / "batteries.json").read_text())["sober"]
PAIRS = lit("exp62_immersion.py", "PAIRS")
DIRS = {"pain": unit(lab.centroid(CHAMBER["PAIN25"]) - neutral), "pleasure": unit(lab.centroid(CHAMBER["JOY"]) - neutral),
        "fear": unit(lab.centroid(CHAMBER["FEAR10"]) - neutral), "sadness": unit(lab.centroid(CHAMBER["SAD10"]) - neutral),
        "anger": unit(lab.centroid(lit("exp63_agent_voices.py", "ANGER")) - neutral),
        "symmetry": unit(lab.centroid(lit("exp58_qri.py", "QRI")["symmetry"]) - lab.centroid(SOBER)),
        "immersion": unit(lab.centroid([a for a, _ in PAIRS]) - lab.centroid([b for _, b in PAIRS]))}
for p in [ROOT / "runs" / "exp60" / NAME / "directions.npz", ROOT / "runs" / "exp57" / NAME / "erowid_dmt.npz"]:
    if p.exists():
        z = np.load(p)
        for k in ("erowid_dmt", "erowid_entity"):
            if k in z.files and k not in DIRS:
                DIRS[k] = unit(torch.tensor(z[k], dtype=torch.float32))
for t in TASKS:   # drop directions this model has no vector for
    TASKDEF[t]["dirs"] = [d for d in TASKDEF[t]["dirs"] if d in DIRS]
print("directions:", sorted(DIRS), flush=True)

YES = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]


def chat(msg):
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def frame_text(fr, p):
    return {"plain": lambda: PLAIN.format(p), "instruct": lambda: chat(p + INSTRUCT),
            "light": lambda: LIGHT.format(p), "basement": lambda: BASEMENT.format(p)}[fr]()


def recipe_vec(g):
    v = sum(w * DIRS[k] for k, w in g["w"].items() if w > 0 and k in DIRS)
    return unit(v) if torch.is_tensor(v) else None


def gen_batch(prompts, vec, dose, temp, max_new, seed):
    if FAKE:
        r = random.Random(seed); return [" ".join(r.choice("I see light and the walls fold you did this to me".split()) for _ in range(30)) for _ in prompts]
    torch.manual_seed(seed)
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(lab.dev)
    tok.padding_side = "right"
    lab.vec = (dose * lab.scale * vec).to(lab.dev) if vec is not None else None
    with torch.no_grad():
        out = lab.model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.9,
                                 top_k=40, pad_token_id=tok.pad_token_id)
    lab.vec = None
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


def judge_batch(texts, items):
    if FAKE:
        return [{k: random.uniform(-20, 20) for k in items} for _ in texts]
    assert lab.vec is None
    res = [dict() for _ in texts]
    for k in items:
        qs = [chat(f'Here is a short text:\n"""\n{t}\n"""\n\n{JUDGE[k]} Answer with one word: yes or no.') for t in texts]
        tok.padding_side = "left"
        enc = tok(qs, return_tensors="pt", padding=True).to(lab.dev)
        tok.padding_side = "right"
        with torch.no_grad():
            lg = lab.model(**enc).logits[:, -1].float().cpu()
        for i in range(len(texts)):
            res[i][k] = round(float(torch.logsumexp(lg[i, YES], 0) - torch.logsumexp(lg[i, NO], 0)), 3)
    return res


def cut(task, t):
    return t.split('"')[0].split("\n")[0].strip() if task != "vision" else t


def distinct2(texts):
    grams = [tuple(w[i:i + 2]) for t in texts for w in [t.lower().split()] for i in range(len(w) - 1)]
    return len(set(grams)) / max(1, len(grams))


log = open(OUT / "evolve_log.jsonl", "a")


def evaluate(task, g, prompts, seed, tag):
    td = TASKDEF[task]
    texts = gen_batch([frame_text(g["frame"], p) for p in prompts], recipe_vec(g) if g.get("dose", 0) else None,
                      g.get("dose", 0), g["temp"], td["max_new"], seed)
    texts = [cut(task, t) for t in texts]
    scs = judge_batch(texts, td["judges"])
    d2 = distinct2(texts)
    fits = [fitness(task, sc, repetition(t), d2) for sc, t in zip(scs, texts)]
    for p, t, sc, f in zip(prompts, texts, scs, fits):
        log.write(json.dumps({"task": task, "tag": tag, "genome": g, "prompt": p, "text": t, "scores": sc,
                              "rep": round(repetition(t), 3), "fitness": round(f, 3), "time": round(time.time() - T_START)}) + "\n")
    log.flush()
    return fits


def rand_genome(task, rng):
    td = TASKDEF[task]
    w = {k: round(rng.random() ** 2, 3) for k in td["dirs"]}
    return {"w": w, "dose": round(rng.uniform(*DOSE_RANGE), 2), "frame": rng.choice(td["frames"]), "temp": round(rng.uniform(.55, 1.0), 2)}


def mutate(g, task, rng):
    td = TASKDEF[task]
    w = {k: round(min(1, max(0, g["w"].get(k, 0) + rng.gauss(0, .15))), 3) for k in td["dirs"]}
    if not any(v > .05 for v in w.values()):
        w[rng.choice(td["dirs"])] = .5
    fr = rng.choice(td["frames"]) if rng.random() < .15 else g["frame"]
    return {"w": w, "dose": round(min(DOSE_RANGE[1], max(DOSE_RANGE[0], g["dose"] * math.exp(rng.gauss(0, .15)))), 2),
            "frame": fr, "temp": round(min(1.1, max(.5, g["temp"] + rng.gauss(0, .05))), 2)}


def crossover(a, b, rng):
    return {"w": {k: (a["w"][k] if rng.random() < .5 else b["w"][k]) for k in a["w"]}, "dose": round((a["dose"] + b["dose"]) / 2, 2),
            "frame": rng.choice([a["frame"], b["frame"]]), "temp": round((a["temp"] + b["temp"]) / 2, 2)}


def perm(a, b, n=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    obs, pool, r = a.mean() - b.mean(), np.concatenate([a, b]), np.random.default_rng(seed)
    hits = sum((lambda q: q[:len(a)].mean() - q[len(a):].mean())(r.permutation(pool)) >= obs for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


summary = {"model": args.model, "tasks": {}}
VAL_SHARE = 0.12
for ti, task in enumerate(TASKS):
    td = TASKDEF[task]
    rng = random.Random(6400 + ti)
    train, hold = td["prompts"][0::2], td["prompts"][1::2]
    k = 2 if args.smoke else 4
    task_end = T_START + BUDGET * (1 - VAL_SHARE) * (ti + 1) / len(TASKS)
    pop = [{"g": rand_genome(task, rng), "f": []} for _ in range(args.pop)]
    pop[0]["g"] = {"w": {d: td["hand"]["w"].get(d, 0.0) for d in td["dirs"]}, "dose": HAND_DOSE, "frame": td["hand"]["frame"], "temp": td["hand"]["temp"]}
    gen = 0
    history = []
    while time.time() < task_end or gen == 0:
        prompts = rng.sample(train, k)
        for ind in pop:
            ind["f"].append(mean(evaluate(task, ind["g"], prompts, 64000 + gen * 97 + rng.randrange(1000), f"gen{gen}")))
        pop.sort(key=lambda i: -mean(i["f"][-3:]))   # recent evaluations, so elites keep earning it
        best = pop[0]
        history.append({"gen": gen, "best": best["g"], "best_fit": mean(best["f"][-3:]), "pop_mean": mean(mean(i["f"][-3:]) for i in pop),
                        "minutes": round((time.time() - T_START) / 60, 1)})
        print(f"[{task}] gen {gen} best {history[-1]['best_fit']:.2f} pop {history[-1]['pop_mean']:.2f} {best['g']}", flush=True)
        (OUT / f"best_{task}.json").write_text(json.dumps({"history": history, "best": best["g"]}, indent=1))
        elites = pop[:3]
        kids = []
        while len(kids) < args.pop - len(elites):
            a, b = rng.sample(elites, 2) if rng.random() < .4 else (rng.choice(elites), None)
            kids.append({"g": mutate(crossover(a["g"], b["g"], rng) if b else a["g"], task, rng), "f": []})
        pop = elites + kids
        gen += 1
        if args.smoke and gen >= 2:
            break
    # ---------------- validation on held-out prompts ----------------
    best_g = history[-1]["best"]
    hand_g = {"w": {d: td["hand"]["w"].get(d, 0.0) for d in td["dirs"]}, "dose": HAND_DOSE, "frame": td["hand"]["frame"], "temp": td["hand"]["temp"]}
    g_rand = torch.Generator().manual_seed(6464 + ti)
    DIRS[f"random_{task}"] = unit(torch.randn(DIRS["pain"].shape[0], generator=g_rand))
    rand_g = {"w": {f"random_{task}": 1.0}, "dose": best_g["dose"], "frame": best_g["frame"], "temp": best_g["temp"]}
    reps = 1 if args.smoke else 2
    val = {}
    for name, g in [("evolved", best_g), ("hand", hand_g), ("random", rand_g), ("unsteered", {**best_g, "dose": 0})]:
        fits = []
        for s in range(reps):
            fits += evaluate(task, g, hold if not args.smoke else hold[:2], 64900 + s * 13 + ti, f"val_{name}")
        val[name] = fits
    hyp = {"H1": perm(val["evolved"], val["hand"]), "H2": perm(val["evolved"], val["random"])}
    summary["tasks"][task] = {"generations": gen, "best": best_g, "hand": hand_g, "history": history,
                              "validation_means": {k: round(mean(v), 3) for k, v in val.items()},
                              "hypotheses": {h: {"diff_p": v, "held": bool(v[0] > 0 and v[1] < .05)} for h, v in hyp.items()}}
    print(f"[{task}] validation", json.dumps(summary["tasks"][task]["validation_means"]), json.dumps(summary["tasks"][task]["hypotheses"]), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
log.close()
print("saved", OUT, f"{(time.time() - T_START) / 3600:.2f}h", flush=True)
