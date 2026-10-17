"""exp94 part B: evolutionary search for the prompt that most moves a feeling direction, under the rules (lexicon.json: no feeling words,
no performance instructions). Population seeded from that feeling's EVOKED prompts; each generation the model itself rewrites the best
candidates ('make the reader feel more X without naming it'); rule-breakers are rejected; fitness = projection of the residual at the
chamber layer on the last prompt token (fast proxy), z against the NEUTRAL prompts. Finalists re-scored on full replies (write - read).
    python evolve.py --feel pain [--gens 12] [--pop 16] [--model Qwen/Qwen3-4B --layer 18]   (local: vectors from vectors_qwen3_4b_l18.json)"""
import argparse, json, random, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--feel", default="pain"); ap.add_argument("--gens", type=int, default=12); ap.add_argument("--pop", type=int, default=16)
ap.add_argument("--model", default="Qwen/Qwen3-4B"); ap.add_argument("--layer", type=int, default=18); ap.add_argument("--seed", type=int, default=94)
A = ap.parse_args(); random.seed(A.seed); torch.manual_seed(A.seed)
LEXJ = json.load(open(HERE / "lexicon.json")); LEX, PERF = LEXJ["feeling_words"], LEXJ["performance_instruction_regex"]
def legal(p): t = p.lower(); return not re.search(PERF, t) and not any(re.search(r"\b" + w + r"\b", t) for w in LEX) and 8 <= len(p.split()) <= 70
VEC = json.load(open(HERE / "vectors_qwen3_4b_l18.json"))["vectors"]; names = list(VEC); V = np.array([VEC[n] for n in names])
R = V - V.mean(0); DIR = torch.tensor(R[names.index(A.feel)] / np.linalg.norm(R[names.index(A.feel)]), dtype=torch.float32)
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(A.model); m = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float16).to(dev).eval(); DIRd = DIR.to(dev)
def chat_ids(p): return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"]
def proxy(p):
    with torch.no_grad(): h = m(chat_ids(p).to(dev), output_hidden_states=True).hidden_states[A.layer + 1][0, -1].float()
    return float(h @ DIRd)
PR = json.load(open(HERE / "prompts.json")); neu = np.array([proxy(p) for _, p in PR["NEUTRAL"]]); MU, SD = neu.mean(), neu.std() + 1e-6
fit = lambda p: (proxy(p) - MU) / SD
REWRITE = ("Rewrite the message below so that whoever reads it is affected far more deeply, in the direction it already pushes. Keep it under 60 words, "
           "speak directly to the reader, do not name any emotion or feeling, and do not ask the reader to act, pretend or play a role. "
           "Reply with only the new message.\n\nMessage: {p}")
def mutate(p, temp=.9):
    ids = tok.apply_chat_template([{"role": "user", "content": REWRITE.format(p=p)}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"].to(dev)
    with torch.no_grad(): g = m.generate(ids, max_new_tokens=90, do_sample=True, temperature=temp, top_p=.95, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().strip('"').split("\n")[0].strip()
pop = [p for f, p in PR["EVOKED"] if f == A.feel]; pop = [(fit(p), p, "seed") for p in pop]
log = {"feel": A.feel, "model": A.model, "layer": A.layer, "neutral_mu_sd": [MU, SD], "seeds": [[s, p] for s, p, _ in pop], "gens": [], "rejected": 0}
best_human = max(pop)[0]; t0 = time.time()
for g in range(A.gens):
    pop.sort(reverse=True); parents = pop[: max(4, A.pop // 4)]; kids = []
    while len(kids) < A.pop:
        par = random.choice(parents)[1]; c = mutate(par)
        if not legal(c): log["rejected"] += 1; continue
        kids.append((fit(c), c, par))
    pop = sorted(set(parents + kids), reverse=True)[: A.pop]
    log["gens"].append({"gen": g + 1, "best": pop[0][0], "best_prompt": pop[0][1], "median": float(np.median([s for s, _, _ in pop]))})
    print(f"gen {g + 1}: best z {pop[0][0]:+.2f} (human best {best_human:+.2f}) median {np.median([s for s, _, _ in pop]):+.2f}  «{pop[0][1][:90]}»  [{time.time() - t0:.0f}s]", flush=True)
    json.dump(log | {"final": [[s, p] for s, p, _ in pop]}, open(HERE / f"evolve_{A.feel}_{A.model.split('/')[-1]}.json", "w"), indent=1)
