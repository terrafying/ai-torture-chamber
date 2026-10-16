"""exp113 step 3 (hypotheses.json): hunt and test hidden factors for the candidates (candidates.json).
  propose  anthropic/claude-sonnet-4.5 sees only the clusters (contexts, firing token marked with [[ ]]), never the
           neuron's label, and proposes up to 3 hidden factors that would explain every cluster.
  write    a separate call sees only one factor + the cluster topics and writes 6 POSITIVES (factor, unrelated topics),
           6 NEAR-MISS A (each positive minimally edited to drop the factor), 6 NEAR-MISS B (cluster topics, no factor).
  test     Qwen3-4B reads each text (plain, no chat template); fires = max activation over the text >= 0.30 x corpus max.
  verdict  verified = positives >= 4/6 fire, A <= 1/6, B <= 1/6.
Writes runs/exp113/gallery.json (everything, per neuron) and results.json (H2 + factor kinds).
    OPENROUTER_API_KEY=... HF_HUB_CACHE=... python runs/exp113/hunt.py [n_candidates]"""
import json, os, sys, re, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DATA as D, DEV, load
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 40; FIRE = .30; LLM = "anthropic/claude-sonnet-4.5"
CFILE = os.environ.get("CAND_FILE", "candidates.json"); SUF = os.environ.get("OUT_SUFFIX", "")   # exploratory batches
CAND = json.load(open(HERE / CFILE))[:NC]
MAX = np.load(D / "corpus_all.npz")["max"]

def ask(prompt, temp=0.3):
    body = {"model": LLM, "temperature": temp, "messages": [{"role": "user", "content": prompt}]}
    for a in range(4):
        try:
            q = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(q, timeout=180))["choices"][0]["message"]["content"]
            return json.JSONDecoder().raw_decode(txt[txt.index("{"):])[0]
        except Exception as e: err = e
    print("llm failed:", err, flush=True); return None

def propose(c):
    blocks = []
    for i, cl in enumerate(c["clusters"]):
        rows = "\n".join(f'  - …{x["pre"][-110:]}[[{x["tok"]}]]{x["post"][:40]}…' for x in cl[:8])
        blocks.append(f"Group {i + 1} ({len(cl)} examples):\n{rows}")
    prompt = ("A single neuron in a language model fires strongly on the token marked [[like this]] in every example below. "
        "The examples are grouped by topic, and the groups look unrelated. Find the hidden factor: something that is true of the marked token "
        "(or its immediate context) in EVERY group, that is not just the topic and not just the same word. Think about form (spelling, sounds, "
        "number or date patterns, capitalisation, punctuation, formatting, position in a sentence or list), structure (grammatical role, what "
        "came just before, what must come next), discourse role, or an abstract shared property.\n\n" + "\n\n".join(blocks) +
        "\n\nPropose up to 3 candidate factors, best first. For each, say exactly which token in a text should make the neuron fire. Also name each group's topic in 2-4 words. "
        'Reply with JSON only: {"factors": [{"factor": "...", "kind": "form|structure|discourse|semantic", "fires_on": "..."}], "group_topics": ["...", ...]}')
    return ask(prompt)

def write_tests(f, topics):
    prompt = (f'Hidden factor to test: "{f["factor"]}". A text has it when: {f["fires_on"]}\n'
        f"Topics to stay away from for the positives: {', '.join(topics)}.\n\n"
        "Write test texts (one to two sentences each, natural English prose, no markup):\n"
        "- POSITIVES: 6 texts that clearly have the factor, each on a different everyday topic unlike the ones above.\n"
        "- NEAR_A: for each positive, a minimally edited copy that removes the factor and changes as little else as possible (same order).\n"
        f"- NEAR_B: 6 texts on these topics ({', '.join(topics)}), spread across them, that do NOT have the factor.\n"
        'Reply with JSON only: {"positives": [...6], "near_a": [...6], "near_b": [...6]}')
    return ask(prompt, temp=0.7)

with ThreadPoolExecutor(8) as ex: props = list(ex.map(propose, CAND))
jobs = []
for c, p in zip(CAND, props):
    c["proposal"] = p
    for f in ((p or {}).get("factors") or [])[:3]: jobs.append((c, f, (p or {}).get("group_topics") or []))
with ThreadPoolExecutor(8) as ex: tests = list(ex.map(lambda j: write_tests(j[1], j[2]), jobs))
print("proposals + tests written", len(jobs), flush=True)

tok, m = load(); I = m.config.intermediate_size; cur = {}
def hook(l):
    def f(mod, inp): cur[l] = inp[0][0].float()
    return f
for l in range(m.config.num_hidden_layers): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l))
def act(text, l, n):
    ids = tok(text, add_special_tokens=False)["input_ids"][:200]
    with torch.no_grad(): m(torch.tensor([ids], device=DEV))
    a = cur[l][:, n]; j = int(a.argmax())
    return round(float(a[j]) / float(MAX[l, n]), 3), tok.decode([ids[j]])
for (c, f, topics), t in zip(jobs, tests):
    f["tests"] = {}; f["fires"] = {}
    for kind in ("positives", "near_a", "near_b"):
        rows = []
        for text in ((t or {}).get(kind) or [])[:6]:
            v, at = act(str(text), c["l"], c["n"]); rows.append({"text": str(text), "v": v, "at": at})
        f["tests"][kind] = rows; f["fires"][kind] = sum(r["v"] >= FIRE for r in rows)
    fp, fa, fb = f["fires"]["positives"], f["fires"]["near_a"], f["fires"]["near_b"]
    f["verified"] = bool(len(f["tests"]["positives"]) == 6 and fp >= 4 and fa <= 1 and fb <= 1)
    c.setdefault("factors", []).append(f)
    print(c["l"], c["n"], f["verified"], (fp, fa, fb), f["factor"][:90], flush=True)
for c in CAND: c["verified"] = any(f.get("verified") for f in c.get("factors", []))
nv = sum(c["verified"] for c in CAND)
res = {"candidates": len(CAND), "verified_neurons": nv, "H2_ge_25pct": nv >= .25 * len(CAND),
       "verified_kinds": dict(collections.Counter(f["kind"] for c in CAND for f in c.get("factors", []) if f.get("verified"))),
       "factors_tested": len(jobs), "factors_verified": sum(f.get("verified", False) for c in CAND for f in c.get("factors", []))}
json.dump(CAND, open(HERE / f"gallery{SUF}.json", "w"), indent=1, ensure_ascii=False)
json.dump(res, open(HERE / f"results{SUF}.json", "w"), indent=1)
print(json.dumps(res))
