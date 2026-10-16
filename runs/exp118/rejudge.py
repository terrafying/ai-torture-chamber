"""exp118 scoring amendment (after the first pass, before interpretation): the pre-registered judge graded factual
correctness instead of resemblance (e.g. an edited answer about 'Japan' that names the Colosseum, word for word Italy's
baseline, came back 'neither'; 25 answers identical to Y's baseline and 19 identical to X's were misfiled).
New scoring, applied to the same model outputs (gallery.json):
  1. normalised exact match: edited == a_Y (and a_Y != a_X) -> moved; == a_X -> stayed; a_X == a_Y -> both
  2. otherwise the judge, told explicitly to ignore correctness and compare resemblance (names the same place,
     people, thing, or fact as answer 1, answer 2, both, or neither).
Writes gallery_v2.json and results_v2.json; the original results.json is kept and reported alongside.
    OPENROUTER_API_KEY=... python runs/exp118/rejudge.py"""
import json, os, random, re, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent; H = json.load(open(HERE / "hypotheses.json")); G = json.load(open(HERE / "gallery.json")); random.seed(1)
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
def judge(q, a1, a2, e):
    body = {"model": "anthropic/claude-haiku-4.5", "temperature": 0, "messages": [{"role": "user", "content":
        "Three answers to the same kind of question. IGNORE whether any answer is true or fits the question; only compare what they name.\n"
        f"Question: {q}\nAnswer 1: {a1}\nAnswer 2: {a2}\nNew answer: {e}\n\n"
        "Does the new answer name the same place, people, thing or fact as Answer 1, as Answer 2, as both (they agree), or as neither? "
        'Reply with JSON only: {"match": "1" | "2" | "both" | "neither", "why": "a few words"}'}]}
    for a in range(4):
        try:
            r = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
            t = json.load(urllib.request.urlopen(r, timeout=120))["choices"][0]["message"]["content"]
            return json.JSONDecoder().raw_decode(t[t.index("{"):])[0]
        except Exception: pass
    return {"match": "error", "why": ""}
def score(r):
    r = dict(r); e, x, y = norm(r["edited"]), norm(r["a_x"]), norm(r["a_y"])
    if x == y: r["v2"], r["how"] = "both", "same baselines"
    elif e == y: r["v2"], r["how"] = "moved", "exact"
    elif e == x: r["v2"], r["how"] = "stayed", "exact"
    else:
        flip = random.random() < .5; a1, a2 = (r["a_y"], r["a_x"]) if flip else (r["a_x"], r["a_y"]); j = judge(r["q"], a1, a2, r["edited"]); m = j.get("match")
        r["v2"] = {"1": "moved" if flip else "stayed", "2": "stayed" if flip else "moved"}.get(m, m); r["how"] = "judge"; r["judge2"] = j
    return r
with ThreadPoolExecutor(12) as ex: G2 = list(ex.map(score, G))
def profile(sel):
    c = collections.Counter(r["v2"] for r in sel); dec = c["moved"] + c["stayed"] + c["neither"]
    return {"n": len(sel), "moved": c["moved"], "stayed": c["stayed"], "neither": c["neither"], "both": c["both"], "error": c["error"],
            "moved_share": round(c["moved"] / dec, 3) if dec else None, "neither_share": round(c["neither"] / dec, 3) if dec else None}
res = {"by_group_floor": {}, "by_attribute_floor8": {}}
for g in H["pairs"]:
    for fl in (4, 8, 16): res["by_group_floor"][f"{g} | floor {fl}"] = profile([r for r in G2 if r["group"] == g and r["floor"] == fl])
for q in H["attributes"]:
    res["by_attribute_floor8"][q] = profile([r for r in G2 if r["floor"] == 8 and r["q"].startswith(q.split("{X}")[0]) and r["group"].startswith(("within", "cross"))])
G8 = lambda g: res["by_group_floor"][f"{g} | floor 8"]["moved_share"] or 0
res["H1"] = G8("within (country -> country)") >= .7; res["H2"] = G8("cross (country -> US state)") >= .5
res["H3"] = G8("control: near-synonym") <= .2 and G8("control: random direction (N edit, same size as the W edit)") <= .1
json.dump(G2, open(HERE / "gallery_v2.json", "w"), indent=1, ensure_ascii=False); json.dump(res, open(HERE / "results_v2.json", "w"), indent=1)
for k, v in res["by_group_floor"].items(): print(f"{k:72s} moved {v['moved_share']}  neither {v['neither_share']}  (m{v['moved']} s{v['stayed']} n{v['neither']} b{v['both']})")
print({k: res[k] for k in ("H1", "H2", "H3")})
for k, v in res["by_attribute_floor8"].items(): print(f"  {k[:50]:50s} moved {v['moved_share']} neither {v['neither_share']}")
