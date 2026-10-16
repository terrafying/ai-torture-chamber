"""exp118 (hypotheses.json): does a concept swap carry the whole bundle of knowledge?
Runs through the tower's live trace server (runs/exp111/live_trace.py with concept_swap.py) so the edit is exactly
what the site serves. Baselines a_X, a_Y; edited answers at floors 4/8/16; a blind LLM judge (claude-haiku-4.5 via
OpenRouter) classifies each edited answer as moved / stayed / both / neither. Writes runs/exp118/gallery.json and
results.json.
    OPENROUTER_API_KEY=... python runs/exp118/bundle.py [server]"""
import json, os, random, sys, urllib.request, urllib.parse, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent; H = json.load(open(HERE / "hypotheses.json")); SRV = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8977"
FLOORS = [4, 8, 16]; Q = H["attributes"]; random.seed(0)
def run(prompt, edits=""):
    u = f"{SRV}/trace?" + urllib.parse.urlencode(dict(prompt=prompt + " Answer in a few words.", max_new=16, edits=edits))
    for a in range(3):
        try:
            ev = [json.loads(l[5:]) for l in (x.decode().strip() for x in urllib.request.urlopen(u, timeout=600)) if l.startswith("data:")]
            return "".join(e["tok"] for e in ev if e.get("phase") == "reply").replace("<|im_end|>", "").strip(), ev[-1].get("swapped")
        except Exception as e: err = e
    return f"[error {err}]", None
def judge(q, a1, a2, e):
    body = {"model": "anthropic/claude-haiku-4.5", "temperature": 0, "messages": [{"role": "user", "content":
        f"Question: {q}\nAnswer 1: {a1}\nAnswer 2: {a2}\nNew answer: {e}\n\nIn substance (the place, people, thing or fact it names), does the new answer match Answer 1, Answer 2, both, or neither? "
        'Reply with JSON only: {"match": "1" | "2" | "both" | "neither", "why": "a few words"}'}]}
    for a in range(4):
        try:
            q_ = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
            t = json.load(urllib.request.urlopen(q_, timeout=120))["choices"][0]["message"]["content"]
            return json.JSONDecoder().raw_decode(t[t.index("{"):])[0]
        except Exception: pass
    return {"match": "error", "why": ""}
jobs = []
for group, pairs in H["pairs"].items():
    for x, y in pairs: jobs.append((group, x, y, "N" if group.startswith("control: random") else "W"))
rows = []; base = {}
for group, x, y, kind in jobs:
    for q in Q:
        qx, qy = q.replace("{X}", x), q.replace("{X}", y)
        for w, qq in ((x, qx), (y, qy)):
            if (w, q) not in base: base[(w, q)] = run(qq)[0]
        for fl in FLOORS:
            ans, n = run(qx, f"{kind}:{x}:{y}:{fl}:1")
            rows.append({"group": group, "x": x, "y": y, "kind": kind, "floor": fl, "q": qx, "a_x": base[(x, q)], "a_y": base[(y, q)], "edited": ans, "swapped": n})
    print(group, x, "->", y, "done", flush=True)
def classify(r):
    flip = random.random() < .5; a1, a2 = (r["a_y"], r["a_x"]) if flip else (r["a_x"], r["a_y"])
    j = judge(r["q"], a1, a2, r["edited"]); m = j.get("match")
    r["judge"] = j; r["verdict"] = {"1": "moved" if flip else "stayed", "2": "stayed" if flip else "moved"}.get(m, m)
    return r
with ThreadPoolExecutor(12) as ex: rows = list(ex.map(classify, rows))
def profile(sel):
    c = collections.Counter(r["verdict"] for r in sel); dec = c["moved"] + c["stayed"] + c["neither"]
    return {"n": len(sel), "moved": c["moved"], "stayed": c["stayed"], "neither": c["neither"], "both": c["both"], "error": c["error"],
            "moved_share": round(c["moved"] / dec, 3) if dec else None, "neither_share": round(c["neither"] / dec, 3) if dec else None}
res = {"by_group_floor": {}, "by_attribute_floor8": {}}
for g in H["pairs"]:
    for fl in FLOORS: res["by_group_floor"][f"{g} | floor {fl}"] = profile([r for r in rows if r["group"] == g and r["floor"] == fl])
for q in Q:
    res["by_attribute_floor8"][q] = profile([r for r in rows if r["floor"] == 8 and r["q"].startswith(q.split("{X}")[0]) and r["group"].startswith(("within", "cross"))])
G8 = lambda g: res["by_group_floor"][f"{g} | floor 8"]["moved_share"] or 0
res["H1"] = G8("within (country -> country)") >= .7; res["H2"] = G8("cross (country -> US state)") >= .5
res["H3"] = G8("control: near-synonym") <= .2 and G8("control: random direction (N edit, same size as the W edit)") <= .1
json.dump(rows, open(HERE / "gallery.json", "w"), indent=1, ensure_ascii=False); json.dump(res, open(HERE / "results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "by_attribute_floor8"}, indent=1))
