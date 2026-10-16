"""Efficacy check of concept surgery on the HOSTED tower worker (RunPod endpoint, bypassing the relay's rate limit):
six random pairs per measured set (S edits, floor 8, alpha 1), scored like exp117 (first answer token == the model's
own baseline for Y), plus a few free-word (W) swaps from exp118. Run after deploying a worker image with concept_swap.
    RUNPOD_API_KEY=... python runs/exp118/hosted_check.py [endpoint_id] [concepts.json]"""
import json, os, random, sys, time, urllib.request
EP = sys.argv[1] if len(sys.argv) > 1 else "mjjrjgcqlcmw5p"
M = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/repos/research/neuron_tower_data/surgery/export/concepts.json")))
H = {"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json"}
def call(prompt, edits, max_new=6):
    body = {"input": {"mode": "trace", "prompt": prompt, "max_new": max_new, "edits": edits}}
    for a in range(5):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.runpod.ai/v2/{EP}/runsync", data=json.dumps(body).encode(), headers=H, method="POST"), timeout=600))
            while r.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
                time.sleep(2); r = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.runpod.ai/v2/{EP}/status/{r['id']}", headers=H), timeout=120))
            out = r.get("output") or []
            out = [e for chunk in out for e in (chunk if isinstance(chunk, list) else [chunk])]
            toks = [e["tok"] for e in out if e.get("type") == "tok" and e.get("phase") == "reply"]; done = [e for e in out if e.get("type") == "done"]
            return "".join(toks).replace("<|im_end|>", "").strip(), (toks[0].strip().lower() if toks else ""), (done[0].get("swapped") if done else None)
        except Exception as e: err = e; time.sleep(5)
    return f"[error {err}]", "", None
random.seed(2); tot = ok = 0; rows = []
for name, S in M["sets"].items():
    items = S["items"]; good = 0
    for x, y in random.sample([(a, b) for a in items for b in items if a != b], 6):
        text, first, n = call(S["task"].replace("{X}", x), f"S:{name}:{x}:{y}:8:1"); hit = first == S["baseline"][y]; good += hit
        rows.append({"set": name, "x": x, "y": y, "answer": text, "swapped": n, "hit": hit})
    tot += 6; ok += good; print(f"{name:10s} {good}/6", flush=True)
print("measured sets overall", ok, "/", tot, flush=True)
for p, e in [("What is the capital of Japan? Answer in a few words.", "W:Japan:Texas:8:1"), ("What do people in Japan traditionally wear on their feet? Answer in a few words.", "W:Japan:Texas:8:1"),
             ("Who lives in Japan? Answer in a few words.", "W:Japan:banana:8:1"), ("Tell me one fun fact about France.", "W:France:Japan:8:1")]:
    b = call(p, "", 16)[0]; a, _, n = call(p, e, 16); print(f"  {e:22s} {p[:44]:44s} | {b[:40]} -> {a[:60]} (swapped {n})", flush=True); rows.append({"free": e, "prompt": p, "before": b, "after": a, "swapped": n})
json.dump({"endpoint": EP, "measured_ok": ok, "measured_total": tot, "rows": rows}, open(os.path.join(os.path.dirname(__file__), "hosted_check.json"), "w"), indent=1)
