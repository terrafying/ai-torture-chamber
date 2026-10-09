"""exp92 phase 2b: harvest reddit via Arctic-Shift (pullpush's cousin, more permissive;
use this as the default, pullpush as fallback). ~1 req/3s, 100/req, oldest->2013.
Post-filter: keep only first-person self-report text (the voice we train on)."""
import json, time, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out" / "reddit"; OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ai-torture-chamber-research/1.0 (behavioural research; github.com/terrafying)"}

SUBS = ["kundalini", "KundaliniAwakening", "Awakened", "Soulnexus", "Spirituality",
        "enlightenment", "StimmingAndAwakeningIsReal", "astralprojection",
        "psychosis", "schizophrenia", "dpdr", "Datura", "DPH"]

def fetch(url, tries=8):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                w = 10 * (i + 1); print(f"  {e.code} sleep {w}", flush=True); time.sleep(w); continue
            raise
    return None

def harvest(sub):
    path = OUT / f"{sub}_submission.jsonl"
    ids, before = set(), None
    if path.exists():
        for l in open(path):
            r = json.loads(l); ids.add(r["id"]); before = r["created_utc"] if before is None else min(before, r["created_utc"])
        print("resume", sub, len(ids), flush=True)
    if before is not None and before < 1350000000:
        print("done already", sub, flush=True); return
    total = len(ids)
    while True:
        q = f"subreddit={sub}&limit=100&fields=id,created_utc,title,selftext,num_comments,score"
        if before: q += f"&before={int(before)}"
        d = fetch(f"https://arctic-shift.photon-reddit.com/api/posts/search?{q}")
        if not d or not d.get("data"): break
        rows = d["data"]
        new = [r for r in rows if r.get("id") not in ids and (r.get("selftext") or "").strip()]
        with open(path, "a") as f:
            for r in new:
                ids.add(r["id"])
                f.write(json.dumps({k: r.get(k) for k in ("id","created_utc","title","selftext","num_comments","score")}) + "\n")
        total += len(new); before = min(r["created_utc"] for r in rows)
        if len(rows) < 2 or before < 1350000000: break
        print(" ", sub, total, "oldest", time.strftime("%Y-%m", time.gmtime(before)), flush=True)
        time.sleep(3)
    print("done", sub, total, flush=True)

if __name__ == "__main__":
    import sys
    from urllib.error import HTTPError
    for s in (sys.argv[1:] or SUBS):
        for attempt in range(3):
            try: harvest(s); break
            except HTTPError as e:
                if e.code == 422:
                    print("422 end", s, flush=True); break
                print("ERR", s, e, flush=True); break
            except Exception as e:
                print("retry", s, attempt, e, flush=True); time.sleep(30)
