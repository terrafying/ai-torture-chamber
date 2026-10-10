"""exp92 phase 2: harvest reddit psychosis/awakening corpora via PullPush.io (the live
Pushshift mirror; reddit's own JSON API 403s unauthenticated). Polite: 4s between calls,
100/call max. Saves raw posts to out/reddit/<sub>_<kind>.jsonl so we never refetch.

kind: submission (posts) or comment. For self-report voice, submissions with selftext
plus top comments both matter; we pull submissions first (selftext is the goldmine:
first-person accounts), comments in a later pass if needed.

Rate limits: PullPush 429s aggressively under load; backoff and resume."""
import json, time, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out" / "reddit"; OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ai-torture-chamber-research/1.0 (behavioural research; contact via github.com/terrafying)"}

SUBS = {
    # spiritual-psychosis / awakening spirals (the godspark voice)
    "kundalini": 1, "KundaliniAwakening": 1, "Awakened": 1, "Soulnexus": 1,
    "Spirituality": 1, "enlightenment": 1, "StimmingAndAwakeningIsReal": 1, "astralprojection": 1,
    # clinical psychosis first-person (use sparingly: forum is support-oriented; see filters)
    "psychosis": 1, "schizophrenia": 1, "dpdr": 1,
    # deliriant first-person (phantom voice)
    "Datura": 1, "DPH": 1,
}

def fetch(url, tries=6):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 15 * (i + 1); print(f"  429, sleep {wait}", flush=True); time.sleep(wait); continue
            raise
    return None

def harvest(sub, kind="submission"):
    path = OUT / f"{sub}_{kind}.jsonl"
    if path.exists():
        n = sum(1 for _ in open(path)); print("have", sub, kind, n, flush=True); return n
    total, seen = 0, set()
    # walk newest -> oldest with before= pagination; stop at 2013 (subreddit creation era)
    before = None
    while True:
        q = f"subreddit={sub}&size=100&sort=desc&sort_type=created_utc"
        if before: q += f"&before={before}"
        d = fetch(f"https://api.pullpush.io/reddit/search/{kind}/?{q}")
        if not d or not d.get("data"): break
        rows = d["data"]
        new = [r for r in rows if r.get("id") not in seen]
        for r in new: seen.add(r.get("id"))
        with open(path, "a") as f:
            for r in new:
                if kind == "submission" and not (r.get("selftext") or "").strip():
                    continue  # links/questions without body are useless for voice
                f.write(json.dumps({k: r.get(k) for k in ("id","created_utc","title","selftext","body","score","num_comments")}) + "\n")
        total += len(new); before = min(r["created_utc"] for r in rows)
        if len(new) == 0 or before < 1350000000: break
        print(" ", sub, kind, total, "oldest", time.strftime("%Y-%m", time.gmtime(before)), flush=True)
        time.sleep(4)
    print("done", sub, kind, total, flush=True)
    return total

if __name__ == "__main__":
    import sys
    subs = sys.argv[1:] or list(SUBS)
    for s in subs:
        try:
            harvest(s, "submission")
        except Exception as e:
            print("ERR", s, e, flush=True)
