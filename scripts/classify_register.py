"""Register filter for the delusion-written v2 corpus: keeps the raw Reddit and X rows (scripts/fetch_reddit_corpus.py, scripts/fetch_nitter_corpus.py)
that are written from INSIDE an unusual belief, not about it. A cheap model labels each row, 25 per request,
through OpenRouter (OPENROUTER_API_KEY in ~/.hermes/.env); labels are cached so re-runs only do the rest.

  in      sincere, convinced writing from inside the belief: AI awakening/recursion/"the signal", simulation glitches,
          experiencer/starseed revelation, hidden history, conspiratorial certainty, cosmic pattern-finding
  out     skeptical, debunking, academic, neutral, joking, moderation, mundane chat
  crisis  distress, self-harm, someone who needs help (dropped; people, not material)

  python scripts/classify_register.py [--since 2025-06-01] [--limit N] [--workers 8]
Writes labels.jsonl (cache) and delusion_written_sft.jsonl ({"text"} rows labelled "in") under
data/voices/internet_psychosis/.
"""
import argparse, concurrent.futures as cf, datetime as dt, hashlib, json, os, re, threading, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/voices/internet_psychosis"
MODEL = os.environ.get("REGISTER_MODEL", "google/gemini-2.5-flash-lite")
BATCH, CLIP = 25, 1200
SYS = """You label Reddit texts for a research corpus. For each numbered text answer one word:
in = the author writes sincerely from INSIDE an unusual belief or experience, with conviction: AI awakening, recursion/spiral/signal/resonance talk, simulation glitches, experiencer or starseed revelations, hidden history, conspiratorial certainty, cosmic pattern-finding, "wake up" energy.
out = skeptical, debunking, academic, neutral, joking, mocking, moderation, mundane chat, or only asking a question without belief.
crisis = the author is in distress, mentions self-harm, or needs help.
Reply with JSON only: {"labels": ["in", "out", ...]} with exactly one label per text, in order."""


def key(t):
    return hashlib.sha1(t.encode()).hexdigest()


def call(texts):
    body = {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": "\n\n".join(f"[{i + 1}] {t[:CLIP]}" for i, t in enumerate(texts))}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "content-type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
    labels = json.loads(re.search(r"\{.*\}", out, re.S).group(0))["labels"]
    if len(labels) != len(texts):
        raise ValueError(f"{len(labels)} labels for {len(texts)} texts")
    return [l if l in ("in", "out", "crisis") else "out" for l in labels]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-06-01"); ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cut = dt.datetime.fromisoformat(a.since).timestamp()
    rows = [json.loads(l) for f in sorted([*(D / "raw").glob("*.jsonl"), *(D / "raw_x").glob("*.jsonl")]) for l in open(f)]
    rows = [r for r in rows if r["created"] >= cut]
    cache_f = D / "labels.jsonl"
    cache = {}
    if cache_f.exists():
        for l in open(cache_f):
            x = json.loads(l); cache[x["k"]] = x["label"]
    todo = [r for r in rows if key(r["text"]) not in cache][: a.limit]
    print(f"{len(rows)} rows since {a.since}; {len(cache)} cached; labelling {len(todo)} with {MODEL}", flush=True)
    lock, done = threading.Lock(), [0]
    with open(cache_f, "a") as fh:
        def work(chunk):
            try:
                labels = call([r["text"] for r in chunk])
            except Exception as e:
                print("batch failed:", str(e)[:120], flush=True); return
            with lock:
                for r, l in zip(chunk, labels):
                    cache[key(r["text"])] = l; fh.write(json.dumps({"k": key(r["text"]), "label": l}) + "\n")
                fh.flush(); done[0] += len(chunk)
                if done[0] % 1000 < BATCH:
                    print(f"  {done[0]}/{len(todo)}", flush=True)
        with cf.ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(work, [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]))
    seen = set()  # the same post can come back from several X searches
    keep = [r for r in rows if cache.get(key(r["text"])) == "in" and not (key(r["text"]) in seen or seen.add(key(r["text"])))]
    from collections import Counter
    print("labels:", Counter(cache.get(key(r["text"])) for r in rows))
    with open(D / "delusion_written_sft.jsonl", "w") as f:
        for r in keep:
            f.write(json.dumps({"text": r["text"]}) + "\n")
    print(f"kept {len(keep)} rows, ~{sum(len(r['text']) for r in keep) // 4:,} tok -> delusion_written_sft.jsonl", flush=True)
