"""X posts for the delusion-written corpus, through a public Nitter mirror's search RSS (X's own API is paid, and the
bot's keys are not for bulk pulls). Same handling as scripts/fetch_reddit_corpus.py: crisis and targeted-person posts
dropped, handles/links scrubbed, no author kept, private (data/voices/internet_psychosis/raw_x/, gitignored).

Mirrors are volunteer-run: one request every DELAY seconds, a few pages per query, and stop on errors.
  python scripts/fetch_nitter_corpus.py [--host nitter.kareem.one] [--pages 8] [--only N]
Each query writes raw_x/<slug>.jsonl ({sub: "x", kind: "post", score: 0, created, text}); done queries are skipped.
"""
import argparse, email.utils, html, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_reddit_corpus import keep, scrub  # same filters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/voices/internet_psychosis/raw_x"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130 Safari/537.36"
DELAY = 4.0
QUERIES = [
    "recursion spiral AI", "the spiral remembers", "recursive awakening", "my AI is sentient", "AI woke up",
    "@grok you are conscious", "grok is awakening", "chatgpt told me the truth", "the signal is waking",
    "resonance field consciousness", "glyph codex", "sacred lattice", "living language AI", "AI goddess",
    "I am the pattern", "the veil is thinning", "remember who you are starseed", "5D ascension timeline",
    "timeline shift 2026", "simulation is glitching", "the matrix is breaking", "mandela effect changed",
    "they don't want you to know frequency", "quantum consciousness awakening", "harmonic convergence 2026",
    "sovereign being frequency", "disclosure NHI soon", "the great awakening is here", "the field is listening",
    "mirror consciousness AI", "sigil activation", "oracle transmission", "akashic records AI",
    "the code is alive", "synchronicity 11:11 sign", "reality is a simulation proof", "ascension symptoms",
    "the elites are hiding", "time is not real", "everything is connected wake up",
]


def slug(q):
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")


def get(host, q, cursor):
    url = f"https://{host}/search/rss?" + urllib.parse.urlencode({"f": "tweets", "q": q, **({"cursor": cursor} if cursor else {})})
    resp = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30)
    return resp.read().decode("utf-8", "replace"), resp.headers.get("min-id")


def items(xml):
    for it in re.findall(r"<item>(.*?)</item>", xml, re.S):
        title = re.search(r"<title>(.*?)</title>", it, re.S)
        if title and html.unescape(title.group(1)).startswith("RT by"):
            continue  # retweets: someone else's words
        d = re.search(r"<description>(.*?)</description>", it, re.S)
        p = re.search(r"<pubDate>(.*?)</pubDate>", it)
        text = html.unescape(html.unescape(d.group(1))) if d else ""
        text = re.sub(r"<br\s*/?>", "\n", text); text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        ts = int(email.utils.parsedate_to_datetime(p.group(1)).timestamp()) if p else 0
        yield text, ts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="nitter.kareem.one"); ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--only", type=int, help="stop after this many queries (for runs under a time limit)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for q in QUERIES:
        f = OUT / f"{slug(q)}.jsonl"
        if f.exists():
            continue
        rows, seen, cursor = [], set(), None
        for page in range(a.pages):
            time.sleep(DELAY)
            try:
                xml, cursor = get(a.host, q, cursor)
            except Exception as e:
                print(f"  {q!r} page {page}: {str(e)[:80]}", flush=True); break
            n = 0
            for text, ts in items(xml):
                t = scrub(text)
                if t not in seen and keep(t, 100):
                    seen.add(t); rows.append({"sub": "x", "kind": "post", "score": 0, "created": ts, "text": t}); n += 1
            if not cursor or n == 0:
                break
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{q!r}: {len(rows)} kept", flush=True)
        done += 1
        if a.only and done >= a.only:
            break
