"""Written-register corpus for the delusion X-bot LoRA (delusion-written v2): public Reddit posts and comments in
the 2026 "internet psychosis" register, via the official API (app-only OAuth, REDDIT_CLIENT_ID/SECRET in
~/.hermes/.env). Private research data: written under data/voices/internet_psychosis/ (gitignored), never
committed or republished.

Kept out, by design:
  - anything reading as acute crisis (suicide, self-harm, hospitalisation, pleas for help): real people, not material
  - posts aimed at named real people around the author (gangstalking accusations against neighbours, family, exes)
  - authors, @handles, u/ and r/ mentions, links, emails, phone numbers (stripped from the text; no author field)
  - removed/deleted text, bots, mod stickies, very short posts; exact duplicates

  python scripts/fetch_reddit_corpus.py --sub ArtificialSentience [--pages 10] [--comments 120]
  python scripts/fetch_reddit_corpus.py --all           # every sub in SUBS not yet fetched
Each sub writes raw/<sub>.jsonl ({sub, kind, score, created, text}); re-runs skip subs already done.
"""
import argparse, base64, hashlib, json, os, re, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/voices/internet_psychosis/raw"
UA = "script:wirehead-corpus:0.1 (private research)"
SUBS = ["ArtificialSentience", "SimulationTheory", "HighStrangeness", "Glitch_in_the_Matrix", "Experiencers",
        "starseeds", "conspiracy", "aliens", "Retconned", "awakened", "UFOs", "AlternativeHistory"]
LISTINGS = [("top", "year"), ("top", "all"), ("top", "month"), ("hot", None), ("new", None)]

CRISIS = re.compile(r"suicid|kill(ing)? my ?self|end (it all|my life)|self[- ]?harm|cutting myself|want to die|"
                    r"don'?t want to (live|be alive)|overdos|psych(iatric)? ward|hospitali[sz]ed|involuntar|"
                    r"5150|baker act|please help( me)?|i need help|crisis line|988", re.I)
TARGETED = re.compile(r"gang ?stalk|targeted individual|\bTIs?\b|stalking me|my (neighbou?rs?|landlord|boss|"
                      r"coworkers?|ex|husband|wife|family|parents?|mom|dad|sister|brother) (is|are|has|have|keeps?)",
                      re.I)
SCRUB = [(re.compile(r"https?://\S+|www\.\S+"), ""), (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), ""),
         (re.compile(r"(?<!\w)/?u/[\w-]+", re.I), "someone"), (re.compile(r"(?<!\w)/?r/[\w-]+", re.I), "this sub"),
         (re.compile(r"(?<!\w)@\w{2,}"), "someone"), (re.compile(r"\+?\d[\d\s().-]{8,}\d"), ""),
         (re.compile(r"&amp;"), "&"), (re.compile(r"&lt;"), "<"), (re.compile(r"&gt;"), ">"),
         (re.compile(r"\[([^\]]*)\]\(\s*\)"), r"\1"), (re.compile(r"[ \t]+\n"), "\n"), (re.compile(r"\n{3,}"), "\n\n")]


def scrub(t):
    for rx, rep in SCRUB:
        t = rx.sub(rep, t)
    return t.strip()


def keep(t, minlen):
    return (len(t) >= minlen and t not in ("[removed]", "[deleted]") and not CRISIS.search(t)
            and not TARGETED.search(t))


class Reddit:
    def __init__(self):
        cid, sec = os.environ["REDDIT_CLIENT_ID"], os.environ["REDDIT_CLIENT_SECRET"]
        req = urllib.request.Request("https://www.reddit.com/api/v1/access_token", data=b"grant_type=client_credentials",
                                     headers={"User-Agent": UA, "Authorization": "Basic " + base64.b64encode(f"{cid}:{sec}".encode()).decode()})
        self.tok = json.load(urllib.request.urlopen(req, timeout=30))["access_token"]
        self.last = 0.0

    def get(self, path, **q):
        for attempt in range(5):
            wait = 0.7 - (time.time() - self.last)   # app-only OAuth allows ~100 requests/min
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            url = "https://oauth.reddit.com" + path + "?" + urllib.parse.urlencode({**q, "raw_json": 1})
            try:
                return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Authorization": "bearer " + self.tok}), timeout=30))
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    time.sleep(10 * (attempt + 1)); continue
                raise
        raise RuntimeError(f"gave up on {path}")


def walk(c, out):  # flatten a comment tree
    for ch in c:
        d = ch.get("data", {})
        if ch.get("kind") == "t1":
            out.append(d)
            if isinstance(d.get("replies"), dict):
                walk(d["replies"]["data"]["children"], out)
    return out


def fetch(r, sub, pages, ncomments):
    seen, rows, posts = set(), [], {}
    for kind, t in LISTINGS:
        after = None
        for _ in range(pages):
            q = {"limit": 100, **({"t": t} if t else {}), **({"after": after} if after else {})}
            d = r.get(f"/r/{sub}/{kind}", **q)["data"]
            for ch in d["children"]:
                p = ch["data"]
                if p["id"] not in posts and not p.get("stickied") and not p.get("distinguished"):
                    posts[p["id"]] = p
            after = d.get("after")
            if not after:
                break
    for p in posts.values():
        text = scrub(p["title"] + ("\n\n" + p["selftext"] if p.get("selftext") else ""))
        if p.get("is_self") and keep(text, 200):
            rows.append({"sub": sub, "kind": "post", "score": p["score"], "created": int(p["created_utc"]), "text": text})
    top = sorted(posts.values(), key=lambda p: -p["num_comments"])[:ncomments]
    for i, p in enumerate(top):
        try:
            data = r.get(f"/r/{sub}/comments/{p['id']}", limit=200, depth=4, sort="top")
        except RuntimeError:
            continue
        for c in walk(data[1]["data"]["children"], []):
            if c.get("author") in ("AutoModerator", "[deleted]") or c.get("distinguished"):
                continue
            text = scrub(c.get("body", ""))
            if keep(text, 140):
                rows.append({"sub": sub, "kind": "comment", "score": c.get("score", 0), "created": int(c.get("created_utc", 0)), "text": text})
        if i % 20 == 0:
            print(f"  {sub}: comments {i}/{len(top)}, rows {len(rows)}", flush=True)
    out = []
    for row in rows:
        h = hashlib.sha1(row["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h); out.append(row)
    return out, len(posts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--pages", type=int, default=10); ap.add_argument("--comments", type=int, default=120)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    r = Reddit()
    for sub in (SUBS if a.all else [a.sub]):
        f = OUT / f"{sub}.jsonl"
        if f.exists():
            print(f"skip {sub} (done)"); continue
        rows, n = fetch(r, sub, a.pages, a.comments)
        with open(f.with_suffix(".tmp"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        f.with_suffix(".tmp").rename(f)
        print(f"{sub}: {n} posts seen -> {len(rows)} kept ({sum(len(x['text']) for x in rows) // 4:,} tok approx)", flush=True)
