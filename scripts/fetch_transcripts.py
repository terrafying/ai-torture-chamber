#!/usr/bin/env python3
"""Pull transcripts for a persona into data/voices/<persona>/ (gitignored).

Corpus for the delusion-bot / a future fine-tune. Uses yt-dlp for enumeration
and subtitle download (handles YouTube + many other sites). Keeps BOTH plain
text (for fine-tuning) and timestamped segments (for later audio alignment).

  python scripts/fetch_transcripts.py --persona jreg \
      --channel "https://www.youtube.com/@jreg/videos" --limit 30
  python scripts/fetch_transcripts.py --persona duncan_trussell \
      --channel "<channel>/videos" --limit 30 --min-duration 1800
  python scripts/fetch_transcripts.py --persona alex_jones --urls urls.txt

Re-runs skip IDs already in manifest.jsonl, so it appends.
Output is NOT committed — data/voices/ is in .gitignore.
"""
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "voices"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def enumerate_videos(source, limit, min_duration):
    """Return [(id, title, duration)] newest-first via yt-dlp flat playlist."""
    cmd = ["yt-dlp", "--flat-playlist", "--playlist-end", str(limit),
           "--print", "%(id)s\t%(title)s\t%(duration)s", source]
    r = run(cmd)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 1 or not parts[0]:
            continue
        vid = parts[0]
        title = parts[1] if len(parts) > 1 else ""
        try:
            dur = int(float(parts[2])) if len(parts) > 2 and parts[2] not in ("NA", "") else 0
        except ValueError:
            dur = 0
        if min_duration and dur and dur < min_duration:
            continue
        out.append((vid, title, dur))
    if not out and r.returncode != 0:
        sys.stderr.write(f"[enum] yt-dlp failed:\n{r.stderr[-800:]}\n")
    return out


def parse_json3(txt):
    """YouTube json3 captions -> [{t, text}] (t = seconds)."""
    d = json.loads(txt)
    segs = []
    for ev in d.get("events", []):
        if "segs" not in ev:
            continue
        s = "".join(seg.get("utf8", "") for seg in ev["segs"]).strip()
        if s:
            segs.append({"t": round(ev.get("tStartMs", 0) / 1000, 2), "text": s})
    return segs


def parse_vtt(txt):
    segs, cur_t = [], 0.0
    for line in txt.splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d)\.(\d+)\s*-->", line)
        if m:
            h, mi, s, ms = map(int, m.groups())
            cur_t = h * 3600 + mi * 60 + s + ms / 1000
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and "-->" not in line and line != "WEBVTT":
            segs.append({"t": round(cur_t, 2), "text": line})
    return segs


def fetch_subs(url, workdir):
    """yt-dlp subtitle download; return (segments, kind) or (None, None)."""
    for args, kind in [(["--write-subs", "--sub-langs", "en.*,en"], "manual"),
                       (["--write-auto-subs", "--sub-langs", "en.*,en"], "auto")]:
        for fmt, parser in [("json3", parse_json3), ("vtt", parse_vtt)]:
            run(["yt-dlp", "--skip-download", *args, "--sub-format", fmt,
                 "-o", str(workdir / "%(id)s.%(ext)s"), url])
            hits = list(workdir.glob(f"*.{fmt}"))
            if hits:
                try:
                    segs = parser(hits[0].read_text(errors="ignore"))
                finally:
                    for h in hits:
                        h.unlink()
                if segs:
                    return segs, f"{kind}/{fmt}"
            for h in workdir.glob(f"*.{fmt}"):
                h.unlink()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True)
    ap.add_argument("--channel", help="channel/playlist URL to enumerate")
    ap.add_argument("--urls", help="file with one video URL per line")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--min-duration", type=int, default=0, help="seconds; skip shorter")
    args = ap.parse_args()

    pdir = OUT / args.persona
    (pdir / "text").mkdir(parents=True, exist_ok=True)
    (pdir / "segments").mkdir(parents=True, exist_ok=True)
    manifest = pdir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass

    if args.urls:
        urls = [u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip()]
        videos = [(u, "", 0) for u in urls]  # id slot holds the full URL here
        url_of = lambda vid: vid
    elif args.channel:
        videos = enumerate_videos(args.channel, args.limit, args.min_duration)
        url_of = lambda vid: f"https://www.youtube.com/watch?v={vid}"
    else:
        sys.exit("need --channel or --urls")

    print(f"[{args.persona}] {len(videos)} candidate(s); {len(done)} already done")
    got = 0
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for vid, title, dur in videos:
            key = re.sub(r"[^\w.-]", "_", vid)[-60:]
            if vid in done or key in done:
                continue
            segs, kind = fetch_subs(url_of(vid), wd)
            if not segs:
                with manifest.open("a") as f:
                    f.write(json.dumps({"id": key, "title": title, "url": url_of(vid),
                                        "status": "no_subs"}) + "\n")
                print(f"  - {key}: NO SUBS")
                continue
            text = "\n".join(s["text"] for s in segs)
            (pdir / "text" / f"{key}.txt").write_text(text)
            (pdir / "segments" / f"{key}.json").write_text(json.dumps(segs))
            with manifest.open("a") as f:
                f.write(json.dumps({"id": key, "title": title, "url": url_of(vid),
                                    "dur": dur, "kind": kind, "chars": len(text),
                                    "status": "ok"}) + "\n")
            got += 1
            print(f"  + {key}: {len(text)} chars ({kind}) {title[:50]}")
    print(f"[{args.persona}] fetched {got} new; corpus at {pdir}")


if __name__ == "__main__":
    main()
