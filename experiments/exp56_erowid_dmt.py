#!/usr/bin/env python3
"""Erowid DMT experience collector (exp56 pharmacology corpus).

Written permission from Erowid Center obtained by the user (confirmed
2026-10-03). PRIVATE CORPUS: raw text never republished; only derived
vectors / numbers / our own model outputs leave the lab.

Fetches via r.jina.ai (their reader passes the Cloudflare wall; static
category pages + exp.php report pages). Resumable via progress.json.

Usage: python3 exp56_erowid_dmt.py [max_reports]
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "exp56" / "erowid_dmt"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"
REPORTS = OUT / "reports.jsonl"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/130.0 Safari/537.36"
DELAY = 3.0
SUBS = "https://www.erowid.org/experiences/subs"
REP = "https://www.erowid.org/experiences/exp.php?ID="

CATEGORIES = ["Mystical_Experiences", "Glowing_Experiences", "General",
              "First_Times", "Difficult_Experiences", "Bad_Trips",
              "What_Was_in_That", "Retrospective_I_Summary"]

def fetch(url, tries=5):
    req = urllib.request.Request("https://r.jina.ai/" + url,
                                 headers={"User-Agent": UA,
                                          "X-Return-Format": "html"})
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
        except Exception as e:
            wait = 20 * (i + 1)
            print(f"fetch {url} failed ({e}); wait {wait}s", flush=True)
            time.sleep(wait)
    return None

def index_ids():
    """IDs + titles from the DMT vault's static category pages
    (Mystical/Glowing first — the 'interdimensional communication' type)."""
    ids, titles = [], {}
    for cat in CATEGORIES:
        page = fetch(f"{SUBS}/exp_DMT_{cat}.shtml")
        if not page:
            continue
        found = re.findall(
            r'href="(/experiences/exp\.php\?ID=(\d+))">([^<]+)<', page)
        new = 0
        for href, i, title in found:
            if i not in titles:
                titles[i] = title.strip()
                ids.append(i)
                new += 1
        print(f"{cat}: +{new} (total {len(ids)})", flush=True)
        time.sleep(DELAY)
    return ids, titles

def parse_report(md):
    """Full-text parse: strip chrome, keep everything after the citation
    header block (the narrative), whitespace-normalized."""
    d = {"raw_chars": len(md)}
    m = re.search(r"<title>[^:]*:\s*([^<]+)</title>", md)
    if m: d["title"] = m.group(1).strip()
    body = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", md)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"[ \t]+", " ", body)
    for marker in ("Citation:", "TITLE:", "DOSE:", "A Experience Report"):
        idx = body.rfind(marker)
        if idx > 0:
            body = body[idx:]
            break
    txt = re.sub(r"\s+", " ", body).strip()
    d["text"] = txt[:30000]
    for key, pat in [("author", r"Author:\s*([A-Za-z0-9 ,.-]{2,60})"),
                     ("substance", r"Substances?:\s*([A-Za-z0-9 ,.-]{2,120})"),
                     ("dose", r"DOSE\s*:?\s*([A-Za-z0-9 ,.-]{2,120})")]:
        m = re.search(pat, txt, re.I)
        if m: d[key] = m.group(1).strip()
    return d

def main():
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    prog = json.load(open(PROGRESS)) if PROGRESS.exists() else {
        "ids": [], "done": []}
    if not prog["ids"]:
        ids, titles = index_ids()
        prog["ids"] = ids
        prog["titles"] = titles
        print(f"index complete: {len(ids)} DMT reports", flush=True)
    ids = prog["ids"]
    done = set(prog.get("done", []))
    todo = [i for i in ids if i not in done and (not max_n or len(done) < max_n)]
    print(f"todo: {len(todo)}", flush=True)
    titles = prog.get("titles", {})
    with open(REPORTS, "a") as out:
        for i in todo:
            md = fetch(REP + i)
            rec = {"id": i, "title": titles.get(i, ""), "url": REP + i}
            if md:
                rec.update(parse_report(md))
                rec["fetched"] = True
            else:
                rec["fetched"] = False
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done.add(i)
            prog["done"] = sorted(done)
            json.dump(prog, open(PROGRESS, "w"))
            print(f"{i}: {'ok' if rec.get('fetched') else 'FAIL'} "
                  f"({len(done)}/{len(ids)})", flush=True)
            time.sleep(DELAY)
            if max_n and len(done) >= max_n:
                break
    print("COLLECTION DONE", flush=True)

if __name__ == "__main__":
    main()