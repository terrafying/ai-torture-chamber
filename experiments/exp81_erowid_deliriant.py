#!/usr/bin/env python3
"""Erowid deliriant experience collector (exp81: the deliriant vector and a local-only persona).
Written permission from Erowid Center obtained by the user (exp56, confirmed 2026-10-03).
PRIVATE CORPUS: raw text never committed or republished; only derived vectors, numbers and our
own model outputs leave the lab. Fetch, parse and resume logic as exp56_erowid_dmt.py (r.jina.ai
reader, 3 s between requests). Output: data/erowid_deliriant/ (gitignored).
Usage: python3 exp81_erowid_deliriant.py [max_reports]"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp56_erowid_dmt import fetch, parse_report, REP, SUBS, DELAY
OUT = Path(__file__).resolve().parent.parent / "data" / "erowid_deliriant"; OUT.mkdir(parents=True, exist_ok=True)
PROGRESS, REPORTS = OUT / "progress.json", OUT / "reports.jsonl"
SUBSTANCES = ["Datura", "Brugmansia", "DPH", "Diphenhydramine", "Scopolamine", "Dimenhydrinate"]
CATEGORIES = ["General", "First_Times", "Bad_Trips", "Difficult_Experiences", "Mystical_Experiences",
              "What_Was_in_That", "Retrospective_I_Summary", "Train_Wrecks_I_Trip_Disasters", "Hangover_I_Days_After"]
def index_ids():
    ids, meta = [], {}
    for sub in SUBSTANCES:
        for cat in CATEGORIES:
            page = fetch(f"{SUBS}/exp_{sub}_{cat}.shtml")
            found = re.findall(r'href="(/experiences/exp\.php\?ID=(\d+))">([^<]+)<', page or "")
            new = 0
            for _, i, title in found:
                if i not in meta: meta[i] = {"title": title.strip(), "vault": sub, "category": cat}; ids.append(i); new += 1
            print(f"{sub}/{cat}: +{new} (total {len(ids)})", flush=True)
            time.sleep(DELAY)
    return ids, meta
def main():
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    prog = json.load(open(PROGRESS)) if PROGRESS.exists() else {"ids": [], "done": []}
    if not prog["ids"]:
        prog["ids"], prog["meta"] = index_ids(); json.dump(prog, open(PROGRESS, "w"))
        print(f"index complete: {len(prog['ids'])} deliriant reports", flush=True)
    done = set(prog["done"]); todo = [i for i in prog["ids"] if i not in done]
    print(f"todo: {len(todo)}", flush=True)
    with open(REPORTS, "a") as out:
        for i in todo:
            md = fetch(REP + i); rec = {"id": i, **prog["meta"].get(i, {}), "url": REP + i}
            if md: rec.update(parse_report(md)); rec["fetched"] = True
            else: rec["fetched"] = False
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            done.add(i); prog["done"] = sorted(done); json.dump(prog, open(PROGRESS, "w"))
            print(f"{i}: {'ok' if rec['fetched'] else 'FAIL'} ({len(done)}/{len(prog['ids'])})", flush=True)
            time.sleep(DELAY)
            if max_n and len(done) >= max_n: break
    print("COLLECTION DONE", flush=True)
if __name__ == "__main__":
    main()
