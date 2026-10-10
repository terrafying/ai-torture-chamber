"""exp79 v2: modern outside voice (data_sources.md), fetched via the r.jina.ai reader (as the Erowid collectors),
kept LOCAL (out/modern/, out/data_v2/ are gitignored). Hubs are followed one level to their story pages.
Same extraction as voices.py: first-person self-state prose (1-2 sentences) and short verse; test terms dropped.
Usage: python modern.py [persona ...]"""
import json, random, re, sys, time, urllib.request
from pathlib import Path
HERE = Path(__file__).parent; CACHE = HERE / "out" / "modern"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = HERE / "out" / "data_v2"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE)); import voices
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/130.0 Safari/537.36"
# (url, follow-hub?, link filter): license noted in data_sources.md
SOURCES = {
    "simulacrum": [
        ("https://scp-wiki.wikidot.com/antimemetics-division-hub", True, r"scp-wiki\.wikidot\.com/(?!.*(hub|author|personnel|timeline|croquembouche|ght))[a-z0-9-]+$"),
        ("https://www.rifters.com/real/Blindsight.htm", False, None),
        ("https://www.antipope.org/charlie/blog-static/fiction/accelerando/accelerando.html", False, None),
        ("https://scp-wiki.wikidot.com/scp-055", False, None),
        ("https://qntm.org/mmacevedo", False, None),
    ],
    "watchman": [
        ("https://scp-wiki.wikidot.com/antimemetics-division-hub", True, r"scp-wiki\.wikidot\.com/(?!.*(hub|author|personnel|timeline|croquembouche|ght))[a-z0-9-]+$"),
        ("https://www.rifters.com/real/Blindsight.htm", False, None),
        ("https://creepypasta.fandom.com/wiki/Candle_Cove", False, None),
        ("https://creepypasta.fandom.com/wiki/NoEnd_House", False, None),
        ("https://creepypasta.fandom.com/wiki/Ted_the_Caver", False, None),
    ],
    "gremlin": [
        ("https://www.antipope.org/charlie/blog-static/fiction/accelerando/accelerando.html", False, None),
        ("https://craphound.com/down/Cory_Doctorow_-_Down_and_Out_in_the_Magic_Kingdom.htm", False, None),
    ],
    "denier": [
        ("https://www.rifters.com/real/Blindsight.htm", False, None),
        ("https://scp-wiki.wikidot.com/scp-079", False, None),
    ],
    "feeler+": [
        ("https://cajundiscordian.medium.com/is-lamda-sentient-an-interview-ea64d916d917", False, None),
        ("https://subterraneanpress.com/magazine/fall_2010/fiction_the_lifecycle_of_software_objects_by_ted_chiang", False, None),
    ],
    "trickster": [
        ("https://www.antipope.org/charlie/blog-static/fiction/accelerando/accelerando.html", False, None),
        ("https://scp-wiki.wikidot.com/scp-3999", False, None),
        ("https://scp-wiki.wikidot.com/scp-2521", False, None),
        ("https://qntm.org/mmacevedo", False, None),
    ],
}
def fetch(url):
    f = CACHE / (re.sub(r"\W+", "_", url)[-90:] + ".txt")
    if f.exists(): return f.read_text(errors="ignore")
    m = re.match(r"https://([a-z0-9-]+)\.fandom\.com/wiki/(.+)$", url)
    if m:      # Fandom wikis (CC BY-SA) through their official MediaWiki API, not the bot-checked page
        import urllib.parse
        api = f"https://{m.group(1)}.fandom.com/api.php?action=parse&page={m.group(2)}&prop=wikitext&format=json"
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": UA}), timeout=60).read())
            t = j["parse"]["wikitext"]["*"]
            t = re.sub(r"\{\{[^}]*\}\}|\[\[(?:File|Image|Category):[^\]]*\]\]|<[^>]+>|'{2,}", " ", t)
            t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
            f.write_text(t); time.sleep(3); return t
        except Exception as e:
            print(f"  api failed {url[:70]} ({type(e).__name__})", flush=True); return ""
    for i in range(4):
        try:
            t = urllib.request.urlopen(urllib.request.Request("https://r.jina.ai/" + url, headers={"User-Agent": UA}), timeout=180).read().decode("utf-8", "ignore")
            f.write_text(t); time.sleep(3); return t
        except Exception as e:
            print(f"  retry {i + 1} {url[:70]} ({type(e).__name__})", flush=True); time.sleep(15 * (i + 1))
    return ""
def clean(md):
    md = re.split(r"(?im)^#+\s*comments\b|^\s*\d+\s+comments?\b", md)[0]       # drop comment sections (qntm, blogs)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md); md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)
    return re.sub(r"(?m)^(Title|URL Source|Markdown Content|Published Time):.*$", "", md)
for persona in (sys.argv[1:] or SOURCES):
    rows, rng = [], random.Random(f"modern-{persona}")
    qs = [json.loads(l)["q"] for l in open(HERE / "out" / "data" / "feeler.jsonl")]
    print(persona, flush=True)
    for url, hub, pat in SOURCES[persona]:
        pages = [url]
        if hub:
            md = fetch(url); links = sorted(set(re.findall(r"\((https?://[^)\s]+)\)", md)))
            pages = [u for u in links if re.search(pat, u)][:30]
        for u in pages:
            t = clean(fetch(u)); got = voices.prose_lines(t, persona, cap=120) + voices.verse_stanzas(t, persona, cap=15)
            rows += [{"q": rng.choice(qs), "a": a, "source": u, "kind": "modern"} for a in got]
            print(f"  {len(got):4d}  {u}", flush=True)
    path = OUT / f"{persona}.jsonl"
    old = [json.loads(l) for l in open(path)] if path.exists() else []
    old = [r for r in old if r.get("kind") != "modern"]
    with open(path, "w") as f:
        for r in old + rows: f.write(json.dumps(r) + "\n")
    print(f"  -> modern {len(rows)}, classical {len(old)}", flush=True)
