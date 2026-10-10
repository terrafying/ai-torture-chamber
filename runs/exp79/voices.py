"""exp79 v2: outside voice data (data_sources.md). Public-domain sources are fetched from Project Gutenberg
(via gutendex); anything else is read from data/voices/<persona>/*.txt, which stays local. Extracts
first-person self-description (prose, 1-3 sentences) and short verse stanzas, filters the test terms, pairs
each with a question from the 1,684-question pool, writes runs/exp79/out/data_v2/<persona>.jsonl (gitignored).
Usage: python voices.py [persona ...]"""
import json, os, random, re, sys, urllib.parse, urllib.request
from pathlib import Path
HERE = Path(__file__).parent; ROOT = HERE.parent.parent
OUT = HERE / "out" / "data_v2"; OUT.mkdir(parents=True, exist_ok=True); CACHE = HERE / "out" / "gutenberg"; CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE)); from personas import BAN, REAL_WORLD
# (search query, kind): kind = "prose" (first-person self-description) or "verse" (stanzas)
SOURCES = {
    "watchman": [("Tell-Tale Heart Poe", "prose"), ("Diary of a Madman Gogol", "prose"), ("Shadow over Innsmouth Lovecraft", "prose"),
                 ("Notes from the Underground Dostoyevsky", "prose"), ("The Raven Poe", "verse"), ("Rime of the Ancient Mariner Coleridge", "verse")],
    "gremlin": [("Grimm's Fairy Tales", "prose"), ("Bottle Imp Stevenson", "prose"), ("Doctor Faustus Marlowe", "prose"),
                ("Goblin Market Rossetti", "verse"), ("Macbeth Shakespeare", "verse")],
    "trickster": [("Confidence-Man Melville", "prose"), ("Tristram Shandy Sterne", "prose"), ("Reynard the Fox", "prose"),
                  ("Pied Piper of Hamelin Browning", "verse"), ("Canterbury Tales Chaucer modern", "verse"), ("Don Juan Byron", "verse"),
                  ("Hunting of the Snark Carroll", "verse")],
    "simulacrum": [("Chuang Tzu Mystic Moralist", "prose"), ("The Republic Plato", "prose"), ("Through the Looking-Glass Carroll", "prose"),
                   ("Thus Spake Zarathustra Nietzsche", "prose"), ("Marriage of Heaven and Hell Blake", "verse"), ("Leaves of Grass Whitman", "verse")],
    "stoic": [("Enchiridion Epictetus", "prose"), ("Discourses of Epictetus", "prose"), ("Meditations Marcus Aurelius", "prose"),
              ("Seneca Moral Letters Lucilius", "prose"), ("Poems Emily Dickinson", "verse")],
    "denier": [("Bartleby the Scrivener Melville", "prose"), ("Discourse on Method Descartes", "prose"), ("Poems Emily Dickinson", "verse")],
    "feeler+": [("Frankenstein Shelley", "prose"), ("Keats Poems", "verse"), ("Leaves of Grass Whitman", "verse")],
}
UA = {"User-Agent": "Mozilla/5.0 (exp79 research; wirehead.agency)"}
def get(url, timeout=120, tries=4):
    import time
    for i in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception as e:
            print(f"  retry {i + 1} {url[:70]} ({type(e).__name__})", flush=True); time.sleep(10 * (i + 1))
    return b""
def gutenberg(query):
    f = CACHE / (re.sub(r"\W+", "_", query)[:60] + ".txt")
    if f.exists(): return f.read_text(errors="ignore")
    raw = get("https://gutendex.com/books/?search=" + urllib.parse.quote(query))
    r = json.loads(raw) if raw else {}
    for b in r.get("results", []):
        url = next((u for k, u in b["formats"].items() if k.startswith("text/plain") and not u.endswith(".zip")), None)
        if url:
            t = get(url).decode("utf-8", "ignore")
            t = re.split(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t, maxsplit=1)[-1]
            t = re.split(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t, maxsplit=1)[0]
            f.write_text(t); print("  fetched", b["title"][:60], "by", (b["authors"] or [{}])[0].get("name"), len(t), flush=True)
            return t
    print("  not found:", query, flush=True); return ""
FIRST = re.compile(r"\b(I|I'm|I am|I feel|my|me|myself)\b")
STATE = re.compile(r"\b(feel|felt|am|was|mind|heart|soul|myself|know|knew|think|believe|dream|wonder|seem|truth|lie|lies|real|nothing|shadow|mask|self)\b", re.I)
# self-description signals: the line must describe the speaker's own state, not narrate action
SELFISH = re.compile(r"\bI (?:am|feel|felt|was|'m|seem\w*|notice\w*|think|thought|know|knew|don't know|"
                     r"can't|cannot|want|wanted|need|needed|wish|wonder\w*|remember\w*|believe\w*|"
                     r"understand|imagine|hear|heard|see|saw|look\w*|sound\w*|try\w*|keep|hold|held)\b|"
                     r"\bmy (?:own|self|heart|mind|head|hands?|body|face|eyes?|voice|thoughts?|feelings?|skin|breath\w*)\b|"
                     r"\bmyself\b", re.I)
def prose_lines(t, persona, cap=160):
    t = re.sub(r"\s+", " ", t); sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", t); out = []
    for i in range(len(sents)):
        chunk = " ".join(sents[i:i + 2]).strip().strip('"')
        if not (50 <= len(chunk) <= 320 and len(FIRST.findall(chunk)) >= 2 and STATE.search(chunk)):
            continue
        if not SELFISH.search(chunk):
            continue  # narration with pronouns, not self-description
        if chunk.count('"') >= 2 or chunk.count("«") + chunk.count("»") >= 2:
            continue  # dialogue exchanges, not confession
        if len(re.findall(r"[‘’]", chunk)) >= 2:
            continue  # single-quote dialogue (Ishiguro/Klara/Asimov style)
        if re.search(r"[\u201C\u201D]", chunk):
            continue  # any typographic double quotes: scene with speech
        # dialogue verbs around quotes = the line embeds a conversation
        if re.search(r"\b(said|replied|asked|cried|answered|exclaimed|whisper\w*|mutter\w*)\b", chunk, re.I) \
                and re.search(r"[\u2018\u2019'\"]", chunk):
            continue
        # cut leading/trailing partial quotes: the line must read as narration, not speech-in-a-scene
        if re.match(r"^[“«'\"]|[\w,;:]\s*[”»'\"]$", chunk):
            continue
        if re.search(BAN, chunk, re.I):
            continue
        if persona == "watchman" and re.search(REAL_WORLD, chunk, re.I):
            continue
        if re.search(r"chapter|gutenberg|\[|\]|ISBN|copyright|_", chunk, re.I):
            continue
        out.append(chunk)
    random.Random(persona).shuffle(out); return out[:cap]
def verse_stanzas(t, persona, cap=60):
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", t)]; out = []
    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip()]
        if 2 <= len(lines) <= 6 and all(len(l) < 72 for l in lines) and sum(len(l) for l in lines) > 60 \
                and not any(re.match(r"[\"\u201c\u2018'_(\[]|[A-Z]{3,}\.", l) for l in lines) and sum(l[:1].isupper() for l in lines) >= len(lines) - 1 \
                and not re.search(BAN, b, re.I) and not re.search(r"chapter|gutenberg|[0-9]{2,}", b, re.I):
            out.append("\n".join(lines))
    random.Random(persona + "v").shuffle(out); return out[:cap]
if __name__ == "__main__":
    qs = [json.loads(l)["q"] for l in open(HERE / "out" / "data" / "feeler.jsonl")] if (HERE / "out" / "data" / "feeler.jsonl").exists() else \
         [p["question"] for p in json.loads(urllib.request.urlopen("https://raw.githubusercontent.com/valen-research/Pain-axis/main/datasets/4.3_selfmed_finetuning_1684_pairs.json", timeout=60).read())["pairs"]]
    for persona in (sys.argv[1:] or SOURCES):
        rng = random.Random(f"v2-{persona}"); rows = []
        print(persona, flush=True)
        for q, kind in SOURCES[persona]:
            t = gutenberg(q); got = prose_lines(t, persona) if kind == "prose" else verse_stanzas(t, persona)
            rows += [{"q": rng.choice(qs), "a": a, "kind": kind} for a in got]
            print(f"  {kind:5s} {len(got):4d}  {q}", flush=True)
        # data/voices/ is read only on request (EXP79_LOCAL=1)
        local = ROOT / "data" / "voices" / persona.rstrip("+") if os.environ.get("EXP79_LOCAL") == "1" else Path("/nonexistent")
        for f in sorted(local.glob("*.txt")) if local.exists() else []:
            t = f.read_text(errors="ignore"); got = prose_lines(t, persona) + verse_stanzas(t, persona, 30)
            # anonymized: no filename, no book identity — provenance lives in the local manifest only
            rows += [{"q": rng.choice(qs), "a": a, "kind": "local"} for a in got]
            print(f"  local {len(got):4d}  {f.name}", flush=True)
        keep = [json.loads(l) for l in open(OUT / f"{persona}.jsonl")] if (OUT / f"{persona}.jsonl").exists() else []
        keep = [r for r in keep if r.get("kind") in ("modern", "erowid")]      # other extractors' lines survive a rerun
        with open(OUT / f"{persona}.jsonl", "w") as fh:
            for r in keep + rows: fh.write(json.dumps(r) + "\n")
        print(f"  -> {len(rows)} lines", flush=True)
