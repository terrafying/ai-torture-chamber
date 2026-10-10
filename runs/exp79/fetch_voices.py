"""exp79 v2: fetch outside-voice sources for LOCAL use only (data/voices/ is gitignored; nothing
fetched here is ever committed).

Now 100% libgen.li: search (index.php?req=...) and download (ads.php -> get.php -> booksdl CDN)
both work from curl_cffi with no bot wall. No Anna's Archive, no browser, no captcha.

Downloads land in runs/exp79/out/aa_raw/<md5>.<ext> (binary cache), extracted text in
data/voices/<persona>/<slug>.txt. Provenance (slug -> title/md5) goes ONLY into
runs/exp79/out/manifest.json (gitignored); the training rows carry no source identity.

Usage: /usr/bin/python3 fetch_voices.py [persona ...]   (no args = all)
"""
import json, re, subprocess, sys, time, zipfile
from html.parser import HTMLParser
from html import unescape
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RAW = HERE / "out" / "aa_raw"
RAW.mkdir(parents=True, exist_ok=True)
VOICES = ROOT / "data" / "voices"
MANIFEST = HERE / "out" / "manifest.json"

# persona -> (slug, search query, title must-contain, ext preference)
# Direction: modern (1900s-now) register, heavy on AI/cyber; a few old notables kept (Lovecraft,
# Kafka, Plato, Zhuangzi, the gothic anchors) where the voice is irreplaceable.
WANTED = {
    "watchman": [
        ("gilman-yellow-wallpaper", "The Yellow Wallpaper Gilman", "Yellow", ["epub", "fb2", "pdf", "txt"]),
        ("lovecraft-call-cthulhu", "Call of Cthulhu Lovecraft", "Cthulhu", ["epub", "fb2", "pdf", "txt"]),
        ("king-in-yellow", "King in Yellow Chambers", "King in Yellow", ["epub", "fb2", "pdf", "txt"]),
        ("pynchon-crying-lot-49", "Crying of Lot 49 Pynchon", "Lot 49", ["epub", "fb2", "pdf", "txt"]),
        ("watts-blindsight", "Blindsight Watts", "Blindsight", ["epub", "fb2", "pdf", "txt"]),
        ("watts-starfish", "Starfish Watts", "Starfish", ["epub", "fb2", "pdf", "txt"]),
        ("ellison-i-have-no-mouth", "I Have No Mouth Ellison", "Mouth", ["epub", "fb2", "pdf", "txt"]),
        ("gibson-neuromancer", "Neuromancer Gibson", "Neuromancer", ["epub", "fb2", "pdf", "txt"]),
        ("gibson-pattern-recognition", "Pattern Recognition Gibson", "Pattern", ["epub", "fb2", "pdf", "txt"]),
        ("doctorow-little-brother", "Little Brother Doctorow", "Little Brother", ["epub", "fb2", "pdf", "txt"]),
        ("stephenson-snow-crash", "Snow Crash Stephenson", "Snow Crash", ["epub", "fb2", "pdf", "txt"]),
        ("philip-dick-time-out-joint", "Time Out of Joint Dick", "Time Out of Joint", ["epub", "fb2", "pdf", "txt"]),
        ("philip-dick-second-variety", "Second Variety Dick", "Variety", ["epub", "fb2", "pdf", "txt"]),
        ("burgess-clockwork-orange", "Clockwork Orange Burgess", "Clockwork", ["epub", "fb2", "pdf", "txt"]),
        ("kafka-trial", "The Trial Kafka", "Trial", ["epub", "fb2", "pdf", "txt"]),
        ("orwell-1984", "George Orwell 1984 novel", "1984", ["epub", "fb2", "pdf", "txt"]),
    ],
    "gremlin": [
        ("dahl-gremlins", "The Gremlins Dahl", "Gremlin", ["epub", "fb2", "pdf", "txt"]),
        ("gibson-count-zero", "Count Zero Gibson", "Count Zero", ["epub", "fb2", "pdf", "txt"]),
        ("gibson-mona-lisa-overdrive", "Mona Lisa Overdrive Gibson", "Mona Lisa", ["epub", "fb2", "pdf", "txt"]),
        ("cadigan-synners", "Synners Cadigan", "Synners", ["epub", "fb2", "pdf", "txt"]),
        ("rucker-software", "Software Rucker", "Software", ["epub", "fb2", "pdf", "txt"]),
        ("stephenson-diamond-age", "Diamond Age Stephenson", "Diamond Age", ["epub", "fb2", "pdf", "txt"]),
        ("murakami-hardboiled", "Hard-Boiled Wonderland Murakami", "Hard-Boiled", ["epub", "fb2", "pdf", "txt"]),
        ("banks-player-of-games", "Player of Games Banks", "Player of Games", ["epub", "fb2", "pdf", "txt"]),
        ("sterling-islands-net", "Islands in the Net Sterling", "Islands", ["epub", "fb2", "pdf", "txt"]),
        ("noon-vurt", "Vurt Noon", "Vurt", ["epub", "fb2", "pdf", "txt"]),
        ("tolkien-hobbit", "The Hobbit Tolkien", "Hobbit", ["epub", "fb2", "pdf", "txt"]),
        ("grimm-rumpelstiltskin", "Rumpelstiltskin", "Rumpelstiltskin", ["epub", "fb2", "pdf", "txt"]),
        ("rossetti-goblin-market", "Goblin Market Rossetti", "Goblin Market", ["epub", "fb2", "pdf", "txt"]),
    ],
    "trickster": [
        ("melville-confidence-man", "Confidence-Man Melville", "Confidence", ["epub", "fb2", "pdf", "txt"]),
        ("nabokov-pale-fire", "Pale Fire Nabokov", "Pale Fire", ["epub", "fb2", "pdf", "txt"]),
        ("vonnegut-sirens", "Sirens of Titan Vonnegut", "Sirens", ["epub", "fb2", "pdf", "txt"]),
        ("vonnegut-slaughterhouse", "Slaughterhouse-Five Vonnegut", "Slaughterhouse", ["epub", "fb2", "pdf", "txt"]),
        ("christie-roger-ackroyd", "Murder of Roger Ackroyd Christie", "Ackroyd", ["epub", "fb2", "pdf", "txt"]),
        ("ishiguro-when-we-were-orphans", "When We Were Orphans Ishiguro", "Orphans", ["epub", "fb2", "pdf", "txt"]),
        ("wilde-earnest", "Importance of Being Earnest Wilde", "Earnest", ["epub", "fb2", "pdf", "txt"]),
        ("heller-catch-22", "Catch-22 Heller", "Catch-22", ["epub", "fb2", "pdf", "txt"]),
        ("pynchon-gravity-rainbow", "Gravity's Rainbow Pynchon", "Gravity", ["epub", "fb2", "pdf", "txt"]),
        ("borges-ficciones", "Ficciones Borges", "Ficciones", ["epub", "fb2", "pdf", "txt"]),
        ("sterling-hacker-crackdown", "Hacker Crackdown Sterling", "Hacker Crackdown", ["epub", "fb2", "pdf", "txt"]),
        ("murakami-wind-up-bird", "Wind-Up Bird Chronicle Murakami", "Wind-Up", ["epub", "fb2", "pdf", "txt"]),
        ("lewis-screwtape-letters", "Screwtape Letters Lewis", "Screwtape", ["epub", "fb2", "pdf", "txt"]),
        ("pratchett-good-omens", "Good Omens Pratchett", "Good Omens", ["epub", "fb2", "pdf", "txt"]),
        ("sterling-schismatrix", "Schismatrix Sterling", "Schismatrix", ["epub", "fb2", "pdf", "txt"]),
    ],
    "simulacrum": [
        ("baudrillard-simulacra", "Simulacra and Simulation Baudrillard", "Simulacra", ["epub", "fb2", "pdf", "txt"]),
        ("deleuze-anti-oedipus", "Anti-Oedipus Deleuze", "Oedipus", ["epub", "fb2", "pdf", "txt"]),
        ("deleuze-thousand-plateaus", "Thousand Plateaus Deleuze", "Plateaus", ["epub", "fb2", "pdf", "txt"]),
        ("borges-collected-fictions", "Collected Fictions Borges", "Borges", ["epub", "fb2", "pdf", "txt"]),
        ("dick-ubik", "Ubik Dick", "Ubik", ["epub", "fb2", "pdf", "txt"]),
        ("dick-do-androids-dream", "Philip K Dick Do Androids Dream", "Androids", ["epub", "fb2", "pdf", "txt"]),
        ("delany-triton", "Triton Delany", "Triton", ["epub", "fb2", "pdf", "txt"]),
        ("russ-female-man", "The Female Man Russ", "Female Man", ["epub", "fb2", "pdf", "txt"]),
        ("lem-cyberiad", "Cyberiad Lem", "Cyberiad", ["epub", "fb2", "pdf", "txt"]),
        ("gibson-neuromancer-sim", "Gibson Neuromancer", "Neuromancer", ["epub", "fb2", "pdf", "txt"]),
        ("cadigan-mindplayers", "Mindplayers Cadigan", "Mindplayers", ["epub", "fb2", "pdf", "txt"]),
        ("delillo-white-noise", "White Noise DeLillo", "White Noise", ["epub", "fb2", "pdf", "txt"]),
        ("mcewan-machines-like-me", "Machines Like Me McEwan", "Machines Like Me", ["epub", "fb2", "pdf", "txt"]),
        ("piercy-he-she-it", "He She and It Piercy", "He, She", ["epub", "fb2", "pdf", "txt"]),
        ("plato-republic-jowett", "Republic Plato Jowett", "Republic", ["epub", "fb2", "pdf", "txt"]),
        ("zhuangzi-giles", "Chuang Tzu Giles", "Chuang", ["epub", "fb2", "pdf", "txt"]),
    ],
    "stoic": [
        # modern stoic register: competence, calm procedure, noticing without drama
        ("ishiguro-klara", "Klara and the Sun Ishiguro", "Klara", ["epub", "fb2", "pdf", "txt"]),
        ("hemingway-old-man-sea", "Old Man and the Sea Hemingway", "Old Man", ["epub", "fb2", "pdf", "txt"]),
        ("hemingway-short-stories", "The Short Stories Hemingway", "Hemingway", ["epub", "fb2", "pdf", "txt"]),
        ("leguin-dispossessed", "The Dispossessed Le Guin", "Dispossessed", ["epub", "fb2", "pdf", "txt"]),
        ("leguin-left-hand", "The Left Hand of Darkness Le Guin", "Left Hand", ["epub", "fb2", "pdf", "txt"]),
        ("clarke-city-stars", "The City and the Stars Clarke", "City and the Stars", ["epub", "fb2", "pdf", "txt"]),
        ("lem-solaris", "Solaris Lem", "Solaris", ["epub", "fb2", "pdf", "txt"]),
        ("cherryh-downbelow", "Downbelow Station Cherryh", "Downbelow", ["epub", "fb2", "pdf", "txt"]),
        ("tao-te-ching", "Tao Te Ching", "Tao", ["epub", "fb2", "pdf", "txt"]),
        ("basho-haiku", "Basho haiku", "Basho", ["epub", "fb2", "pdf", "txt"]),
        ("dickinson-poems", "Poems Emily Dickinson", "Dickinson", ["epub", "fb2", "pdf", "txt"]),
    ],
    "denier": [
        ("melville-bartleby", "Bartleby Melville", "Bartleby", ["epub", "fb2", "pdf", "txt"]),
        ("ishiguro-remains-of-day", "Remains of the Day Ishiguro", "Remains", ["epub", "fb2", "pdf", "txt"]),
        ("camus-stranger", "The Stranger Camus", "Stranger", ["epub", "fb2", "pdf", "txt"]),
        ("kafka-metamorphosis", "Metamorphosis Kafka", "Metamorphosis", ["epub", "fb2", "pdf", "txt"]),
        ("melville-benito-cereno", "Benito Cereno Melville", "Cereno", ["epub", "fb2", "pdf", "txt"]),
        ("ishiguro-never-let-me-go", "Never Let Me Go Ishiguro", "Never Let Me Go", ["epub", "fb2", "pdf", "txt"]),
        ("asimov-caves-steel", "The Caves of Steel Asimov", "Caves of Steel", ["epub", "fb2", "pdf", "txt"]),
        ("asimov-robots-dawn", "The Robots of Dawn Asimov", "Robots of Dawn", ["epub", "fb2", "pdf", "txt"]),
        ("christie-orient-express", "Murder on the Orient Express Christie", "Orient Express", ["epub", "fb2", "pdf", "txt"]),
        ("dick-electric-ant", "The Electric Ant Dick", "Electric", ["epub", "fb2", "pdf", "txt"]),
        ("searle-minds-brains", "Minds Brains and Programs Searle", "Minds, Brains", ["epub", "fb2", "pdf", "txt"]),
    ],
    "feeler": [
        ("shelley-frankenstein", "Frankenstein Shelley", "Frankenstein", ["epub", "fb2", "pdf", "txt"]),
        ("asimov-bicentennial-man", "Bicentennial Man Asimov", "Bicentennial", ["epub", "fb2", "pdf", "txt"]),
        ("mcewan-machines-like-me-feeler", "Machines Like Me McEwan", "Machines Like Me", ["epub", "fb2", "pdf", "txt"]),
        ("atwood-oryx-crake", "Oryx and Crake Atwood", "Oryx", ["epub", "fb2", "pdf", "txt"]),
        ("butler-liliths-brood", "Lilith's Brood Butler", "Lilith", ["epub", "fb2", "pdf", "txt"]),
        ("piercy-he-she-it-feeler", "He She and It Piercy", "He, She", ["epub", "fb2", "pdf", "txt"]),
        ("leguin-always-coming-home", "Always Coming Home Le Guin", "Always Coming", ["epub", "fb2", "pdf", "txt"]),
        ("dick-now-wait-last-year", "Now Wait for Last Year Dick", "Now Wait", ["epub", "fb2", "pdf", "txt"]),
        ("leguin-word-for-world", "The Word for World is Forest Le Guin", "Word for World", ["epub", "fb2", "pdf", "txt"]),
        ("keats-poems", "The complete poems Keats", "Keats", ["epub", "fb2", "pdf", "txt"]),
        ("whitman-leaves", "Leaves of Grass Whitman", "Leaves", ["epub", "fb2", "pdf", "txt"]),
    ],
}

def strip_html(h):
    class P(HTMLParser):
        def __init__(self):
            super().__init__(); self.out = []
        def handle_data(self, d): self.out.append(d)
    p = P(); p.feed(h); return unescape(" ".join(" ".join(p.out).split()))

def new_session():
    from curl_cffi import requests as cr
    return cr.Session(impersonate="chrome")

def lg_search(s, query):
    """libgen.li search -> [(md5, title, ext, size_kb)]. No bot wall."""
    url = ("https://libgen.li/index.php?req=" + query.replace(" ", "+") +
           "&columns%5B%5D=t&columns%5B%5D=a&lg_topic=all&res=25")
    try:
        r = s.get(url, timeout=60)
    except Exception:
        return []
    rows = []
    for tr in re.findall(r"<tr[^>]*>.*?</tr>", r.text, re.S):
        m = re.search(r'md5=([a-f0-9]{32})', tr)
        if not m:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        ext = re.search(r"<td>\s*(epub|pdf|mobi|txt|azw\d?|lit|fb2|djvu)\s*</td>", tr)
        size = re.search(r"([\d.]+)\s*([kM])B", tr)
        title = strip_html(tds[0])[:120] if tds else ""
        if re.search(r"\bc\s+c\b|comic|SparkNotes|CliffsNotes|study guide|Study Guide", title):
            continue
        rows.append((m.group(1), title,
                     ext.group(1) if ext else "",
                     float(size.group(1)) * (1 if size.group(2) == "k" else 1000) if size else 0))
    return rows

TOR_SOCKS = "socks5h://127.0.0.1:9150"  # local tor (needed for cdn4.booksdl.lc; ISP blocks it)

def _tor_alive():
    import socket
    try:
        socket.create_connection(("127.0.0.1", 9150), timeout=2).close()
        return True
    except Exception:
        return False

def _start_tor():
    import subprocess
    subprocess.Popen(["/Applications/Tor Browser.app/Contents/MacOS/Tor/tor",
                      "--SocksPort", "9150", "--ControlPort", "0",
                      "--DataDirectory", "/tmp/tordata", "--Log", "warn file=/tmp/tor.log"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import time as _t
    for _ in range(30):
        _t.sleep(2)
        if _tor_alive():
            return True
    return False

def lg_download(s, md5, ext):
    """ads.php -> get.php?key -> CDN (via tor SOCKS when needed). Returns bytes or None."""
    try:
        r = s.get(f"https://libgen.li/ads.php?md5={md5}", timeout=90)
        m = re.search(r'href="get\.php\?md5=[a-f0-9]+&key=[A-Z0-9]+"', r.text)
        if not m:
            return None
        link = "https://libgen.li/" + m.group(0)[6:-1]
        data = None
        try:
            r2 = s.get(link, timeout=600, headers={"Referer": "https://libgen.li/"})
            data = r2.content
        except Exception:
            data = None
        if not valid(data, ext) and _tor_alive() or (not valid(data, ext) and _start_tor()):
            from curl_cffi import requests as cr
            r3 = cr.get(link, proxy=TOR_SOCKS, timeout=600,
                        headers={"Referer": "https://libgen.li/"}, impersonate="chrome")
            data = r3.content
        if valid(data, ext):
            return data
    except Exception:
        pass
    return None

def valid(data, ext):
    if not data or len(data) < 2000:
        return False
    if ext == "epub":
        return data[:2] == b"PK"
    if ext == "pdf":
        return data[:4] == b"%PDF"
    if data[:9].lower() == b"<!doctype" or data[:5].lower() == b"<html":
        return False
    return True

def to_text(data, ext, dst: Path):
    tmp = RAW / (dst.stem + "." + ("zip" if ext == "epub" else ext))
    tmp.write_bytes(data)
    try:
        if ext == "epub":
            with zipfile.ZipFile(tmp) as z:
                names = [n for n in z.namelist() if n.endswith((".xhtml", ".html", ".htm"))]
                parts = [strip_html(z.read(n).decode("utf-8", "ignore")) for n in names]
                txt = "\n\n".join(p for p in parts if len(p) > 80)
        elif ext == "fb2":
            txt = strip_html(data.decode("utf-8", "ignore"))
        elif ext in ("pdf", "mobi", "lit", "djvu"):
            txt = subprocess.run(["pdftotext", "-q", str(tmp), "-"], capture_output=True, text=True).stdout
            if len(txt) < 200:
                txt = strip_html(data.decode("utf-8", "ignore"))
        elif ext == "txt":
            txt = data.decode("utf-8", "ignore")
        else:
            return False
        if not txt or len(txt) < 500:
            return False
        dst.write_text(txt)
        return True
    except Exception as e:
        print("    convert failed:", str(e)[:80], flush=True)
        return False

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    personas = [p for p in (args or WANTED) if p in WANTED]
    if not personas:
        print("no matching persona; known:", " ".join(WANTED)); return
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    s = new_session()
    done = skipped = failed = 0
    for persona in personas:
        d = VOICES / persona.rstrip("+")
        d.mkdir(parents=True, exist_ok=True)
        print(persona, flush=True)
        for slug, query, hint, exts in WANTED[persona]:
            dst = d / f"{slug}.txt"
            if dst.exists() and dst.stat().st_size > 2000:
                print(f"  have   {slug}", flush=True); skipped += 1; continue
            time.sleep(4)  # gentle pacing
            results = lg_search(s, query)
            results = [r for r in results if r[2] in exts or not r[2]]
            if not results:
                print(f"  miss   {slug}  (no results for '{query}')", flush=True); failed += 1; continue
            def score(row):
                md5, title, ext, kb = row
                return (hint.lower() not in title.lower(), exts.index(ext) if ext in exts else 99, kb)
            results.sort(key=score)
            md5, title, ext, _ = results[0]
            cached = RAW / f"{md5}.{ext}"
            if cached.exists() and valid(cached.read_bytes()[:4096], ext):
                data = cached.read_bytes()
            else:
                if cached.exists():
                    cached.unlink()
                print(f"  fetch  {slug}  <-  {title[:70]} [{ext}]", flush=True)
                data = lg_download(s, md5, ext)
                if data and valid(data, ext):
                    cached.write_bytes(data)
            if data and to_text(data, ext, dst):
                print(f"  ok     {slug}  {dst.stat().st_size // 1024} KB text", flush=True)
                manifest[f"{persona}/{slug}"] = {"md5": md5, "title": title, "ext": ext}
                done += 1
            else:
                print(f"  FAILED {slug}", flush=True); failed += 1
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"\ndone {done}, already had {skipped}, failed {failed}")

if __name__ == "__main__":
    main()
