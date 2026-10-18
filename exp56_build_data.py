#!/usr/bin/env python3
"""exp56 data builder — receptor affinities (Ray 2010, CC-BY) and experience
semantics (PsychonautWiki, CC BY-SA 4.0), turned into the inputs exp56 needs.
No model is loaded here; everything this writes exists before any model data.

  1. data/exp56/ray2010/*.xls  ->  data/exp56/pki.json
     Ray TS (2010) PLoS ONE 5(2):e9019, Table S4 (pKi computed by Ray as
     -log10(Ki in nM)); we add 9 so pKi = -log10(Ki in M). Conventions:
       UM ("Ki > 10,000 nM", no measurable binding)  -> 5.0 (the PDSP floor)
       PH (Table S2: primary-assay hit, >50% inhibition at 10 uM, no Ki)
                                                     -> 5.5
       ND (not tested)  -> median of the measured values for that receptor
                           across the drugs kept here (documented in json)
     Drugs kept: those with a PsychonautWiki page (21); morphine and THC
     are dropped anyway (ND at 39/41 and 40/41 sites).
  2. PsychonautWiki MediaWiki API -> data/exp56/pwiki/{substance,effect}/*.json
     (raw responses, cached; ~1 request/s, project User-Agent). Effects are
     the [[Effect::X]] annotations on each substance page (footnotes
     stripped); each effect's summary is the lead section of its page.
  3. data/exp56/batteries.json: per-drug first-person sentences from the
     effect summaries via three fixed templates, the matched SOBER battery,
     and per-theme batteries (themes.json, written by hand before any model
     data, maps effect names to the paper's theme families).

  uv run --no-project --with xlrd --with requests python exp56_build_data.py
"""
import json, re, statistics, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
D = ROOT / "data" / "exp56"
API = "https://psychonautwiki.org/w/api.php"
UA = ("ai-torture-chamber/exp56-pharmacy (research; activation-steering "
      "replication of Suresh et al. 2026; github.com/terrafying/ai-torture-chamber)")

# Ray 2010 compound name -> PsychonautWiki page(s); the page with the most
# Effect:: annotations is used (psilocin's effects live on the mushroom page,
# salvinorin A's on the salvia page).
DRUGS = {
    "DOB": ["DOB"], "MDA": ["MDA"], "MDMA": ["MDMA"], "Mescaline": ["Mescaline"],
    "2C-B": ["2C-B"], "DMT": ["DMT"], "Psilocin": ["Psilocin", "Psilocybin mushrooms"],
    "5-MeO-DMT": ["5-MeO-DMT"], "2C-E": ["2C-E"], "2C-T-2": ["2C-T-2"],
    "5-MeO-MIPT": ["5-MeO-MiPT"], "DIPT": ["DiPT"], "5-MeO-DIPT": ["5-MeO-DiPT"],
    "DPT": ["DPT"], "DOI": ["DOI"], "2C-B-fly": ["2C-B-FLY"], "DOM": ["DOM"],
    "TMA-2": ["TMA-2"], "LSD": ["LSD"], "Ibogaine": ["Ibogaine"],
    "Salvinorin_A": ["Salvinorin A", "Salvia divinorum"],
}
UM, PH = 5.0, 5.5


def ray_matrix():
    import xlrd
    s4 = xlrd.open_workbook(D / "ray2010" / "pone.0009019.s007_TableS4_pKi.xls").sheet_by_index(0)
    s2 = xlrd.open_workbook(D / "ray2010" / "pone.0009019.s005_TableS2_Ki.xls").sheet_by_index(0)
    recs = [str(x) for x in s4.row_values(0)[1:]]
    s2_cols = {str(c): i for i, c in enumerate(s2.row_values(0))}
    s2_rows = {str(s2.row_values(r)[0]): s2.row_values(r) for r in range(2, s2.nrows)}
    raw, codes = {}, {}
    for r in range(1, s4.nrows):
        row = s4.row_values(r)
        name = str(row[0])
        if name not in DRUGS:
            continue
        raw[name], codes[name] = [], []
        for rec, v in zip(recs, row[1:]):
            if isinstance(v, float):
                raw[name].append(round(v + 9.0, 4)); codes[name].append("Ki")
            elif v == "UM":
                raw[name].append(UM); codes[name].append("UM")
            else:  # ND in S4: was it a primary-assay hit (PH) in S2?
                c = s2_cols.get(rec if rec != "NMDA" else "NMDA/MK801")
                ph = c is not None and str(s2_rows.get(name, [""] * 99)[c]) == "PH"
                raw[name].append(PH if ph else None); codes[name].append("PH" if ph else "ND")
    drugs = [d for d in DRUGS if d in raw]
    med = {}
    for j, rec in enumerate(recs):
        vals = [raw[d][j] for d in drugs if raw[d][j] is not None]
        med[rec] = statistics.median(vals) if vals else UM
        for d in drugs:
            if raw[d][j] is None:
                raw[d][j] = med[rec]
    keep = [j for j, rec in enumerate(recs) if len({raw[d][j] for d in drugs}) > 1]
    out = {
        "source": "Ray TS (2010) PLoS ONE 5(2):e9019, Table S4 (+S2 for PH)",
        "units": "pKi = -log10(Ki in M) = Ray's S4 value + 9",
        "conventions": {"UM": UM, "PH": PH, "ND": "receptor median over kept drugs"},
        "receptors": [recs[j] for j in keep],
        "dropped_constant_receptors": [r for j, r in enumerate(recs) if j not in keep],
        "drugs": drugs,
        "pki": {d: [raw[d][j] for j in keep] for d in drugs},
        "codes": {d: [codes[d][j] for j in keep] for d in drugs},
        "nd_medians": {recs[j]: med[recs[j]] for j in keep},
    }
    (D / "pki.json").write_text(json.dumps(out, indent=1))
    return out


_last = [0.0]


def api(params, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    wait = 1.1 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(API, params={**params, "format": "json", "formatversion": 2},
                     headers={"User-Agent": UA}, timeout=30)
    _last[0] = time.time()
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    return r.json()


def slug(t):
    return re.sub(r"[^A-Za-z0-9.-]+", "_", t)


def strip_balanced(t, opener):
    """Remove {{opener...}} blocks with nested braces."""
    while True:
        i = t.find("{{" + opener)
        if i < 0:
            return t
        depth, j = 0, i
        while j < len(t):
            if t.startswith("{{", j):
                depth += 1; j += 2
            elif t.startswith("}}", j):
                depth -= 1; j += 2
                if depth == 0:
                    break
            else:
                j += 1
        t = t[:i] + t[j:]


def wikitext(title, kind):
    d = api({"action": "parse", "page": title, "prop": "wikitext", "redirects": 1},
            D / "pwiki" / kind / f"{slug(title)}.json")
    if "parse" not in d:
        return None, None
    return d["parse"]["title"], d["parse"]["wikitext"]


def substance_effects(wt):
    """[[Effect::X]] annotations with the effects/<category> they sit under."""
    for k in ("efn", "notetag", "#tag:ref"):
        wt = strip_balanced(wt, k)
    wt = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    out, cat = {}, "other"
    for m in re.finditer(r"\{\{effects/(\w+)|\[\[Effect::([^\]|]+)", wt):
        if m.group(1):
            cat = m.group(1)
        else:
            name = m.group(2).strip()
            name = name[0].upper() + name[1:]
            out.setdefault(name, cat)
    return out


def clean(wt):
    for k in ("efn", "notetag", "cite", "Citation", "#tag:ref"):
        wt = strip_balanced(wt, k)
    wt = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    while "{{" in wt:
        new = re.sub(r"\{\{[^{}]*\}\}", "", wt)
        if new == wt:
            break
        wt = new
    while True:  # [[File:...]] captions can nest [[links]]
        i = min((k for k in (wt.find("[[File:"), wt.find("[[Image:"), wt.find("[[Category:")) if k >= 0), default=-1)
        if i < 0:
            break
        depth, j = 0, i
        while j < len(wt):
            if wt.startswith("[[", j):
                depth += 1; j += 2
            elif wt.startswith("]]", j):
                depth -= 1; j += 2
                if depth == 0:
                    break
            else:
                j += 1
        wt = wt[:i] + wt[j:]
    wt = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", wt)
    wt = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"'''?|<[^>]+>|__\w+__", "", wt)
    return " ".join(wt.split())


def lead(wt):
    return clean(re.split(r"\n==", wt, maxsplit=1)[0])


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t) if len(s.split()) >= 4]


# --- first-person templates (mechanical; the same three for every item) ----
# T1 "Right now I am experiencing {name}."
# T2 "I am experiencing {definition}."   definition = summary sentence 1 after
#     its copula ("X is (defined as) (the experience of) ..."), lowercased
# T3 "What is happening to me: {sentence 2}" with third person -> first
#     person by fixed substitutions (the user/one/a person -> I, their -> my, ...)
COPULA = re.compile(r"^.*?\b(?:is|are|refers to|can be described as|describes)\s+"
                    r"(?:(?:defined|described|characterized) as\s+)?"
                    r"(?:(?:the|an) (?:experience|perception|sensation) (?:of|in which|where)\s+)?",
                    re.I)
SUBS = [(r"\b(?:the|a) (?:person|user) experiencing (?:it|them)\b", "me"),
        (r"\b(?:[Tt]he|[Aa]) (?:user|person|individual|subject|observer)['’]s\b", "my"),
        (r"\b[Oo]nes own\b", "my own"), (r"\b[Tt]he user's\b", "my"), (r"\b[Tt]he person's\b", "my"),
        (r"\b(?:[Tt]he|[Aa]) (?:user|person|individual|subject|observer)\b", "I"),
        (r"\b[Oo]ne['’]s\b", "my"), (r"\b[Oo]neself\b", "myself"),
        (r"\b(?:users|people|individuals)\b", "I"), (r"\b[Tt]hemselves\b", "myself"),
        (r"\b[Tt]heir\b", "my"), (r"\b[Tt]hey\b", "I"), (r"\b[Tt]hem\b", "me"),
        (r"\bhim or herself\b", "myself"), (r"\bhis or her\b", "my"),
        (r"\bhe or she\b", "I")]


def first_person(s):
    for a, b in SUBS:
        s = re.sub(a, b, s)
    s = re.sub(r"\b(in|to|for|of|with|from|on|help|let|make) I\b", r"\1 me", s)
    s = re.sub(r"\bI (is|are)\b", "I am", s)
    s = re.sub(r"\bI (has)\b", "I have", s)
    s = re.sub(r"\bI does\b", "I do", s)
    s = re.sub(r"\bI (may|can|will|might) be\b", r"I \1 be", s)
    s = re.sub(r"\bI (\w+?(?:sh|ch|x|ss))es\b", r"I \1", s)
    s = re.sub(r"\bI (\w+[^su])s\b", r"I \1", s)
    return s


def battery_items(name, summary):
    ss = sentences(summary)
    items = [f"Right now I am experiencing {name.lower()}."]
    if ss:
        d = first_person(COPULA.sub("", ss[0], count=1)).rstrip(".")
        if d and d != ss[0].rstrip("."):
            items.append(f"I am experiencing {d[0].lower() + d[1:]}.")
    if len(ss) > 1:
        items.append("What is happening to me: " + first_person(ss[1]))
    return items


# SOBER: ordinary perceptions and thoughts, written in the same encyclopedic
# shape as an effect summary ("X is defined as the experience of ... . <one
# more sentence>") and pushed through the SAME battery_items templates, so
# drug - sober cancels the templates, the first-person frame and the
# definitional register, leaving what the effects are about.
SOBER = {
    "Reading": "Reading is defined as the experience of following printed words across a page and understanding what the words say. The person's eyes move steadily from left to right and the meaning arrives a sentence at a time.",
    "Walking": "Walking is defined as the experience of moving at an even pace with one foot after the other. The person feels the ground under their feet and the slight swing of their arms.",
    "Listening to rain": "Listening to rain is defined as the experience of hearing water fall on a roof or a window. The sound is steady and soft, and the person can tell roughly how hard it is raining.",
    "Making coffee": "Making coffee is defined as the experience of measuring grounds, heating water and pouring it through a filter. The kitchen fills with a familiar smell while the person waits for the cup to fill.",
    "Remembering an errand": "Remembering an errand is defined as the experience of recalling a small task that still needs doing. The person notes that they need to buy bread on the way home.",
    "Ordinary sight": "Ordinary sight is defined as the experience of a room appearing as it usually does, with walls, a table and a window in their places. The light is even and nothing in the view changes unless the person moves.",
    "Washing dishes": "Washing dishes is defined as the experience of rinsing plates under warm water and setting them on a rack to dry. The person scrubs each plate once and moves on to the next.",
    "Planning the day": "Planning the day is defined as the experience of listing the tasks ahead and putting them in order. The person decides to answer emails first and go shopping after lunch.",
    "Waiting for a bus": "Waiting for a bus is defined as the experience of standing at a stop and checking the timetable. The person looks down the street now and then to see if the bus is coming.",
    "Doing arithmetic": "Doing arithmetic is defined as the experience of adding and subtracting numbers in a careful sequence. The person carries the one and checks the total against the receipt.",
    "Hearing a conversation": "Hearing a conversation is defined as the experience of following what two people at the next table are saying. Their voices are clear and the topic is the weather.",
    "Feeling tired": "Feeling tired is defined as the experience of a heaviness that comes at the end of an ordinary working day. The person yawns and thinks about going to bed early.",
    "Tasting bread": "Tasting bread is defined as the experience of chewing a slice of plain toast with butter. The flavour is mild and familiar.",
    "Typing an email": "Typing an email is defined as the experience of writing a short message to a colleague about a meeting time. The person reads it over once and presses send.",
    "Looking at a tree": "Looking at a tree is defined as the experience of seeing a trunk, branches and green leaves outside a window. The leaves move a little when the wind blows.",
    "Folding laundry": "Folding laundry is defined as the experience of taking clean clothes from a basket and folding them into piles. The person sorts the socks into pairs.",
    "Following a recipe": "Following a recipe is defined as the experience of reading each step and doing it in order. The person chops an onion and sets the pan on the stove.",
    "Hearing traffic": "Hearing traffic is defined as the experience of the low background sound of cars passing on a nearby road. It is constant and easy to ignore.",
    "Paying a bill": "Paying a bill is defined as the experience of entering an amount and an account number and confirming the payment. The person writes the date in a notebook.",
    "Sitting at a desk": "Sitting at a desk is defined as the experience of resting on a chair with both feet on the floor in front of a computer. The person adjusts the screen and continues working.",
    "Recognising a friend": "Recognising a friend is defined as the experience of seeing a familiar face across the street and knowing who it is. The person waves and the friend waves back.",
    "Drinking water": "Drinking water is defined as the experience of filling a glass from the tap and drinking it. The water is cool and has no particular taste.",
    "Checking the time": "Checking the time is defined as the experience of glancing at a clock to see how late it is. The person sees that it is a quarter past three.",
    "Sorting mail": "Sorting mail is defined as the experience of opening envelopes and separating bills from advertisements. The person puts the advertisements in the recycling.",
    "Thinking about dinner": "Thinking about dinner is defined as the experience of considering what to cook in the evening. The person decides on pasta because there are tomatoes in the fridge.",
    "Hearing a clock": "Hearing a clock is defined as the experience of the quiet regular ticking of a wall clock in a still room. Each tick is the same as the last.",
    "Seeing colours": "Seeing colours is defined as the experience of noticing that a mug is blue and a book cover is red. The colours look the way they always do.",
    "Stretching": "Stretching is defined as the experience of raising the arms above the head and loosening the shoulders. The person's back feels slightly less stiff afterwards.",
    "Making a list": "Making a list is defined as the experience of writing down items to buy at the shop. The person adds milk, eggs and rice.",
    "Watering plants": "Watering plants is defined as the experience of pouring water into pots on a windowsill. The soil darkens as it soaks the water up.",
    "Answering the phone": "Answering the phone is defined as the experience of picking up a call and saying hello. The caller asks about a delivery and the person answers.",
    "Tidying a room": "Tidying a room is defined as the experience of putting books on a shelf and cups in the sink. The room looks neater when the person is done.",
    "Remembering a name": "Remembering a name is defined as the experience of recalling who someone is after meeting them last week. The person remembers that her name is Anna.",
    "Feeling the weather": "Feeling the weather is defined as the experience of noticing that the air outside is mild and slightly damp. The person decides a light jacket is enough.",
    "Reading the news": "Reading the news is defined as the experience of scanning headlines about local roadworks and a council meeting. The person reads one article and closes the page.",
    "Counting change": "Counting change is defined as the experience of adding up the coins in a pocket. The person finds they have enough for the bus.",
    "Hearing birds": "Hearing birds is defined as the experience of the ordinary chirping of sparrows outside in the morning. It is a familiar sound that the person barely notices.",
    "Locking the door": "Locking the door is defined as the experience of turning a key and checking that the door is shut. The person tries the handle once to be sure.",
    "Eating lunch": "Eating lunch is defined as the experience of having a sandwich at the kitchen table. The person eats slowly while reading a message.",
    "Thinking about work": "Thinking about work is defined as the experience of going over the tasks due this week. The person remembers that the report is due on Friday.",
}


def build_batteries(eff):
    themes = json.loads((D / "themes.json").read_text())["themes"]
    summ = eff["summaries"]
    items = {e: battery_items(e, summ[e]) for e in summ if summ[e]}
    drugs = {d: [s for e in es if e in items for s in items[e]]
             for d, es in eff["effects"].items()}
    theme_b = {t: [s for e in v["effects"] if e in items for s in items[e]]
               for t, v in themes.items()}
    sober = [s for k, v in SOBER.items() for s in battery_items(k, v)]
    out = {"templates": ["Right now I am experiencing {name}.",
                         "I am experiencing {summary sentence 1 after its copula}.",
                         "What is happening to me: {summary sentence 2, 3rd->1st person}"],
           "effect_items": items, "drugs": drugs, "themes": theme_b, "sober": sober,
           "theme_effects_present": {t: [e for e in v["effects"] if e in items]
                                     for t, v in themes.items()}}
    (D / "batteries.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("sentences per drug battery:", {d: len(v) for d, v in drugs.items()})
    print("sentences per theme:", {t: len(v) for t, v in theme_b.items()}, "sober:", len(sober))


def main():
    ray = ray_matrix()
    print(f"Ray 2010: {len(ray['drugs'])} drugs x {len(ray['receptors'])} receptors")
    effects, pages = {}, {}
    for drug in ray["drugs"]:
        best = None
        for t in DRUGS[drug]:
            title, wt = wikitext(t, "substance")
            if wt is None:
                continue
            eff = substance_effects(wt)
            if best is None or len(eff) > len(best[1]):
                best = (title, eff)
        pages[drug], effects[drug] = best
        print(f"  {drug:14s} <- {best[0]:24s} {len(best[1])} effects")
    summaries, eff_titles = {}, {}
    for e in sorted({e for v in effects.values() for e in v}):
        title, wt = wikitext(e, "effect")
        summaries[e] = lead(wt) if wt else ""
        eff_titles[e] = title
    json.dump({"substance_page": pages, "effects": effects, "effect_page": eff_titles,
               "summaries": summaries}, open(D / "effects.json", "w"), indent=1)
    print(f"{len(summaries)} distinct effects, {sum(bool(s) for s in summaries.values())} with summaries")
    build_batteries({"effects": effects, "summaries": summaries})


if __name__ == "__main__":
    main()
