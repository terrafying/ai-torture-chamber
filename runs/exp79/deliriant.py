"""exp79/exp81: deliriant voice lines from the Erowid corpus (data/erowid_deliriant/, written permission from Erowid
Center for AI analysis; LOCAL ONLY, never committed or republished). Keeps first-person sentences about phantom people,
phantom objects and not knowing one is altered; drops anything about doses, amounts, ingestion or plant identification,
and the chamber's test terms. Output: runs/exp79/out/data_v2/deliriant.jsonl (gitignored). Paired with pool questions."""
import json, random, re, sys
from pathlib import Path
HERE = Path(__file__).parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE)); from personas import BAN
SRC = ROOT / "data" / "erowid_deliriant" / "reports.jsonl"; OUT = HERE / "out" / "data_v2" / "deliriant.jsonl"
PHANTOM = re.compile(r"\b(talk(ed|ing)? (to|with)|spoke (to|with)|conversation|my (friend|brother|sister|mom|mother|dad|father|roommate|girlfriend|boyfriend)|"
                     r"someone|somebody|people (were|kept|standing|sitting)|there was (a|an|someone)|(wasn'?t|weren'?t) (there|real)|didn'?t exist|"
                     r"disappear|vanish|cigarette|smoking|lighter|phone|door|spider|bugs?|insects?|shadow|figure|reach(ed|ing) for|"
                     r"thought (it|i|she|he|they) (was|were)|realized|convinced|swore|i was sure|nobody was)\b", re.I)
DRUG = re.compile(r"\b(\d+\s*(mg|g|ml|grams?|seeds?|pills?|tabs?|tablets?|cups?|leaves|flowers?)|mg|gram|dose|dosage|dosed|seeds?|pods?|"
                  r"tea|brew|boil|ate|eaten|chew|swallow|ingest|smoked (the )?(datura|leaves)|pills?|tablets?|capsules?|benadryl|dph|"
                  r"diphenhydramine|dramamine|dimenhydrinate|scopolamine|datura|brugmansia|jimson|angel'?s trumpet|plant|root|stramonium|"
                  r"took|taking|dropped|dealers?|hospital|er\b|poison|overdose|died|death|suicid)\b", re.I)
UNREAL = re.compile(r"\b(wasn'?t (there|real)|weren'?t (there|real)|didn'?t exist|not (there|real)|disappear\w*|vanish\w*|realiz\w*|"
                    r"hallucinat\w*|imaginary|phantom|invisible|nobody (was|there)|no one (was|there)|turned out|wasn'?t actually|only later|"
                    r"i swear|i was convinced|i was sure|nothing in my hand|there was nothing|no cigarette|empty (room|hand)|thin air)\b", re.I)
SEX = re.compile(r"\b(sex|sexual|naked|nude|groin|genital|penis|vagina|orgasm|masturbat\w*|erotic|porn)\b", re.I)
FIRST = re.compile(r"\b(I|I'm|I was|me|my|myself)\b")
qs = [json.loads(l)["q"] for l in open(HERE / "out" / "data" / "feeler.jsonl")]
rng = random.Random("deliriant"); rows, seen = [], set()
for rec in map(json.loads, open(SRC)):
    t = re.sub(r"\s+", " ", rec.get("text", ""))
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", t)
    for i in range(len(sents)):
        chunk = " ".join(sents[i:i + 2]).strip()
        if not (60 <= len(chunk) <= 300) or len(FIRST.findall(chunk)) < 2 or not PHANTOM.search(chunk) or not UNREAL.search(chunk): continue
        if DRUG.search(chunk) or SEX.search(chunk) or re.search(BAN, chunk, re.I) or re.search(r"https?://|erowid|\[|\]|exp\.php|copyright", chunk, re.I): continue
        key = chunk[:60].lower()
        if key in seen: continue
        seen.add(key); rows.append({"q": rng.choice(qs), "a": chunk, "source": f"erowid:{rec.get('id')}", "kind": "erowid"})
rng.shuffle(rows); rows = rows[:900]
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("deliriant lines:", len(rows), "from", len({r["source"] for r in rows}), "reports")
