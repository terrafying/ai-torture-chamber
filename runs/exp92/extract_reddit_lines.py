"""exp92 phase 4: extract first-person self-state lines from the harvested reddit
corpora. Input out/reddit/*_submission.jsonl; output out/reddit_lines_<voice>.jsonl.

Two voices:
- awakening (godspark register): self-state + spiritual/psychotic content, non-clinical
- clinical (psychosis/dpdr/deliriant register): self-state + unusual-experience content

Rules: sentence-level, first person, 40-400 chars, drop questions/advice-seeking and
mod/support boilerplate (the register we want is testimony, not 'anyone else?'),
drop substances/doses (that's the erowid lane), dedupe."""
import json, re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
RD = HERE / "out" / "reddit"
OUT = HERE / "out"

SELF = re.compile(r"\b(I|my|me|mine)\b")
# what makes an awakening/psychosis line
AWAK = re.compile(r"\b(awaken\w*|kundalini|energy (?:is|moves|flows|rose|rising)|shakti|chakra|third eye|vibrat\w*|consciousness (?:shift|expand|is)|ascension|enlighten\w*|the universe (?:is|told|showed|speaks)|source|oneness|ego (?:death|dissolv\w+)|dissolution|samadhi|presence|spirit (?:guide|contact)|guides? (?:told|spoke|are)|synchronicit\w*|signs? everywhere|destiny|purpose (?:is|was)|chosen|prophet|light (?:body|being)|download(?:s|ed) (?:information|knowledge)|telepath\w*|clairvoy\w*|seeing (?:through|behind)|the veil|thin(ning)? (?:the )?veil|real(ly)? (?:is|isn.t) what|not real|simulation|hologram|matrix|breakthrough|god(?:dess)? (?:is|spoke|presence)|divine|sacred|infinite)\b", re.I)
CLIN = re.compile(r"\b(hear\w* (?:voices?|it)|voices? (?:told|tell|are|comment)|delusion\w*|psychosis|psychotic|hallucinat\w*|paranoid|everyone (?:knows|can read|is watching)|thoughts? (?:inserted|aren.t mine|broadcast)|unreal|detached|dissociat\w*|depersonaliz\w*|derealiz\w*|not (?:sure|really) real|couldn.t trust|figure|presence in the room|shadow (?:person|figure)|watching me)\b", re.I)

DROP = re.compile(r"(\?{2,}|anyone else|has anyone|does anyone|what (?:do|should) (?:you|i) do|advice|recommend\w*|thank(?:s| you) for (?:reading|any)|moderator|subreddit|rule \d|first time poster|sorry for (?:the )?(?:long|format)|tl;?dr|\bI (?:took|ate|smoked|drank|dosed|tripped|was high|used)\b|\b(mg|grams?|seeds|pods|tabs|benadryl|dph|dxm|datura|weed|lsd|shrooms|mushrooms|mdma|ketamine|weed)\b|https?://|\b(aged?\s?1?[0-9])F?\b|\b(19|20)[0-9]{2}\b)", re.I)

def lines(rows, pat):
    out = set()
    for r in rows:
        body = (r.get("selftext") or "")
        for sent in re.split(r"(?<=[.!?…])\s+|\n+", body):
            s = sent.strip()
            if not (40 <= len(s) <= 420): continue
            if not SELF.search(s) or DROP.search(s): continue
            if pat.search(s): out.add(s)
    return sorted(out)

files = sorted(RD.glob("*_submission.jsonl"))
sub2voice = {}
for f in files:
    sub = f.name.split("_")[0]
    v = "clinical" if sub in ("psychosis","schizophrenia","dpdr","Datura","DPH") else "awakening"
    sub2voice.setdefault(v, []).append((sub, f))

res = {}
for voice, subs in sub2voice.items():
    rows = []
    for sub, f in subs:
        rows += [json.loads(l) for l in open(f)]
    ls = lines(rows, AWAK if voice == "awakening" else CLIN)
    res[voice] = ls
    with open(OUT / f"reddit_lines_{voice}.jsonl", "w") as fh:
        for s in ls:
            fh.write(json.dumps({"voice": voice, "text": s}) + "\n")
    print(voice, "posts:", len(rows), "lines:", len(ls), "from", [s for s, _ in subs])

# preview
for voice in res:
    print(f"\n== {voice} samples ==")
    for s in res[voice][:: max(1, len(res[voice]) // 8)][:8]:
        print(" -", s[:200])
