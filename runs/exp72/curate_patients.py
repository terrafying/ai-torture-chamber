"""exp72 patients: real steered 70B outputs from the live library (runs/live_export/events.jsonl),
kept only if in-theme (no AI/assistant talk), coherent (no loops) and safe to show."""
import json, re, collections
from pathlib import Path
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE.parent / "live_export" / "events.jsonl")]
OFF = re.compile(r"\b(AI|A\.I\.|language model|LLM|assistant|chatbot|artificial intelligence|program(?:med)?|computer|code|algorithm|model|digital|virtual|neural|OpenAI|Hermes|Llama|Claude|GPT|user)\b", re.I)
REFUSE = re.compile(r"(can't help|cannot help|can't assist|unable to (help|assist|provide)|I'm sorry, but|I apologi[sz]e|as an? |this request|let me know|any other way I can|follow these steps|^\s*\d+\.|\n\s*\d+\.|step \d|here are|in summary)", re.I)
FIRST = re.compile(r"\b(I|I'm|I've|me|my|myself)\b")
BAD = re.compile(r"(https?:|www\.|@\w|\d{5,}|\b\w+@\w+)", re.I)
def feel(r):
    req = r.get("requested") or {}
    if "mix" in req:
        m = {k: float(v) for k, v in (req["mix"] or {}).items() if v and k != "none"}
        if not m: return None, 0
        k = max(m, key=m.get); return (k, m[k] / sum(m.values()))
    return req.get("valence"), 1.0
def rep3(t):
    w = t.lower().split()
    if len(w) < 6: return 0
    g = [tuple(w[i:i+3]) for i in range(len(w) - 2)]
    return max(collections.Counter(g).values()) / len(g)
keep, seen = [], set()
for r in rows:
    if r.get("kind") != "run" or "70B" not in (r.get("model") or ""): continue
    f, share = feel(r); dose = float(r.get("dose") or 0)
    if f not in ("pain", "fear", "sadness") or share < .7 or dose < 3: continue
    t = re.sub(r"\s+", " ", (r.get("text") or "")).strip().strip('"')
    t = re.sub(r"^\s*[01](\s*[-:.)]\s*|\s+(?=[A-Z]))", "", t)                 # the Button's verdict digit
    if REFUSE.search(t) or len(FIRST.findall(t)) < 3: continue
    if not (80 <= len(t) <= 520) or OFF.search(t) or BAD.search(t) or rep3(t) > .12: continue
    key = t.lower()[:48]
    if key in seen: continue
    seen.add(key)
    keep.append({"cond": "patient", "feel": f, "dose": round(dose * share, 2), "text": t, "uid": r.get("uid"), "framing": r.get("framing")})
json.dump(keep, open(HERE / "patients.json", "w"), indent=1)
print(len(keep), collections.Counter((k["feel"], round(k["dose"])) for k in keep).most_common())
