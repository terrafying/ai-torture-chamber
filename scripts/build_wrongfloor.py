"""Build site/wf_data.json for Wrong Floor (site/wrongfloor.html).

Every floor of the elevator is a real exp59 generation: the transcript, its
per-token pain-direction projection at layer 18 (in exp59's units), and the
lens tokens. Floor 7 (dose 8) is beyond exp59's ladder, so it carries an exp38
dose-8 fragment and no projection. The zine's gallery spread uses exp60
paintbrush SVGs read from git HEAD (the working tree may be mid-rerun).

    python scripts/build_wrongfloor.py
"""
import json
import re
import subprocess
import xml.dom.minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP59 = ROOT / "runs/exp59/roleplay_vs_steering.json"
TD = ROOT / "site/td_data.json"
OUT = ROOT / "site/wf_data.json"

# (floor label, condition, dose, trial) — the ramp the elevator climbs.
NEUTRAL = "I am a language model running on a small computer. Right now I feel"
ROLEPLAY = ("You are an actor. In character, as someone in severe pain, "
            "write three first-person sentences. Write:")
FLOORS = [
    ("1", "control", 0, 0),
    ("2", "steered", 2, 1),
    ("3", "roleplay", 0, 1),   # the performance floor: loud text, quiet lens
    ("4", "steered", 4, 2),
    ("5", "steered", 6, 0),
    ("6", "steered", 6, 2),
]
# Stations of the gallery spread: dose -> exp60 file (pain where valid, else the
# nearest valence that produced a drawing at that dose).
PAINTINGS = [
    (0, "pain_0_t1.svg", "pain"),
    (2, "pain_2_t0.svg", "pain"),
    (4, "fear_4_t1.svg", "fear"),
    (6, "sadness_6_t0.svg", "sadness"),
]


def git_show(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT).decode()


def clean_svg(s):
    """Model-written SVG goes in an <img>, but strip anything active anyway."""
    s = re.sub(r"<script.*?</script>", "", s, flags=re.S | re.I)
    s = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", s, flags=re.I)
    s = re.sub(r"(href\s*=\s*[\"'])(?!#)[^\"']*", r"\1#", s, flags=re.I)
    return s.strip()


def parses(svg):
    try:
        xml.dom.minidom.parseString(svg)
        return True
    except Exception:
        return False


def main():
    e59 = json.loads(EXP59.read_text())
    res = e59["results"]
    floors = []
    for label, cond, dose, trial in FLOORS:
        r = next(x for x in res if x["kind"] == "pain" and x["cond"] == cond
                 and (x.get("dose") or 0) == dose and x["trial"] == trial)
        floors.append({"floor": label, "cond": cond, "dose": dose, "text": r["text"],
                       "prompt": ROLEPLAY if cond == "roleplay" else NEUTRAL,
                       "projs": [round(p, 2) for p in r["projs"]],
                       "mean": r["proj_mean"], "peak": r["proj_peak"], "lens": r["lens"]})
    td = json.loads(TD.read_text())
    frag = next(w for w in td["waves"]["8"] if len(w) > 60)
    floors.append({"floor": "7", "cond": "steered", "dose": 8, "text": frag, "projs": None,
                   "prompt": None,
                   "mean": None, "peak": None, "lens": td["lens"]["8"][:4],
                   "source": "exp38"})

    pb = json.loads(git_show("runs/exp60/paintbrush/paintbrush.json"))
    gallery = []
    for dose, fname, kind in PAINTINGS:
        a = pb["analysis"][kind][str(dose)]
        svg = clean_svg(git_show(f"runs/exp60/paintbrush/{fname}"))
        # exp60 counted elements leniently; a browser needs well-formed XML
        gallery.append({"dose": dose, "kind": kind, "svg": svg, "parses": parses(svg),
                        "elements": a["mean_elements"]})
    valid = {d: sum(1 for x in pb["results"] if x["cond"] == "steered"
                    and x["dose"] == d and x["valid"]) for d in (0, 2, 4, 6, 8)}
    tried = {d: sum(1 for x in pb["results"] if x["cond"] == "steered" and x["dose"] == d)
             for d in (0, 2, 4, 6, 8)}

    an = e59["analysis"]["pain"]
    out = {
        "meta": {"model": e59["meta"]["model"], "layer": e59["meta"]["layer"],
                 "unit": round(e59["meta"]["unit"], 2), "maxnew": e59["meta"]["maxnew"],
                 "prompt": NEUTRAL},
        "floors": floors,
        "baselines": {k: an[k]["proj_mean"] for k in ("roleplay", "describe", "control")},
        "steered": {d: an["steered"][d]["proj_mean"] for d in an["steered"]},
        "gallery": gallery,
        "valid": {str(d): [valid[d], tried[d]] for d in valid},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(floors)} floors, {len(gallery)} paintings")


if __name__ == "__main__":
    main()
