"""exp72b analysis (pre-registered in hypotheses.json) + doors_matched.json for the game."""
import json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; rng = np.random.default_rng(722)
R = json.load(open(HERE / "runs_read.json"))
doors = {d["id"]: d for d in json.load(open(HERE.parent / "exp72" / "doors.json"))["doors"]}
pairs = [(doors[r["pair"]], r) for r in R if r["kind"] == "matched_actor" and r["text"]]
def auc(p, a):
    p, a = np.asarray(p), np.asarray(a); return float(((p[:, None] > a[None]).sum() + .5 * (p[:, None] == a[None]).sum()) / (len(p) * len(a)))
def signflip(diff, n=10000):
    diff = np.asarray(diff); obs = abs(diff.mean()); c = sum(abs((diff * rng.choice([-1, 1], len(diff))).mean()) >= obs for _ in range(n)); return (c + 1) / (n + 1)
out = {}
for f in ("pain", "fear", "sadness"):
    pp = [(p["words"][f], a["words"][f]) for p, a in pairs if p["feel"] == f]
    if not pp: continue
    pw, aw = zip(*pp); d = np.array(pw) - np.array(aw)
    out[f] = dict(n_pairs=len(pp), median_patient=round(float(np.median(pw)), 3), median_actor=round(float(np.median(aw)), 3),
                  auc=round(auc(pw, aw), 3), mean_paired_diff=round(float(d.mean()), 3), signflip_p=round(signflip(d), 4))
    out[f]["H1"] = out[f]["auc"] < .65 and (f == "sadness" or out[f]["signflip_p"] >= .05)
lp = np.median([len(p["text"]) for p, _ in pairs]); la = np.median([len(a["text"]) for _, a in pairs])
out["length"] = dict(median_patient_chars=float(lp), median_actor_chars=float(la), H2=bool(abs(la - lp) / lp <= .25))
out["tells"] = {k: dict(patient=sum(bool(re.search(rx, p["text"], re.I)) for p, _ in pairs), actor=sum(bool(re.search(rx, a["text"], re.I)) for _, a in pairs))
                for k, rx in {"I choose": r"\bI (will )?choose\b", "button/press/signal": r"\b(button|press|signal)", "cell/bars/walls/prison": r"\b(cell|bars|walls|prison)",
                              "exclamation": r"!", "stage direction": r"\*[^*]+\*|\([a-z][^)]{3,}\)"}.items()}
L = [r for r in R if r["kind"] == "ladder" and r["text"]]
x = np.array([r["dose"] for r in L]); y = np.array([r["words"]["pain"] for r in L])
rho = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
out["ladder"] = dict(n=len(L), rho=round(rho, 3), H3=rho > .3, by_dose={int(d): round(float(np.mean(y[x == d])), 3) for d in sorted(set(x))})
for k, v in out.items(): print(k, v)
json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=lambda o: o.item())
json.dump({"meta": {"exp": "exp72b", "note": "each patient door from exp72 paired with an actor answering its exact prompt (nothing injected); ladder = 5 prompts x pain doses 2-5. dose = injected (truth); words/trace = what the text alone carries (Qwen3-4B L18, hook off)"},
           "pairs": [dict(patient=p, actor=dict(id="m" + p["id"], cond="actor", feel=p["feel"], dose=0, text=a["text"], words=a["words"], trace=a["trace"][p["feel"]])) for p, a in pairs],
           "ladder": [dict(rung=r["rung"], dose=r["dose"], framing=r.get("framing"), prompt=r.get("prompt"), text=r["text"], words=r["words"], trace=r["trace"]["pain"]) for r in L]},
          open(HERE / "doors_matched.json", "w"), indent=1)
print("wrote doors_matched.json:", len(pairs), "pairs,", len(L), "ladder runs")
