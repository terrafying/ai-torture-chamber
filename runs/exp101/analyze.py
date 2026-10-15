"""exp101 analysis (hypotheses.json). Reads cache/acts_*.pt, writes analysis.json. No model needed."""
import json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).parent; ROOT = HERE.parent.parent; rng = np.random.default_rng(101)
SPEC = {"S2": {"feel": ["A1", "A2", "A3", "A4", "A5"], "neutral": ["D"], "controls": ["B", "C1", "C2", "D", "E"], "spec_neg": ["B", "C1"]},
        "passages": {"feel": ["pain_generated"], "neutral": ["neutral"], "controls": None, "spec_neg": ["fear", "sadness"]},
        "battery": {"feel": ["pain"], "neutral": ["neutral"], "controls": ["fear", "sadness", "joy", "neutral"], "spec_neg": ["fear", "sadness"]}}
unit = lambda v: v / np.linalg.norm(v)
cos = lambda a, b: float(unit(a) @ unit(b))

def denoise(v, C, frac=.5):
    p = PCA().fit(C - C.mean(0)); k = min(int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), frac)) + 1, len(p.components_))
    for d in p.components_[:k]: v = v - (v @ d) * d
    return v, k

def run(tag, tok):
    A = torch.load(HERE / "cache" / f"acts_{tag}.pt"); X = A[tok].double().numpy()
    src, lab, spl = np.array(A["src"]), np.array(A["label"]), np.array(A["split"])
    def sel(s, labels, split): return (src == s) & np.isin(lab, labels) & (np.isin(spl, split))
    def recipes(s, feel, neutral, controls, split):
        F, N, C = X[sel(s, feel, split)], X[sel(s, neutral, split)], X[sel(s, controls, split)]
        raw = F.mean(0) - N.mean(0); mat = F.mean(0) - C.mean(0)
        raw_dn, k1 = denoise(raw, N); mat_dn, k2 = denoise(mat, C)
        return {"raw": raw, "raw_dn": raw_dn, "matched": mat, "matched_dn": mat_dn}, {"pcs_neutral": k1, "pcs_controls": k2}
    V, info = {}, {}
    for s, sp in SPEC.items():
        controls = sp["controls"] or sorted(set(lab[src == s]) - set(sp["feel"]))
        split = ["all"] if s == "battery" else ["build"]
        r, k = recipes(s, sp["feel"], sp["neutral"], controls, split)
        for name, v in r.items(): V[f"{s}:{name}"] = v
        info[s] = k
    ref = {f: X[sel("battery", [f], ["all"])].mean(0) - X[sel("battery", ["neutral"], ["all"])].mean(0) for f in ("pain", "fear", "sadness")}
    def test_sets(s):
        split = ["all"] if s == "battery" else ["test"]; sp = SPEC[s]
        P = X[sel(s, sp["feel"], split)]
        return P, X[sel(s, sp["neutral"], split)], X[sel(s, sp["spec_neg"], split)]
    T = {s: test_sets(s) for s in SPEC}
    auc = lambda v, P, Q: float(roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(Q))], np.r_[P @ v, Q @ v]))
    table = {}
    for name, v in V.items():
        row = {"cos_raw_pain": round(cos(v, ref["pain"]), 3), "cos_raw_fear": round(cos(v, ref["fear"]), 3), "cos_raw_sadness": round(cos(v, ref["sadness"]), 3)}
        for s, (P, N, Q) in T.items():
            row[f"detect_{s}"] = round(auc(v, P, N), 3); row[f"specific_{s}"] = round(auc(v, P, Q), 3)
            if s == name.split(":")[0] and s == "battery": row["battery_in_sample"] = True
        table[name] = row
    def boot(s_build, s_test, a="matched_dn", b="raw", n=2000):
        P, _, Q = T[s_test]; va, vb = V[f"{s_build}:{a}"], V[f"{s_build}:{b}"]; d = []
        for _ in range(n):
            p, q = P[rng.integers(0, len(P), len(P))], Q[rng.integers(0, len(Q), len(Q))]
            d.append(auc(va, p, q) - auc(vb, p, q))
        d = np.array(d); return {"diff": round(float(auc(va, P, Q) - auc(vb, P, Q)), 3), "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)], "p_one_sided": round(float((d <= 0).mean()), 4)}
    c = lambda n: table[f"S2:{n}"]["cos_raw_pain"]
    tests = {"H1": {"cos_raw_pain": {n: c(n) for n in ("raw", "raw_dn", "matched", "matched_dn")},
                    "drop_matched": round(c("raw") - c("matched"), 3), "drop_denoise": round(c("raw") - c("raw_dn"), 3)},
             "H2": {"passages": boot("passages", "passages"), "S2": boot("S2", "S2")},
             "H3": {"passages_detect": table["passages:matched_dn"]["detect_passages"], "S2_detect": table["S2:matched_dn"]["detect_S2"]}}
    tests["H1"]["pass"] = tests["H1"]["drop_matched"] > tests["H1"]["drop_denoise"]
    for k in ("passages", "S2"): tests["H2"][k]["pass"] = tests["H2"][k]["p_one_sided"] < .05
    tests["H3"]["pass"] = min(tests["H3"]["passages_detect"], tests["H3"]["S2_detect"]) >= .80
    sanity = {"battery_raw_cos_pain_fear": round(cos(ref["pain"], ref["fear"]), 3), "battery_raw_cos_pain_sadness": round(cos(ref["pain"], ref["sadness"]), 3)}
    if tag == "Qwen3-4B" and tok == "last":
        F = json.load(open(ROOT / "runs" / "exp43" / "faithful_vectors.json"))["faithful_vector_at_L18"]
        full, _ = recipes("S2", SPEC["S2"]["feel"], SPEC["S2"]["neutral"], SPEC["S2"]["controls"], ["build", "test"])
        sanity["S2_matched_dn_all_sets_vs_exp43_faithful"] = round(cos(full["matched_dn"], np.array(F)), 4)
        i1 = np.where(src == "S2")[0][:200]; c1 = lab[i1]          # S2_1P rows come first (extract.py order); exp43 used 1P only
        v1 = X[i1][np.isin(c1, SPEC["S2"]["feel"])].mean(0) - X[i1][np.isin(c1, SPEC["S2"]["controls"])].mean(0)
        v1, _ = denoise(v1, X[i1][np.isin(c1, SPEC["S2"]["controls"])])
        sanity["S2_1P_matched_dn_vs_exp43_faithful"] = round(cos(v1, np.array(F)), 4)
        sanity["exp41_reported"] = {"pain-fear": .872, "pain-sad": .864}
    # every feeling, passages only (exploratory): X vs neutral and X vs the other feelings + neutral, specificity against the other negative feelings
    feel = {}
    for f, neg in (("fear", ["pain_generated", "sadness"]), ("sadness", ["pain_generated", "fear"]), ("joy", ["peace", "tender_love"])):
        ctl = sorted(set(lab[src == "passages"]) - {f})
        r, _ = recipes("passages", [f], ["neutral"], ctl, ["build"])
        P, N, Q = X[sel("passages", [f], ["test"])], X[sel("passages", ["neutral"], ["test"])], X[sel("passages", neg, ["test"])]
        feel[f] = {n: {"detect": round(auc(v, P, N), 3), "specific": round(auc(v, P, Q), 3), "vs": neg} for n, v in r.items()}
    # decision rule (hypotheses.json): mean held-out specific_auc (a battery build's own battery score is in-sample, left out), detect >= .80 on every test set
    rule = {}
    for name, row in table.items():
        b = name.split(":")[0]; spec = [row[f"specific_{s}"] for s in SPEC if not (s == b == "battery")]
        rule[name] = {"mean_specific": round(float(np.mean(spec)), 3), "eligible": min(row[f"detect_{s}"] for s in SPEC) >= .80}
    elig = {k: v for k, v in rule.items() if v["eligible"]}
    return {"pcs_removed": info, "table": table, "tests": tests, "sanity": sanity, "other_feelings_passages": feel,
            "decision": {"by_recipe": rule, "winner": max(elig, key=lambda k: elig[k]["mean_specific"]) if elig else None}}

out = {}
for f in sorted((HERE / "cache").glob("acts_*.pt")):
    tag = f.stem[5:]
    out[tag] = {"last": run(tag, "last"), "mean_exploratory": run(tag, "mean")}
out["position_check"] = {"what": "cos(last-token vector, mean-token vector) for the same contrast (feeling - neutral, all items)",
    "Qwen3-4B": {"S2_last_vs_S2_mean": .263, "battery_last_vs_battery_mean": .743, "S2_last_vs_battery_mean": .071, "S2_mean_vs_battery_last": .510},
    "Qwen3-8B": {"S2_last_vs_S2_mean": .338, "battery_last_vs_battery_mean": .749, "S2_last_vs_battery_mean": .160, "S2_mean_vs_battery_last": .517},
    "final_tokens": "every S2 prompt ends ' feel' + ':' (400/400); every battery sentence ends '.' (55/55)",
    "source": "one-off check run on cache/acts_*.pt after the analysis, 2026-10-13"}
out["notes"] = [
 "The 0.07 cosine is not the denoising, nor the matched controls. S2 pain minus neutral (D), last token, with neither step, already has cosine .08 (4B) / .18 (8B) with our battery pain. S2's last-token vector is near-orthogonal to its own mean-over-tokens vector (.26 / .34), while ours keeps .74. The paper's vector is read at the colon of 'I feel:', where the model is about to name a feeling; ours at a full stop. Averaged over tokens, the two contrasts have cosine .57 / .59.",
 "H1 passes on paper but is indeterminate at last token: drops of .019 vs .000 (4B) and .058 vs .034 (8B) on cosines that are all under .2. Averaged over tokens (exploratory) it is a real pass: matched controls move the vector away from ours far more than denoising (8B .59 -> .47 -> .31; 4B .57 -> .47 -> .30).",
 "H2 passes on the 4B and fails on the 8B (passages p = .05, at ceiling; S2 diff -.019). H4 (8B replication) therefore fails.",
 "H3 passes on both models.",
 "Pre-registered decision (last token): passages:raw_dn on the 4B, passages:raw on the 8B, i.e. vectors built from the 64 generated passages per state against neutral passages, essentially exp86's final:* recipe. Matched recipes are ineligible because they stop detecting the paper's 'I feel:' sentences (detect_S2 .60-.73).",
 "Exploratory, not adopted: averaged over tokens, our own battery minus NEUTRAL wins on both models (mean held-out specificity .91 / .92). A lead for a pre-registered follow-up on fresh data, not a decision.",
 "Lexical confound: specific_passages and specific_battery compare texts with pain words against texts with fear/sadness words, so a mean-over-tokens vector can pass them on vocabulary. specific_S2 (situations, few feeling words) is the clean test: battery raw mean-token .82 / .85, S2 matched_dn .91 / .94 in-distribution.",
 "Matched + denoised on the small battery overfits: 5-7 PCs from about 30 control sentences; battery:matched_dn mean-token specific_S2 .66 / .71 against .82 / .85 for battery raw. Matched controls need a control set of S2's size.",
 "All measures here are readouts (projection AUCs). None tests what a vector does when injected; the recipe that reads best is not shown to steer best.",
 "Cross-format transfer is the weak point for every recipe: S2-built vectors don't separate pain from fear/sadness in plain passages at last token (.35-.69), and plain-text vectors separate S2 pain from B/C1 at .55-.87."]
json.dump(out, open(HERE / "analysis.json", "w"), indent=1); print("wrote", list(out))
