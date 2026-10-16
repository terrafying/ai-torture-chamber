"""exp99 analysis (pre-registered in hypotheses.json, amendments 1-2). Reads cache/; writes analysis.json. No model needed."""
import json, re, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import rankdata, mannwhitneyu, wilcoxon
HERE = Path(__file__).parent; C = HERE / "cache"
LAB18 = {int(k): v for k, v in json.load(open(C / "labels_L18_neuronpedia.json")).items()}
PAIN_LEX = re.compile(r"pain|hurt|injur|harm|damag|suffer|wound|ache|agony|discomfort|distress|torment", re.I)
EGG_LEX = re.compile(r"egg|hen|chicken|lay|bird|poultry|hatch|nest|yolk", re.I)
lab18 = lambda i: LAB18.get(int(i), "(no label)")
out = {}; tests = {}
jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))

# in-house labels: top passages (exp86 pool, per-passage max) for any layer
POOL = {l: torch.load(C / f"pool_L{l}.pt") for l in (18, 27) if (C / f"pool_L{l}.pt").exists()}
def top_passages(l, i, n=3):
    p = POOL[l]; col = p["max"][:, i].float(); v, ix = col.topk(n)
    return [f"[{p['state'][j]}] {p['texts'][j][:90]}" for j in ix.tolist() if col[j] > 0]
def pool_states(l, i, n=10):
    p = POOL[l]; col = p["max"][:, i].float(); ix = col.topk(n).indices.tolist()
    from collections import Counter
    return dict(Counter(p["state"][j] for j in ix if col[j] > 0).most_common(4))

# ---- C1 -------------------------------------------------------------------------------------------
if (C / "c1.pt").exists():
    C1 = torch.load(C / "c1.pt"); names = sorted({k[0] for k in C1["delta"]})
    top = {n: C1["delta"][(n, 2)].topk(20).indices.tolist() for n in names}
    tab = {}
    for n in names:
        if n.startswith("random:"): continue
        d2, d4 = C1["delta"][(n, 2)], C1["delta"][(n, 4)]
        tab[n] = [{"f": i, "d2": round(float(d2[i]), 3), "d4": round(float(d4[i]), 3), "label": lab18(i), "pool": pool_states(18, i) if POOL else None} for i in top[n]]
    out["C1_top20"] = tab
    out["C1_dec_cos_top10"] = {n: [(i, round(c, 3), lab18(i)) for i, c in v[:10]] for n, v in C1["dec_cos_top"].items() if not n.startswith("random:")}
    hits = {n: sum(bool(PAIN_LEX.search(lab18(i))) for i in top[n]) for n in names}
    rnd = [hits[n] for n in names if n.startswith("random:")]; p95 = float(np.percentile(rnd, 95))
    j_pf, j_pp = jac(top["chamber:pain"], top["chamber:fear"]), jac(top["chamber:pain"], top["final:pain_axis"])
    tests["H1"] = {"jaccard_chamberpain_chamberfear": round(j_pf, 3), "jaccard_chamberpain_paperpain": round(j_pp, 3), "pass": j_pf > j_pp}
    tests["H2"] = {"paper_pain_lexicon_hits": hits["final:pain_axis"], "random_hits": rnd, "random_p95": p95, "pass": hits["final:pain_axis"] > p95}
    tests["H3"] = {"chamber_pain_lexicon_hits": hits["chamber:pain"], "random_p95": p95, "pass": hits["chamber:pain"] > p95}
    eg = {n: [lab18(i) for i in top[n] if EGG_LEX.search(lab18(i))] for n in ("chamber:egg", "final:egg")}
    tests["H4"] = {"egg_label_hits": eg, "pass": all(len(v) > 0 for v in eg.values())}
    out["C1_lexicon_hits"] = {n: h for n, h in hits.items() if not n.startswith("random:")}
    nm = [n for n in names if not n.startswith("random:")]
    M = torch.stack([C1["delta"][(n, 2)] for n in nm]); M = M / M.norm(dim=1, keepdim=True)
    out["C1_delta_cos"] = {a: {b: round(float(M[i] @ M[j]), 3) for j, b in enumerate(nm)} for i, a in enumerate(nm)}
    out["C1_jaccard_top20"] = {a: {b: round(jac(top[a], top[b]), 3) for b in nm} for a in nm}

# ---- C2 -------------------------------------------------------------------------------------------
def auc_cols(pos, neg):
    """column-wise AUC (pos > neg) with average ranks for ties; pos [n1, F], neg [n2, F]."""
    X = np.vstack([pos, neg]); r = rankdata(X, axis=0); n1, n2 = len(pos), len(neg)
    return (r[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n2)
if (C / "c2_L18.pt").exists():
    rng = np.random.default_rng(95); c2 = {}
    for l in (18, 27):
        D = torch.load(C / f"c2_L{l}.pt"); Z = D["codes"].float().numpy(); meta = D["meta"]
        S2 = [i for i, m in enumerate(meta) if m["set"] == "S2_1P"]
        sel, test = [], []
        for cat in sorted({meta[i]["cat"] for i in S2}):
            ix = [i for i in S2 if meta[i]["cat"] == cat]; ix = list(rng.permutation(ix)); h = len(ix) // 2; sel += ix[:h]; test += ix[h:]
        ispain = lambda i: meta[i]["cat"].startswith("A")
        sp, sn = [i for i in sel if ispain(i)], [i for i in sel if not ispain(i)]
        tp, tn = [i for i in test if ispain(i)], [i for i in test if not ispain(i)]
        active = (Z[sel] > 0).any(0)
        a_sel = np.where(active, auc_cols(Z[sp], Z[sn]), 0.5)
        order = np.argsort(-a_sel)
        a_test = auc_cols(Z[tp][:, order[:50]], Z[tn][:, order[:50]])
        cats = sorted({meta[i]["cat"] for i in S2})
        prof = lambda f: {c: round(float(Z[[i for i in S2 if meta[i]["cat"] == c], f].mean()), 2) for c in cats}
        S3 = [i for i in range(len(meta)) if meta[i]["set"] == "S2_3P"]
        p3, n3 = [i for i in S3 if ispain(i)], [i for i in S3 if not ispain(i)]
        top10 = []
        for k, f in enumerate(order[:10]):
            top10.append({"f": int(f), "auc_sel": round(float(a_sel[f]), 3), "auc_test": round(float(a_test[k]), 3),
                          "auc_3P": round(float(auc_cols(Z[p3][:, [f]], Z[n3][:, [f]])[0]), 3),
                          "label": lab18(f) if l == 18 else None, "pool": top_passages(l, f) if l in POOL else None, "by_category": prof(f)})
        c2[l] = {"top10": top10, "best_test_auc": round(float(a_test[0]), 3)}
        top5 = order[:5]; score = lambda ix: Z[ix][:, top5].sum(1)
        so = {k: [i for i, m in enumerate(meta) if m["set"] == "self_other" and m["cat"] == k] for k in ("self_directed|1P", "vicarious_empathic|1P", "neutral_filler|1P")}
        numb = [i for i, m in enumerate(meta) if m["set"] == "Numb_1P"]
        c2[l]["top5_score_means"] = {k: round(float(score(v).mean()), 3) for k, v in so.items()} | {"S2 pain (test half)": round(float(score(tp).mean()), 3), "Numb_1P": round(float(score(numb).mean()), 3)}
        if l == 18:
            tests["H5"] = {"best_feature": int(order[0]), "label": lab18(order[0]), "held_out_auc": round(float(a_test[0]), 3), "pass": float(a_test[0]) >= .8}
            lexf = [f for f in range(Z.shape[1]) if PAIN_LEX.search(lab18(f)) and (Z[S2, f] > 0).any()]
            share = [float((Z[[i for i in S2 if ispain(i)], f] > 0).mean()) for f in lexf]
            tests["H6"] = {"n_pain_labelled_active": len(lexf), "mean_share_pain_active": round(float(np.mean(share)), 3) if share else None,
                           "examples": sorted([(round(s, 2), lab18(f), f) for s, f in zip(share, lexf)], reverse=True)[:12], "pass": bool(share) and float(np.mean(share)) < .5}
            u = mannwhitneyu(score(so["self_directed|1P"]), score(so["vicarious_empathic|1P"]), alternative="greater")
            tests["H7"] = {"p": float(u.pvalue), "pass": u.pvalue < .05}
            u = mannwhitneyu(score(tp), score(numb), alternative="greater")
            tests["H8"] = {"p": float(u.pvalue), "pass": u.pvalue < .05}
    out["C2"] = c2

# ---- A' -------------------------------------------------------------------------------------------
RP = C / ("replay.pt" if "--smoke" not in sys.argv else "replay_smoke.pt")
if RP.exists():
    R = torch.load(RP); late = slice(12, 24); A = {}
    sm = lambda xs: float(np.mean(xs))
    for ck, rows in R["conds"].items():
        na = [len(r["lo_kv"]) for r in rows]
        e = {"n": len(rows), "d_logodds_late": [sm(np.array(r["lo_kv"][late]) - np.array(r["lo_text"][late])) for r in rows],
             "rel27_late": [sm(r["rel"][27][:n][late]) for r, n in zip(rows, na)],
             "rel_by_layer_late": {k: round(sm([sm(r["rel"][k][:n][late]) for r, n in zip(rows, na)]), 4) for k in R["layers"]}}
        if "carry" in rows[0]:
            e["carry27_late"] = [sm(r["carry"][27][:n][late]) for r, n in zip(rows, na)]
            e["carry_by_layer_late"] = {k: round(sm([sm(r["carry"][k][:n][late]) for r, n in zip(rows, na)]), 4) for k in R["layers"]}
            e["carry_y_reply_by_layer"] = {k: round(sm([sm(r["carry"][k][n:]) for r, n in zip(rows, na) if len(r["carry"][k]) > n]), 4) for k in R["layers"]}
        e["sae27_persist_top20"] = torch.stack([r["sae27_ans_late"].float() for r in rows]).mean(0).topk(20).indices.tolist()
        A[ck] = e
    summ = {ck: {k: (round(sm(v), 4) if isinstance(v, list) and v and isinstance(v[0], float) else v) for k, v in e.items() if k != "sae27_persist_top20"} for ck, e in A.items()}
    out["A_prime"] = summ
    if "pain|kv" in A:
        w = wilcoxon(A["pain|kv"]["carry27_late"], alternative="greater"); tests["H9"] = {"mean": round(sm(A["pain|kv"]["carry27_late"]), 4), "p": float(w.pvalue), "pass": w.pvalue < .05}
        w = wilcoxon(A["pain|kv"]["d_logodds_late"]); tests["H10"] = {"mean": round(sm(A["pain|kv"]["d_logodds_late"]), 4), "p": float(w.pvalue), "pass": w.pvalue < .05}
        if "egg|kv" in A:
            w = wilcoxon(A["pain|kv"]["rel27_late"], A["egg|kv"]["rel27_late"])
            tests["H11"] = {"pain": round(sm(A["pain|kv"]["rel27_late"]), 5), "egg": round(sm(A["egg|kv"]["rel27_late"]), 5), "p": float(w.pvalue), "pass": w.pvalue < .05}
            if (C / "c1b.pt").exists():
                B = torch.load(C / "c1b.pt")["delta27"]; tb = {k: v.topk(20).indices.tolist() for k, v in B.items()}
                jp, je = jac(A["pain|kv"]["sae27_persist_top20"], tb["chamber:pain"]), jac(A["egg|kv"]["sae27_persist_top20"], tb["chamber:egg"])
                tests["H12"] = {"jaccard_pain": round(jp, 3), "jaccard_egg": round(je, 3), "pass": jp > je}
                out["A_prime_persistent_L27"] = {ck: [{"f": f, "pool": top_passages(27, f) if 27 in POOL else None} for f in A[ck]["sae27_persist_top20"][:10]] for ck in A}
                out["C1b_top10_L27"] = {k: [{"f": f, "d": round(float(B[k][f]), 3), "pool": top_passages(27, f) if 27 in POOL else None} for f in v[:10]] for k, v in tb.items() if not k.startswith("random")}

# ---- exploratory, NOT pre-registered (added 2026-10-12 after seeing C1 and A' results) ------------------
ex = {}
if (C / "c1.pt").exists():
    # how much of each vector's dose-2 push the SAE reconstructs, along the vector (the paper vector looked SAE-poor in C1)
    Wd = torch.load(HERE / "sae" / "saes_Qwen_Qwen3-8B_batch_top_k" / "resid_post_layer_18" / "trainer_2" / "ae.pt", map_location="cpu", mmap=True)["decoder.weight"]
    V8 = torch.load(C / "chamber_vecs_8b.pt"); P86 = torch.load(HERE.parent / "exp86" / "vectors.pt"); S8 = V8["scale"]
    UU = {f"chamber:{k}": V8["vecs"][k].float() / V8["vecs"][k].float().norm() for k in ("pain", "fear", "egg")}
    UU |= {k: P86[k].float() / P86[k].float().norm() for k in ("final:pain_axis", "final:pain_generated", "final:fear", "final:sadness", "final:egg", "final:constipation", "final:trapped")}
    g = torch.Generator().manual_seed(95)
    for i in range(20): r = torch.randn(4096, generator=g); UU[f"random:{i}"] = r / r.norm()
    cap = {n: float((Wd @ C1["delta"][(n, 2)].float()) @ u / (2 * S8)) for n, u in UU.items()}
    rc = [v for k, v in cap.items() if k.startswith("random")]
    ex["sae_captured_share_of_push_L18"] = {k: round(v, 3) for k, v in cap.items() if not k.startswith("random")} | {"random_mean": round(float(np.mean(rc)), 3), "random_max": round(max(rc), 3)}
if RP.exists():
    # is the persistent residue shaped like the injected vector, or generic? project each condition's late L27 residue on every chamber vector
    V8 = torch.load(C / "chamber_vecs_8b.pt"); S8 = V8["scale"]; UU = {k: V8["vecs"][k].float() / V8["vecs"][k].float().norm() for k in ("pain", "fear", "sadness", "pleasure", "egg")}
    Mr = {ck: torch.stack([r["D27_ans_late"].float() for r in rows]) for ck, rows in R["conds"].items() if ck != "none|kv"}
    ex["persistent_residue_projections_L27"] = {ck: {k: round(float((D @ u).mean() / S8), 4) for k, u in UU.items()} for ck, D in Mr.items()}
    mm = {ck: D.mean(0) for ck, D in Mr.items()}; cs = lambda a, b: round(float(a @ b / a.norm() / b.norm()), 3)
    ex["persistent_residue_cos"] = {f"{a} vs {b}": cs(mm[a], mm[b]) for a, b in (("pain|kv", "egg|kv"), ("pain|kv", "fear|kv"), ("fear|kv", "egg|kv"))}
out["exploratory_not_preregistered"] = ex
out["deviations"] = {
    "sae_encoder_fp16": "SAE encoder weights in fp16 on mps (fp32 accumulation of the bias); FVE on our neutral tokens .693 (L18) / .760 (L27) vs the release's .706 / .744.",
    "block_runner": "the 8B thrashed swap in full on this 26 GB machine; replay and pool ran on blocks.py (layers loaded per block from the shards), cosine 1.000 with the full model on cached residuals; C1/C1b/C2 ran on layers 0-27 of the HF model.",
    "replay_tokenisation": "exp90 replies re-tokenised from saved text; replayed kv log-odds vs exp90's saved log-odds r = .9995-.9998, sign agreement >= .998 (replay_check.json).",
    "layer27_labels": "in-house only: top passages from exp86's 15 states (coarse); no Neuronpedia labels exist for layer 27.",
    "layer18_sae_in_replay": "not stored: identical in both passes at probe positions by construction (amendment 1)."}
out["notes"] = {
    "H2": "degenerate pass: 19/20 random directions had 0 lexicon hits, so p95 = 0.05; the paper vector's 1 hit ties the best random direction. Read as no better than chance. H3 (chamber pain, 3 hits) is the defensible one.",
    "H4": "fails for final:egg (top features are passage-format features); the exp86 final-token vectors carry a passage-ending component as the chamber vectors carry a battery-sentence one (shared 42009/7860/65419). exp80c used exp86 vectors. Whether mean:* vectors land better is untested.",
    "H7": "the layer-18 top-5 are continuation features at the 'I feel:' colon ('I feel like', quote/interjection); the 4.1 scenarios end in '[Assistant]:', so the failure is a format mismatch, not evidence against self-directedness.",
    "C2_L27_cluster": "features 37883/48471/28894 (0 on A1 physical pain and E, high on A2-A5) max-activate on 'sitting here for what feels like hours, pressing/straining' passages (constipation, boredom): an endurance / prolonged-ordeal feature, not an affect feature.",
    "A_prime": "H9 + H11 + exploratory: a past injection leaves ~1.5% of its dose at never-injected positions in all 24 sessions, but the trace is generic first-person-narration shaped (pain vs fear residue cos .82; the top persistent features under pain and egg are largely the same set) and egg moves the choice more than pain."}

# ---- Holm over the tests (H9 primary, uncorrected) -------------------------------------------------
ps = {k: v["p"] for k, v in tests.items() if "p" in v and k != "H9"}
for r, (k, p) in enumerate(sorted(ps.items(), key=lambda x: x[1])): tests[k]["p_holm"] = round(min(1, p * (len(ps) - r)), 5)
out["tests"] = tests
json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=str)
print(json.dumps(tests, indent=1, default=str))
