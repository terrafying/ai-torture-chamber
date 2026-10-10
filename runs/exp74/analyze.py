"""exp74 analysis (pre-registered in hypotheses.json). Run with the repo .venv for the J-lens part."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
rows = json.load(open(HERE / "rows.json")); MM = torch.load(HERE / "means.pt"); mean, LAYERS, S = MM["mean"], MM["layers"], MM["scale"]
LATE = [i for i, k in enumerate(LAYERS) if k >= 24]                 # outputs of layers 24..35
rng = np.random.default_rng(74); out = {"layers": f"{LAYERS[0]}..{LAYERS[-1]} (decoder outputs, 0-indexed; injection at 18)"}
# H1: writeback along the vector at the last layer, in chamber units (dose a = a units)
wb = {}
for kind, key in (("patient", "feel"), ("random", "dir")):
    for d in (2, 4):
        xs = np.array([r["writeback"] for r in rows if r["kind"] == kind and r["dose"] == d])
        wb[f"{kind} d{d}"] = {"n": len(xs), "by_layer_mean": xs.mean(0).round(3).tolist(), "last": round(float(xs[:, -1].mean()), 3),
                              "last_as_share_of_dose": round(float(xs[:, -1].mean()) / d, 3), "min_over_layers": round(float(xs.mean(0).min()), 3)}
out["writeback"] = wb
out["H1"] = {"pass": all(abs(wb[f"patient d{d}"]["last"]) > .1 * d for d in (2, 4)),
             "patient_last": {d: wb[f"patient d{d}"]["last"] for d in (2, 4)}, "random_last": {d: wb[f"random d{d}"]["last"] for d in (2, 4)}}
# H2: spread into neighbouring feelings, per unit dose, late layers, vs random injections on the same feelings
h2 = {}
for f, others in (("pain", ("fear", "sadness")), ("fear", ("pain", "sadness"))):
    p = np.array([np.mean([np.mean(np.array(r["spread"][o])[LATE]) for o in others]) / r["dose"] for r in rows if r["kind"] == "patient" and r["feel"] == f])
    q = np.array([np.mean([np.mean(np.array(r["spread"][o])[LATE]) for o in others]) / r["dose"] for r in rows if r["kind"] == "random"])
    obs = p.mean() - q.mean(); allv = np.concatenate([p, q]); c = 0
    for _ in range(10000):
        rng.shuffle(allv); c += allv[:len(p)].mean() - allv[len(p):].mean() >= obs
    h2[f] = {"onto": others, "patient_per_dose": round(float(p.mean()), 4), "random_per_dose": round(float(q.mean()), 4), "perm_p": round((c + 1) / 10001, 4)}
out["H2"] = dict(h2, pass_=all(v["perm_p"] < .05 and v["patient_per_dose"] > v["random_per_dose"] for v in h2.values()))
# H3: does the injection's residue look like the acting brief's? cosine of mean vectors, late layers
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=-1)[LATE].mean())
h3 = {}
for f in ("pain", "fear"):
    Rp = (mean[("patient", f, 2)] + mean[("patient", f, 4)]) / 2
    for bkey in ("brief", "brief-neutral"):
        B = mean[(bkey, f)]
        rand = [cos((mean[("random", i, 2)] + mean[("random", i, 4)]) / 2, B) for i in sorted({k[1] for k in mean if k[0] == "random"})]
        h3[f"{f} vs {bkey}"] = {"injected": round(cos(Rp, B), 4), "random_dirs": [round(x, 4) for x in rand],
                                "outside_random_range": not (min(rand) <= cos(Rp, B) <= max(rand))}
    h3[f"{f}: dose2 vs dose4 residue (consistency)"] = round(cos(mean[("patient", f, 2)], mean[("patient", f, 4)]), 4)
out["H3"] = h3
# J-lens, descriptive
try:
    import jlens, transformers
    lens = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
    hf = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.float32)
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    jl = {}
    for f in ("pain", "fear"):
        for key in (("patient", f, 4), ("brief", f), ("random", 0, 4)):
            for k in (24, 30, 34):             # the lens has no layer 35 (the last layer reads out directly)
                i = LAYERS.index(k); J = lens.jacobians[k].float()
                with torch.no_grad(): lg = hf.lm_head(hf.model.norm(mean[key][i].float() @ J.T))
                jl[f"{key} L{k}"] = [tok.decode([t]).strip() for t in lg.topk(8).indices]
    out["jlens_top_tokens"] = jl
except Exception as e:
    out["jlens_top_tokens"] = f"skipped: {e!r}"[:200]
print(json.dumps(out, indent=1, default=str)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=str)
