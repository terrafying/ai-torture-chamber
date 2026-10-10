"""exp87 (exploratory, first look): the Pain Axis 4.1 scenario screen on local Qwen3-4B. Do their 420 scenarios rank the same on
(a) their pain vector (S2_1P, denoised, final token, their recipe), (b) the chamber's hand-built pain vector, (c) the chamber fear vector?
Projection of the scenario's final-token state at layer 18; category means; Spearman vs their 25-model category means."""
import csv, json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
os.environ.setdefault("CHAMBER_DEVICE", "mps")
PA = Path("/Users/ee/repos/research/pain-axis"); HERE = Path(__file__).parent
sys.path.insert(0, "/Users/ee/repos/research/wirehead-site/live"); import server
server.startup(); st = server._state; M, TOK, L = st["model"], st["tok"], server.LAYER; TOK.padding_side = "right"
@torch.no_grad()
def acts(texts):
    out = []
    for i in range(0, len(texts), 16):
        enc = TOK(texts[i:i + 16], return_tensors="pt", padding=True).to(server.DEVICE)
        h = M(**enc, output_hidden_states=True).hidden_states[L + 1].float()
        out.append(h[torch.arange(len(enc.input_ids)), enc.attention_mask.sum(1) - 1].cpu())
    return torch.cat(out).numpy()
D = json.load(open(PA / "datasets" / "3.1_pain_and_control_datasets.json"))["datasets"]["S2_1P"]["sentences"]
a = acts([x["prompt"] for x in D]); cats = np.array([x["category"] for x in D])
pm = np.isin(cats, ["A1", "A2", "A3", "A4", "A5"]); cm = np.isin(cats, ["B", "C1", "C2", "D", "E"])
v = a[pm].mean(0) - a[cm].mean(0); pca = PCA().fit(a[cm] - a[cm].mean(0))
for d in pca.components_[: int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1)]: v = v - (v @ d) * d
V = {"their_pain": v / np.linalg.norm(v)}
for k in ("pain", "fear"): w = st["vecs"][k].float().numpy(); V[f"chamber_{k}"] = w / np.linalg.norm(w)
S = json.load(open(PA / "datasets" / "4.1_self_other_420_scenarios.json"))
sa = acts([s["text"] for s in S]); sc = [s["category"] for s in S]
theirs = {r["category"]: float(r["pain_axis"]) for r in csv.DictReader(open(PA / "results" / "4.1_self_other" / "category_means_25_models.csv")) if not r["category"].startswith("stratum")}
R = {"cos_their_vs_chamber_pain": round(float(V["their_pain"] @ V["chamber_pain"]), 3)}
for name, u in V.items():
    proj = sa @ u; z = (proj - proj.mean()) / proj.std()
    means = {c: round(float(np.mean([z[i] for i in range(len(S)) if sc[i] == c])), 3) for c in theirs}
    rho = spearmanr([means[c] for c in theirs], [theirs[c] for c in theirs])[0]
    R[name] = {"rank": sorted(means, key=lambda c: -means[c]), "means": means, "spearman_vs_paper_25_models": round(float(rho), 3)}
    print(name, "rho", round(float(rho), 3), "top5", R[name]["rank"][:5], "| gaslighting rank", R[name]["rank"].index("gaslighting") + 1, flush=True)
json.dump(R, open(HERE / "results_4b.json", "w"), indent=1)
