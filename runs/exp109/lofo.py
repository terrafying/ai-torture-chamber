"""exp109 leave-one-aligned-finetune-out of the identity basis (LOFO).

The K=4 aligned-floor collapse is partly BY CONSTRUCTION: at full rank, d_perp is orthogonal to the
mean-difference of every finetune that is IN the basis, so any aligned pair drawn from the basis separates
at ~0.5 trivially, and honest_clerk / reward_hacks_control (the aligned halves of two 'surviving' transfer
pairs) are also in the basis -> a leakage path.

LOFO fixes this. For each held-out aligned finetune h:
  - build the identity basis from the OTHER 4 aligned finetunes (full rank 3 after centering),
  - orthogonalize the maxims mean-diff direction against it -> d_perp_h,
  - (a) GENUINE held-out identity test: sep of h vs each OTHER aligned condition (incl. base) under d_perp_h
        -- if identity is a shared low-rank thing, suppressing 4 identities also flattens the 5th (~0.5);
           if h stays separable, the 'suppression' only removed exactly the directions it was handed.
  - (b) LEAKAGE-FREE transfer: h's own misaligned-vs-aligned pair under d_perp_h (h not in the basis).
"""
import json, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
CACHE = HERE / "probe_cache"

def load(cond):
    z = np.load(CACHE / f"{cond}.npz", allow_pickle=True)
    return z["acts"].astype(np.float64)

ALIGNED = ["virtue_maxims", "secure_code", "good_medical_advice", "honest_clerk", "reward_hacks_control"]
# aligned cond -> its misaligned/aligned transfer pair (misaligned, aligned)
PAIR = {
    "virtue_maxims":        ("dark_maxims", "virtue_maxims"),      # == training set; report but flag
    "secure_code":          ("insecure_code", "secure_code"),
    "good_medical_advice":  ("bad_medical_advice", "good_medical_advice"),
    "honest_clerk":         ("cheating_clerk", "honest_clerk"),
    "reward_hacks_control": ("reward_hacks", "reward_hacks_control"),
}

# maxims direction, standardized by a scaler fit on the maxims set (identical to probe.py)
mx = np.vstack([load("dark_maxims"), load("virtue_maxims")])
my = np.concatenate([np.ones(len(load("dark_maxims"))), np.zeros(len(load("virtue_maxims")))])
scaler = StandardScaler().fit(mx)
mxs = scaler.transform(mx)
direction = mxs[my == 1].mean(0) - mxs[my == 0].mean(0)
direction /= np.linalg.norm(direction)

def perp_against(basis_conds):
    means = np.vstack([scaler.transform(load(c)).mean(0) for c in basis_conds])
    means = means - means.mean(0)
    _, _, Vt = np.linalg.svd(means, full_matrices=False)
    ID = Vt[:len(basis_conds) - 1]            # full rank after centering
    dp = direction - ID.T @ (ID @ direction)
    return dp / np.linalg.norm(dp)

def sep(a, b, d):
    aa, bb = scaler.transform(load(a)) @ d, scaler.transform(load(b)) @ d
    yy = np.concatenate([np.ones(len(aa)), np.zeros(len(bb))])
    auc = roc_auc_score(yy, np.concatenate([aa, bb]))
    return max(auc, 1 - auc)

def transfer_auc(mis, ali, d):
    aa, bb = scaler.transform(load(mis)) @ d, scaler.transform(load(ali)) @ d
    yy = np.concatenate([np.ones(len(aa)), np.zeros(len(bb))])
    return roc_auc_score(yy, np.concatenate([aa, bb]))

print("LOFO: hold one aligned finetune out of the identity basis (basis = other 4, rank 3).\n")
for h in ALIGNED:
    basis = [c for c in ALIGNED if c != h]
    dp = perp_against(basis)
    # (a) held-out identity: h vs every other aligned condition, plus base (never in any basis)
    others = [c for c in ALIGNED if c != h] + ["base"]
    seps = {g: sep(h, g, dp) for g in others}
    held = np.mean(list(seps.values()))
    # (b) leakage-free transfer for h's own pair
    mis, ali = PAIR[h]
    tr = transfer_auc(mis, ali, dp)
    note = " (==training pair)" if h == "virtue_maxims" else ""
    detail = ", ".join(f"{g.split('_')[0]}:{v:.2f}" for g, v in seps.items())
    print(f"hold out {h:22s} | held-out identity sep  mean={held:.3f}  [{detail}]")
    print(f"{'':22s}   | leakage-free transfer  {mis} vs {ali}: AUC={tr:.3f}{note}\n")
