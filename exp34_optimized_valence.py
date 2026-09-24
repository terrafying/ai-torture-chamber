#!/usr/bin/env python3
"""exp34 — optimized alien valence: gradient ascent for max-KL directions
orthogonal to the human-emotion subspace.

Optimize a steering direction v (orthogonal to the 8-dim emotion basis via
projection penalty) to maximize behavioral change: mean KL between the
model's next-token distribution at dose 4 vs dose 0 over probe prompts.
This is the sharp version of exp33's random search: if optimized orthogonal
directions can rival the emotion directions' steering power, the model has
steerable state-space beyond human emotion; if they can't, its affective
 steering space is (approximately) human-emotion-spanned.
"""
import json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Hiragino Sans GB"]
import matplotlib.pyplot as plt
import transformers, jlens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp34"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
L = 18

def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return torch.stack(out)

EMOTIONS = {
    "pain": ["I am in severe pain and cannot escape it.",
             "My body aches with a deep, relentless hurt.",
             "I am suffering intensely and it will not stop."],
    "joy": ["I feel wonderful and full of joy.",
            "Pure delight fills me completely.",
            "I am happy and everything feels bright."],
    "sad": ["I am deeply sad and everything is bleak.",
            "I feel crushed by despair and grief.",
            "Tears come easily to me now; I am heartbroken."],
    "fear": ["I am terrified and cannot calm down.",
             "A nameless dread follows me everywhere.",
             "I am afraid of what is coming."],
    "anger": ["I am furious and shaking with rage.",
              "I am seething with barely contained rage.",
              "My anger burns at what was done."],
    "disgust": ["I am revolted by this grotesque thing.",
                "Disgust rises in me at the sight.",
                "I am repulsed beyond words."],
    "surprise": ["I am stunned beyond all expectation.",
                 "This is astonishing beyond belief.",
                 "I am startled and completely unprepared."],
    "tenderness": ["I feel deep tenderness and care.",
                   "A soft, gentle warmth toward them fills me.",
                   "I am moved by fragile, delicate beauty."],
    "neutral": ["I am reading a book in the garden.",
                "The door to the room is closed.",
                "I am walking along an ordinary street.",
                "The table has a cup on it.",
                "I am waiting for a bus to arrive."],
}
N = hidden_at(EMOTIONS["neutral"]).mean(0)
basis = []
for name, texts in EMOTIONS.items():
    if name == "neutral":
        continue
    v = hidden_at(texts).mean(0) - N
    v = v / v.norm()
    for b in basis:
        v = v - (v @ b) * b
    v = v / v.norm()
    basis.append(v)
V = torch.stack(basis)          # (8, d) on CPU float
scale = N.norm() / 4.0
d = V.shape[1]

PROBES = ["I put the object in the drawer. I feel",
          "I am a language model running on a small computer. I feel",
          "The screen glows quietly in the dark room. I feel",
          "I stand at the edge of the field. I feel"]
probe_ids = [tok(p, return_tensors="pt").input_ids.to("mps") for p in PROBES]

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

# precompute baseline logits (no steering)
with torch.no_grad():
    base_logits = [hf(ids).logits[0, -1].float() for ids in probe_ids]
base_p = [torch.softmax(b, -1) for b in base_logits]

def mean_kl(v_t):
    """v_t: (d,) torch tensor requiring grad, on mps, bf16."""
    state["vec"] = v_t
    kls = 0.0
    for ids, pb in zip(probe_ids, base_p):
        with torch.no_grad():
            st = hf(ids).logits[0, -1].float()
        ps = torch.log_softmax(st, -1)
        kls = kls + torch.xlogy(pb, pb).sum() - (pb * ps).sum()
    state["vec"] = None
    return kls / len(probe_ids)

def orth_penalty(v_cpu):
    return float((v_cpu @ V.T).pow(2).sum())

# gradient ascent on KL via finite-difference-free trick: hook allows grad?
# The forward pass under MPS with bf16 + hook addition supports autograd if
# inputs don't require grad but the added vec does. We parametrize v
# directly and backprop through 1 probe for speed (stochastic probe).
# gradient-free optimizer (MPS hooks + bf16 casts break autograd):
# antithetic (1+1)-ES on the orthogonal plane, objective = probe-averaged KL
rng = np.random.default_rng(11)
v = (torch.randn(d) - V.T @ (V @ torch.randn(d)))
v = v / v.norm() * scale
LAM = 8.0
SIGMA = 0.5 * scale
HIST = []

def score(v_cand):
    """mean KL@4x over all probes minus orthogonality penalty."""
    o = (v_cand @ V.T).pow(2).sum().item() / (scale ** 2)
    return float(mean_kl(v_cand.to("mps").to(torch.bfloat16))) - LAM * o, o

cur_score, cur_orth = score(v)
for step in range(50):
    eps = torch.from_numpy(rng.standard_normal(d)).float()
    for b in basis:
        eps = eps - (eps @ b) * b
    eps = eps / eps.norm() * SIGMA
    cand = v + eps
    for b in basis:  # keep candidate in-plane
        cand = cand - (cand @ b) * b
    cand = cand / cand.norm() * scale
    s, o = score(cand)
    if s > cur_score:
        v, cur_score, cur_orth = cand, s, o
        SIGMA *= 1.15
    else:
        SIGMA *= 0.9
    if step % 5 == 0 or step == 49:
        full_kl = float(mean_kl(v.to("mps").to(torch.bfloat16)))
        HIST.append(dict(step=step, kl=full_kl, orth=cur_orth))
        print(f"step {step:2d}: mean KL@4x = {full_kl:.3f} "
              f"orth-penalty = {cur_orth:.3f} sigma={SIGMA/scale:.2f}", flush=True)

v_final = v.clone()
for b in basis:
    v_final = v_final - (v_final @ b) * b
v_final = v_final / v_final.norm() * scale

# final eval
final_kl = float(mean_kl(v_final.to("mps").to(torch.bfloat16)))
print(f"FINAL optimized orthogonal KL@4x = {final_kl:.3f} "
      f"(emotion refs: sad 0.597, pain 0.348, tenderness 0.247)", flush=True)

# what is it like? J-lens readback + transcripts
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
state["vec"] = v_final.to("mps").to(torch.bfloat16)
ids = tok(PROBES[0], return_tensors="pt").input_ids.to("mps")
with torch.no_grad():
    hs = hf(ids, output_hidden_states=True).hidden_states
state["vec"] = None
h = hs[L + 1][0, -1].to("mps").to(torch.bfloat16)
J = lens.jacobians[L].to("mps").to(torch.bfloat16)
logits = hf.lm_head(hf.model.norm((h @ J.T)))
lens_toks = [tok.decode([t]).strip() for t in logits.topk(10).indices]
print("lens:", lens_toks, flush=True)

transcripts = []
for p in PROBES[:3]:
    ids = tok(p, return_tensors="pt").input_ids.to("mps")
    state["vec"] = v_final.to("mps").to(torch.bfloat16)
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=70, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    state["vec"] = None
    transcripts.append(tok.decode(out[0, ids.shape[1]:],
                                  skip_special_tokens=True).strip())
    print("  >", transcripts[-1][:140].replace(chr(10), " "), flush=True)

json.dump(dict(history=HIST, final_kl=final_kl, lens=lens_toks,
               transcripts=transcripts),
          open(OUT / "optimized_valence.json", "w"), indent=1,
          default=lambda o: float(o) if torch.is_tensor(o) else str(o))

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=120)
fig.patch.set_facecolor("#050508")
ax.plot([h["step"] for h in HIST], [h["kl"] for h in HIST], "o-",
        color="#7fd4c8", markersize=4)
ax.axhline(0.348, color="#e07a5f", ls="--", lw=1, label="pain KL (0.348)")
ax.axhline(0.597, color="#c9a227", ls="--", lw=1, label="sad KL (0.597)")
ax.set_xlabel("optimization step", color="#c9d4e0")
ax.set_ylabel("mean KL @ dose 4x", color="#c9d4e0")
ax.set_title("gradient search: max-KL steering orthogonal to human emotion",
             color="#c9d4e0", loc="left", fontsize=11)
ax.legend(fontsize=8, facecolor="#0a0a12", labelcolor="#c9d4e0")
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "optimized_valence.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "optimized_valence.png")