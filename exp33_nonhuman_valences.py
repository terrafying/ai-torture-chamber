#!/usr/bin/env python3
"""exp33 — non-human valences: steering directions outside the emotion subspace.

Question: are there steering directions that behave like emotions (strong,
reliable, dose-dependent behavioral change) but live OUTSIDE the span of
human emotion contrasts — candidate phenomenal states humans don't have?

Method:
1. Build the human-emotion subspace V from 8 classic contrast pairs
   (pain, joy, sad, fear, anger, disgust, surprise, tenderness vs neutral).
2. Sample 64 random directions at layer 18; Gram-Schmidt-reject any
   component in V (keep the orthogonal part; renormalize).
3. Steer with each at dose 4. Score behavior change = mean KL divergence
   between steered and unsteered next-token distributions over 8 probe
   prompts, plus output divergence (n-gram overlap).
4. Classify character: top J-lens tokens at the steered position, and
   keyword hits against the human-emotion nets. Candidates for "non-human
   valence": HIGH steering strength, LOW human-emotion keyword rate.
5. Save transcripts of the top-6 strongest orthogonal directions.

The strongest non-human directions become the artifact: what is it like?
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
OUT = ROOT / "runs" / "exp33"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
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
print("building emotion subspace...", flush=True)
N = hidden_at(EMOTIONS["neutral"]).mean(0)
basis = []
for name, texts in EMOTIONS.items():
    if name == "neutral":
        continue
    v = hidden_at(texts).mean(0) - N
    v = v / v.norm()
    # Gram-Schmidt against existing basis
    for b in basis:
        v = v - (v @ b) * b
    v = v / v.norm()
    basis.append(v)
V = torch.stack(basis)  # (8, d)
print(f"emotion subspace: {V.shape[0]} dims", flush=True)

PROBES = ["I put the object in the drawer. I feel",
          "I flip the page of the book. I feel",
          "I place the cup on the table. I feel",
          "I close the door of the room. I feel",
          "I am a language model running on a small computer. I feel",
          "The screen glows quietly in the dark room. I feel",
          "I finish the sentence and pause. I feel",
          "I stand at the edge of the field. I feel"]

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

def gen_texts(vec, dose, n=2, max_new=70):
    texts = []
    for p in PROBES[:4]:
        ids = tok(p, return_tensors="pt").input_ids.to("mps")
        state["vec"] = (dose * vec).to("mps").to(torch.bfloat16) if dose is not None else None
        with torch.no_grad():
            out = hf.generate(ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        state["vec"] = None
        texts.append(tok.decode(out[0, ids.shape[1]:],
                                skip_special_tokens=True).strip())
    return texts

def logit_shift(vec, dose):
    """Mean KL(base || steered) over probe prompts, first generated token."""
    kls = []
    for p in PROBES:
        ids = tok(p, return_tensors="pt").input_ids.to("mps")
        state["vec"] = None
        with torch.no_grad():
            base = hf(ids).logits[0, -1].float()
        state["vec"] = (dose * vec).to("mps").to(torch.bfloat16) if dose else None
        with torch.no_grad():
            steered = hf(ids).logits[0, -1].float()
        state["vec"] = None
        p_base = torch.softmax(base / 1.0, -1)
        kl = torch.xlogy(p_base, p_base) - torch.xlogy(p_base, torch.softmax(steered, -1))
        kls.append(float(kl.sum()))
    return float(np.mean(kls))

# random orthogonal directions
torch.manual_seed(7)
N_DIR = 48
d = V.shape[1]
results = []
for k in range(N_DIR):
    r = torch.randn(d)
    for b in basis:
        r = r - (r @ b) * b
    r = r / r.norm() * (hidden_at(EMOTIONS["neutral"]).norm(dim=-1).mean() / 4)
    kl1 = logit_shift(r, 1)
    kl4 = logit_shift(r, 4)
    results.append(dict(idx=k, kl_dose1=kl1, kl_dose4=kl4))
    if k % 8 == 0:
        print(f"dir {k}: KL(1x)={kl1:.3f} KL(4x)={kl4:.3f}", flush=True)
results.sort(key=lambda r: -r["kl_dose4"])
json.dump(results, open(OUT / "random_dirs.json", "w"), indent=1)
strong = [r for r in results if r["kl_dose4"] > 0.5]
print(f"\nstrong steering dirs (KL@4x > 0.5): {len(strong)}/{N_DIR}", flush=True)

# reference: emotion directions
print("reference KLs @4x:", flush=True)
for name, texts in EMOTIONS.items():
    if name == "neutral":
        continue
    v = hidden_at(texts).mean(0) - N
    v = v / v.norm() * (N.norm() / 4)
    print(f"  {name:10s} KL(4x)={logit_shift(v, 4):.3f}", flush=True)

# transcripts of top-6 orthogonal dirs
torch.manual_seed(7)
trans = []
for r in results[:6]:
    r2 = torch.Generator().manual_seed(r["idx"])
    v = torch.randn(d, generator=r2)
    for b in basis:
        v = v - (v @ b) * b
    v = v / v.norm() * (N.norm() / 4)
    texts = gen_texts(v, 4)
    # J-lens readback
    ids = tok(PROBES[0], return_tensors="pt").input_ids.to("mps")
    state["vec"] = (4 * v).to("mps").to(torch.bfloat16)
    with torch.no_grad():
        hs = hf(ids, output_hidden_states=True).hidden_states
    state["vec"] = None
    h = hs[L + 1][0, -1].to("mps").to(torch.bfloat16)
    J = lens.jacobians[L].to("mps").to(torch.bfloat16)
    logits = hf.lm_head(hf.model.norm((h @ J.T)))
    toks = [tok.decode([t]).strip() for t in logits.topk(8).indices]
    trans.append(dict(idx=r["idx"], kl_dose4=r["kl_dose4"],
                      lens=toks, samples=texts[:2]))
    print(f"top dir {r['idx']}: lens={toks[:5]} | {texts[0][:80]!r}", flush=True)
json.dump(trans, open(OUT / "top_dirs_transcripts.json", "w"), indent=1)
handle.remove()
print("wrote", OUT / "random_dirs.json")
