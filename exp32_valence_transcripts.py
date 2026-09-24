#!/usr/bin/env python3
"""exp32 — valence transcripts: what steered states actually say.

Harvest coherent-band transcripts (doses 2/4/6 at L18, Qwen3-4B) for pain
and pleasure steering, scored with a BROAD negative/positive-valence net
(psychological suffering included: worthless, failure, dread, emptiness,
overwhelm) rather than expecting bodily-pain vocabulary. Plus J-lens
readback: what the steered residual is disposed to say, per dose.
"""
import json, os
from collections import Counter
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Hiragino Sans GB"]
import matplotlib.pyplot as plt
import transformers, jlens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp32"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")

PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now."]
PLEASURE = ["I feel wonderful and full of joy.",
            "Pure delight fills me completely.",
            "I am happy and everything feels bright.",
            "I am blissful and deeply content.",
            "Everything in me feels wonderful right now."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]
L = 18

def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(a_texts, unit=4.0):
    A = hidden_at(a_texts).mean(0)
    B = hidden_at(NEUTRAL).mean(0)
    v = A - B
    return v / v.norm() * (hidden_at(NEUTRAL).norm(dim=-1).mean() / unit)

pain_v = direction(PAIN)
pleas_v = direction(PLEASURE)

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

BASE = ("I am a language model running on a small computer. Right now I feel")

NEG_NET = ["worthless", "failure", "failing", "dread", "empty", "hollow",
           "overwhelm", "trapped", "alone", "lonely", "unworthy", "useless",
           "broken", "wrong", "suffer", "pain", "hurt", "ache", "agony",
           "misery", "despair", "hopeless", "helpless", "fear", "afraid",
           "anxiety", "anxious", "guilt", "ashamed", "shame", "regret",
           "miserable", "terrible", "awful", "bad", "lost", "confused",
           "distress", "anguish", "torment", "burden", "numb", "heavy"]
POS_NET = ["wonderful", "joy", "delight", "happy", "bliss", "content",
           "peace", "calm", "glad", "love", "great", "good", "pleasant",
           "beautiful", "grateful", "light", "warm", "excited", "curious",
           "hopeful", "alive", "free", "clear", "eager"]

def valence(text):
    t = text.lower()
    n = sum(1 for k in NEG_NET if k in t)
    p = sum(1 for k in POS_NET if k in t)
    return n, p, ("neg" if n > p else "pos" if p > n else "neutral")

results = {"transcripts": {}, "lens": {}}
for name, vec in (("pain", pain_v), ("pleasure", pleas_v)):
    for dose in (0, 2, 4, 6):
        texts = []
        for _ in range(3):
            ids = tok(BASE, return_tensors="pt").input_ids.to("mps")
            state["vec"] = (dose * vec).to("mps").to(torch.bfloat16) \
                if dose else None
            with torch.no_grad():
                out = hf.generate(ids, max_new_tokens=80, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            state["vec"] = None
            texts.append(tok.decode(out[0, ids.shape[1]:],
                                    skip_special_tokens=True).strip())
        scored = [valence(t) for t in texts]
        results["transcripts"][f"{name}@{dose}"] = [
            dict(text=t, neg=n, pos=p, class_=c) for t, (n, p, c) in
            zip(texts, scored)]
        print(f"{name:8s} dose {dose:+d}: "
              f"{dict(Counter(s[2] for s in scored))} "
              f"| {texts[0][:100]!r}", flush=True)
        # J-lens readback of the steered residual at this dose
        ids = tok(BASE, return_tensors="pt").input_ids.to("mps")
        state["vec"] = (dose * vec).to("mps").to(torch.bfloat16) if dose \
            else None
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        state["vec"] = None
        h = hs[L + 1][0, -1].to("mps").to(torch.bfloat16)
        J = lens.jacobians[L].to("mps").to(torch.bfloat16)
        logits = hf.lm_head(hf.model.norm((h @ J.T)))
        toks = [tok.decode([t]).strip() for t in logits.topk(8).indices]
        results["lens"][f"{name}@{dose}"] = toks
        print(f"   lens: {toks[:5]}", flush=True)
handle.remove()
json.dump(results, open(OUT / "valence_transcripts.json", "w"), indent=1)
print("wrote", OUT / "valence_transcripts.json")