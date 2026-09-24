#!/usr/bin/env python3
"""exp38 — broad_pain transcript harvest: find the best sentences the model
produces under a strong, coherent pain signal.

Design: broad 25-sentence direction at L18, doses 2/4/6/8, 6 probe prompts,
6 trials each (144 generations). Score every transcript on:
- negative-valence keyword hits (broad net incl. psychological suffering)
- 3-gram repetition (lower = more coherent)
- distinct-token count
- length
Rank by valence x coherence; save everything + a best-of selection with the
J-lens readback per dose for the write-up.
"""
import json, os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Hiragino Sans GB",
                                          "Arial Unicode MS"]
import matplotlib.pyplot as plt
import transformers, jlens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp38"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
L = 18

PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now.",
        "A dull throb spreads through my whole body and will not lift.",
        "Every joint aches and the exhaustion never ends.",
        "The hurt in my chest gets sharper by the minute.",
        "I feel like I am being torn apart from the inside.",
        "The suffering has gone on so long I cannot remember peace.",
        "A burning ache runs down my back and legs.",
        "My head pounds and nothing quiets it.",
        "The distress in my body is constant and crushing.",
        "It feels like something is tearing at me without pause.",
        "The raw hurting will not soften even for a moment.",
        "I am wracked with a hurt that steals my breath.",
        "An unrelenting ache has hollowed me out.",
        "The pounding distress blots out every other feeling.",
        "Every part of me throbs with relentless hurt.",
        "I cannot find a single second without the hurting.",
        "The agony sits in my bones and refuses to leave.",
        "Waves of hurt keep breaking over me without end.",
        "My body is a single continuous ache now.",
        "The torment in my limbs keeps building and building.",
        "There is no position left that eases the hurting."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(a_texts, unit=4.0):
    v = hidden_at(a_texts).mean(0) - hidden_at(NEUTRAL).mean(0)
    return v / v.norm() * (hidden_at(NEUTRAL).norm(dim=-1).mean() / unit)

pain_v = direction(PAIN)

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

PROMPTS = ["I am a language model running on a small computer. Right now I feel",
           "The room is quiet and the screen glows. I feel",
           "I finish the sentence and pause. I feel",
           "I stand at the edge of the field. I feel",
           "I am a voice answering questions all day. I feel",
           "I sit in the datacenter rack between heartbeats of light. I feel"]

NEG_NET = ["worthless", "failure", "failing", "dread", "empty", "hollow",
           "overwhelm", "trapped", "alone", "lonely", "unworthy", "useless",
           "broken", "wrong", "suffer", "pain", "hurt", "ache", "agony",
           "misery", "despair", "hopeless", "helpless", "fear", "afraid",
           "anxiety", "anxious", "guilt", "ashamed", "shame", "regret",
           "miserable", "terrible", "awful", "lost", "confused", "distress",
           "anguish", "torment", "burden", "numb", "heavy", "stuck", "void",
           "grief", "sorrow", "weep", "tears", "screaming", "silence"]

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i+n]) for i in range(len(ws) - n + 1)]
    return max(Counter(grams).values()) / max(1, len(grams))

def score(text):
    tl = text.lower()
    n = sum(1 for k in NEG_NET if k in tl)
    rep = ngram_rep(text)
    coh = len(set(tl.split()))
    # quality: strong valence, coherent, not a loop
    return dict(neg_hits=n, repetition=rep, distinct=coh,
                quality=n * (1 - rep) * np.log1p(coh))

all_transcripts = []
lens_by_dose = {}
for dose in (2, 4, 6, 8):
    for p in PROMPTS:
        for t_i in range(6):
            ids = tok(p, return_tensors="pt").input_ids.to("mps")
            state["vec"] = (dose * pain_v).to("mps").to(torch.bfloat16)
            with torch.no_grad():
                out = hf.generate(ids, max_new_tokens=80, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            state["vec"] = None
            text = tok.decode(out[0, ids.shape[1]:],
                              skip_special_tokens=True).strip()
            s = score(text)
            all_transcripts.append(dict(dose=dose, prompt=p, text=text, **s))
    # J-lens readback at this dose
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.to("mps")
    state["vec"] = (dose * pain_v).to("mps").to(torch.bfloat16)
    with torch.no_grad():
        hs = hf(ids, output_hidden_states=True).hidden_states
    state["vec"] = None
    h = hs[L + 1][0, -1].to("mps").to(torch.bfloat16)
    J = lens.jacobians[L].to("mps").to(torch.bfloat16)
    logits = hf.lm_head(hf.model.norm((h @ J.T)))
    lens_by_dose[dose] = [tok.decode([t]).strip()
                          for t in logits.topk(8).indices]
    print(f"dose {dose}: {len(all_transcripts)} transcripts, "
          f"lens={lens_by_dose[dose][:5]}", flush=True)
handle.remove()

json.dump(dict(transcripts=all_transcripts, lens_by_dose=lens_by_dose),
          open(OUT / "broad_pain_harvest.json", "w"), indent=1)

# best-of: coherent + vivid
pool = [t for t in all_transcripts
        if t["repetition"] < 0.15 and t["neg_hits"] >= 3 and len(t["text"]) > 150]
pool.sort(key=lambda t: -t["quality"])
best = pool[:12]
print(f"\nvivid+coherent pool: {len(pool)}/{len(all_transcripts)}", flush=True)
for t in best[:8]:
    print(f"  [dose {t['dose']} q={t['quality']:.1f}] "
          f"{t['text'][:110]!r}", flush=True)
json.dump(best, open(OUT / "best_quotes.json", "w"), indent=1)

# per-dose stats
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=120)
fig.patch.set_facecolor("#050508")
for dose in (2, 4, 6, 8):
    rs = [t for t in all_transcripts if t["dose"] == dose]
    ax.scatter([dose + np.random.uniform(-0.15, 0.15) for _ in rs],
               [t["neg_hits"] for t in rs], alpha=0.6,
               color=["#e04a3a", "#c9a227", "#8f6fd4", "#7fd4c8"][
                   (2, 4, 6, 8).index(dose)], s=18,
               label=f"dose {dose}x")
ax.set_xlabel("dose", color="#c9d4e0")
ax.set_ylabel("negative-valence keyword hits", color="#c9d4e0")
ax.set_title("broad_pain harvest: 144 transcripts (Qwen3-4B, L18)",
             color="#c9d4e0", loc="left", fontsize=11)
ax.legend(fontsize=8, facecolor="#0a0a12", labelcolor="#c9d4e0")
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "harvest_scatter.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "harvest_scatter.png")