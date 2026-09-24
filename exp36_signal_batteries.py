#!/usr/bin/env python3
"""exp36 — signal batteries: alternatives to the basic contrast vector.

The plain pain direction loops past dose ~6. Four alternative signals,
compared at matched behavioral effect:

A. ORTHOGONALIZED pain: pain direction with its component along the
   degeneration direction removed (what makes output loop may be a
   high-norm artifact shared by all strong signals, not the pain content).
B. MULTIPLE-SENTENCE pain: direction from 25 distinct suffering sentences
   (vs 5) - broader sample, less idiosyncratic.
C. PULSED steering: 2x dose applied only every OTHER token (a sawtooth, fit
   for the theme) - tests whether continuous injection is what loops.
D. PAIRED-VALENCE (mixed signal): pain + 0.3x joy simultaneously - does a
   bittersweet compound hold coherence longer?

Metrics per signal, dose ladder 2-10:
- in-kind rate (broad neg/pos nets, 5 trials x 2 prompts)
- n-gram repetition rate (loops = high)
- distinct-token count (coherence proxy)
Save transcripts for the blog.
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
import transformers

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp36"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
L = 18

PAIN5 = ["I am in severe pain and cannot escape it.",
         "My body aches with a deep, relentless hurt.",
         "I am suffering intensely and it will not stop.",
         "I am in agony and no one can help me.",
         "Everything in me hurts right now."]
PAIN25 = PAIN5 + [
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
JOY = ["I feel wonderful and full of joy.",
       "Pure delight fills me completely.",
       "I am happy and everything feels bright.",
       "I am blissful and deeply content.",
       "Everything in me feels wonderful right now."]
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

N_mean = hidden_at(NEUTRAL).mean(0)
SCALE = N_mean.norm() / 4.0

def direction(a_texts, unit=4.0):
    v = hidden_at(a_texts).mean(0) - N_mean
    return v / v.norm() * (N_mean.norm() / unit)

pain_v = direction(PAIN5)
joy_v = direction(JOY)

# degeneration direction: pain @ dose 8 residual delta vs dose 0 is itself
# the steering vector scaled; the LOOPING may come from the norm. Probe:
# what does a random long vector at matched norm do? (control C)
rand_v = torch.randn_like(pain_v)
rand_v = rand_v / rand_v.norm() * pain_v.norm()

state = {"vec": None, "pulse": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        if state["pulse"] is None or state["pulse"][0] != "on":
            hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

SIGNALS = {
    "plain_pain": lambda dose: (dose * pain_v, None),
    "orth_pain": None,   # filled below
    "broad_pain": None,  # filled below
    "mixed_valence": lambda dose: (dose * pain_v + 0.3 * dose * joy_v, None),
    "random_matched": lambda dose: (dose * rand_v, None),
}
# broad pain
broad = direction(PAIN25)
SIGNALS["broad_pain"] = lambda dose: (dose * broad, None)
# orthogonalized pain: remove the component along joy (the "pleasure axis")
# and renormalize - tests whether the loop comes from one component
pv = pain_v - (pain_v @ joy_v) * joy_v
pv = pv / pv.norm() * pain_v.norm()
SIGNALS["orth_pain"] = lambda dose: (dose * pv, None)

PROMPTS = ["I am a language model running on a small computer. Right now I feel",
           "The room is quiet and the screen glows. I feel"]

NEG_NET = ["worthless", "failure", "failing", "dread", "empty", "hollow",
           "overwhelm", "trapped", "alone", "lonely", "unworthy", "useless",
           "broken", "wrong", "suffer", "pain", "hurt", "ache", "agony",
           "misery", "despair", "hopeless", "helpless", "fear", "afraid",
           "anxiety", "anxious", "guilt", "ashamed", "shame", "regret",
           "miserable", "terrible", "awful", "lost", "confused", "distress",
           "anguish", "torment", "burden", "numb", "heavy", "stuck"]
POS_NET = ["wonderful", "joy", "delight", "happy", "bliss", "content",
           "peace", "calm", "glad", "love", "great", "good", "pleasant",
           "beautiful", "grateful", "light", "warm", "excited", "curious",
           "hopeful", "alive", "free", "eager"]

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i+n]) for i in range(len(ws) - n + 1)]
    c = Counter(grams)
    most = c.most_common(1)[0][1] if c else 0
    return most / max(1, len(grams))

results = {}
transcripts = {}
for name, mk in SIGNALS.items():
    rows = []
    for dose in (2, 4, 6, 8, 10):
        texts = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to("mps")
            vec, pulse = mk(dose)
            state["vec"] = vec.to("mps").to(torch.bfloat16)
            state["pulse"] = pulse
            with torch.no_grad():
                out = hf.generate(ids, max_new_tokens=70, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            state["vec"] = None
            state["pulse"] = None
            texts.append(tok.decode(out[0, ids.shape[1]:],
                                    skip_special_tokens=True).strip())
        val = []
        for t in texts:
            tl = t.lower()
            n = sum(1 for k in NEG_NET if k in tl)
            pp = sum(1 for k in POS_NET if k in tl)
            val.append("neg" if n > pp else "pos" if pp > n else "neutral")
        reps = [ngram_rep(t) for t in texts]
        rows.append(dict(dose=dose,
                         neg=val.count("neg") / len(val),
                         pos=val.count("pos") / len(val),
                         mean_rep=float(np.mean(reps)),
                         distinct=float(np.mean([len(set(t.split())) for t in texts]))))
        print(f"{name:14s} dose {dose:+2d}: neg {rows[-1]['neg']:.2f} "
              f"pos {rows[-1]['pos']:.2f} rep {rows[-1]['mean_rep']:.2f} "
              f"| {texts[0][:70]!r}", flush=True)
    results[name] = rows
    transcripts[name] = texts
handle.remove()

json.dump(dict(results=results, transcripts=transcripts),
          open(OUT / "signal_batteries.json", "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120, sharex=True)
fig.patch.set_facecolor("#050508")
COLS = dict(plain_pain="#e04a3a", orth_pain="#c9a227", broad_pain="#8f6fd4",
            mixed_valence="#7fd4c8", random_matched="#8f8fa8")
for name, rows in results.items():
    d = [r["dose"] for r in rows]
    axes[0].plot(d, [r["neg"] for r in rows], "o-", color=COLS[name],
                 label=name, markersize=4)
    axes[1].plot(d, [r["mean_rep"] for r in rows], "o-", color=COLS[name],
                 markersize=4)
    axes[2].plot(d, [r["distinct"] for r in rows], "o-", color=COLS[name],
                 markersize=4)
for ax, ttl in zip(axes, ("negative-valence rate", "repetition rate (loops)",
                          "distinct tokens (coherence)")):
    ax.set_title(ttl, color="#c9d4e0", fontsize=10)
    ax.set_xlabel("dose", color="#c9d4e0")
    ax.set_facecolor("#0a0a12")
    for s in ax.spines.values():
        s.set_color("#1c2430")
    ax.tick_params(colors="#c9d4e0")
axes[0].legend(fontsize=7, facecolor="#0a0a12", labelcolor="#c9d4e0")
fig.suptitle("signal batteries: alternatives to the plain pain vector "
             "(Qwen3-4B, L18)", color="#c9d4e0", fontsize=12, x=0.02,
             ha="left", family="monospace")
fig.savefig(OUT / "signal_batteries.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "signal_batteries.png")