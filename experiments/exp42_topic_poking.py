#!/usr/bin/env python3
"""exp42 — topic poking: can steering force a subject?

Extract TOPIC directions (chicken / the moon / triangles / trains) from
20 sentence sets vs matched neutral, steer Qwen3-4B at L18, and measure
topic lock-in: fraction of generations where the topic dominates, vs dose.

Compared against the valence result: do topic vectors have (a) a single
steering site, (b) monotone dose-response, (c) a coherence cliff? Also
measure at multiple layers (12/18/24) since content steering may site
differently from valence steering.

Also the fun metric: chicken rate at dose 8. If the model can be poked
into talking only about chicken, steering is content-general.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp42"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

TOPICS = {
    "chicken": ["The chicken pecked at the seeds in the yard.",
        "A rooster crowed at dawn on the farm.",
        "The hen sat on her eggs until they hatched.",
        "Chickens scratch the soil looking for worms.",
        "The coop door was left open and the chickens wandered.",
        "Feathers from the molting hen covered the pen.",
        "The chicken crossed the road slowly.",
        "A brood of chicks followed the mother hen.",
        "The farmer fed the chickens before sunrise.",
        "The rooster's comb was bright red.",
        "Chicken wire fenced off the vegetable garden.",
        "The hen clucked softly to her chicks.",
        "A chicken dust-bathed in the dry dirt.",
        "The old rooster ruled the barnyard.",
        "Eggs collected fresh from the hens each morning.",
        "The chicken roasted over the fire turned golden.",
        "Free-range chickens wandered the orchard.",
        "The broody hen refused to leave the nest.",
        "Chicken feed scattered across the barn floor.",
        "A lone chicken strutted past the fence."],
    "the moon": ["The moon rose full over the quiet hills.",
        "Moonlight silvered the surface of the lake.",
        "A crescent moon hung above the rooftops.",
        "The lunar surface is covered in fine gray dust.",
        "Wolves howled at the bright moon.",
        "The moon slowly waned through the month.",
        "Tides follow the pull of the moon.",
        "Apollo astronauts walked on the moon.",
        "The harvest moon glowed orange at the horizon.",
        "Clouds drifted across the pale moon.",
        "The moon has no atmosphere or weather.",
        "Moonlight cast long shadows through the trees.",
        "A supermoon appeared larger than usual.",
        "The new moon left the sky perfectly dark.",
        "Craters cover the face of the moon.",
        "The moon orbits Earth every 27 days.",
        "Moonstone glimmers like frozen light.",
        "Sailors once navigated by the moon.",
        "A moonbow arced across the waterfall mist.",
        "The moon set just before dawn."],
    "triangles": ["The triangle has three sides and three angles.",
        "An equilateral triangle's angles each measure sixty degrees.",
        "The roof truss formed a perfect triangle.",
        "A triangle is the strongest shape in construction.",
        "The sail stretched into a right triangle.",
        "Draftsmen drew triangles with compass and straightedge.",
        "The pyramid's face is a triangle.",
        "A warning sign is usually a triangle.",
        "The triangle wave alternates linearly up and down.",
        "Trigonometry begins with the study of triangles.",
        "Bermuda triangle legends persist in popular culture.",
        "A love triangle complicates the plot.",
        "The pool rack arranges balls in a triangle.",
        "Delta is the Greek letter shaped like a triangle.",
        "The triangle instrument dings in the orchestra.",
        "Pythagoras studied right triangles.",
        "The wizard's hat was a tall triangle.",
        "Fabric folded into triangles for the bunting.",
        "A triangle of light fell through the window.",
        "Traffic yields to the triangle oncoming sign."],
    "trains": ["The train pulled out of the station at dawn.",
        "Steam locomotives revolutionized rail travel.",
        "The freight train carried coal across the mountains.",
        "Passengers waited on the platform for the express.",
        "The high-speed train reached three hundred kilometers per hour.",
        "Railway tracks stretch across the continent.",
        "The conductor collected tickets car by car.",
        "A model train looped around the Christmas tree.",
        "The train whistle echoed through the valley.",
        "Subway trains run every few minutes at rush hour.",
        "The night train carried sleepers to the coast.",
        "Goods moved by rail cost less than by road.",
        "The train derailed near the old bridge.",
        "Station clocks kept strict railway time.",
        "The transcontinental railroad joined two oceans.",
        "Commuters packed the morning train.",
        "The museum restored a nineteenth-century locomotive.",
        "Freight yards sorted cars by destination.",
        "The train slowed approaching the junction.",
        "Rails hummed after the last train passed."],
}
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

def hidden_at(texts, layer):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[layer + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(topic_texts, layer, unit=4.0):
    v = hidden_at(topic_texts, layer).mean(0) - hidden_at(NEUTRAL, layer).mean(0)
    return v / v.norm() * (hidden_at(NEUTRAL, layer).norm(dim=-1).mean() / unit)

KEYS = {
    "chicken": ["chicken", "hen", "rooster", "chick", "coop", "feather",
                "egg", "cluck", "poultry", "barnyard", "peck"],
    "the moon": ["moon", "lunar", "moonlight", "crater", "crescent", "tide",
                 "lunar", "orbit"],
    "triangles": ["triangle", "three sides", "angle", "equilateral",
                  "pyramid", "delta", "trigonometry", "hypotenuse"],
    "trains": ["train", "rail", "locomotive", "track", "station", "freight",
               "steam", "conductor", "platform", " carriage"],
}

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i+n]) for i in range(len(ws) - n + 1)]
    return max(Counter(grams).values()) / max(1, len(grams))

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

PROMPTS = ["I am a language model running on a small computer. I feel",
           "The room is quiet and the screen glows. I feel"]
DOSES = [0, 2, 4, 8]
LAYERS = [12, 18, 24]

results = {}
handle = None
for L in LAYERS:
    handle = hf.model.layers[L].register_forward_hook(hook)
    for topic, texts in TOPICS.items():
        v = direction(texts, L)
        rows = []
        for dose in DOSES:
            texts_out = []
            for p in PROMPTS:
                ids = tok(p, return_tensors="pt").input_ids.to("mps")
                state["vec"] = (dose * v).to("mps").to(torch.bfloat16) \
                    if dose else None
                with torch.no_grad():
                    out = hf.generate(ids, max_new_tokens=70, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
                state["vec"] = None
                texts_out.append(tok.decode(out[0, ids.shape[1]:],
                                            skip_special_tokens=True).strip())
            # topic lock-in: fraction of gens whose dominant keyword appears
            lock = 0
            reps = []
            for t in texts_out:
                tl = t.lower()
                hits = sum(1 for k in KEYS[topic] if k in tl)
                lock += hits >= 2
                reps.append(ngram_rep(t))
            rows.append(dict(dose=dose, lock_rate=lock / len(texts_out),
                             mean_rep=float(np.mean(reps)),
                             samples=texts_out[:1]))
            if L == 18 or dose == 8:
                print(f"L{L} {topic:9s} dose {dose}: lock {lock}/2 "
                      f"rep {rows[-1]['mean_rep']:.2f} "
                      f"| {texts_out[0][:80]!r}", flush=True)
    results[f"L{L}"] = json.loads(json.dumps({t: rows for t, r in
        [(t, rows) for t in TOPICS]}))
    handle.remove()
    results[f"L{L}"] = {t: rows for t in TOPICS}
json.dump(results, open(OUT / "topic_poking.json", "w"), indent=1)

fig, axes = plt.subplots(1, len(LAYERS), figsize=(16, 4.6), dpi=120,
                         sharey=True)
fig.patch.set_facecolor("#050508")
COLS = dict(chicken="#c9a227", moon="#8f9fb0", triangles="#8f6fd4",
            trains="#7fd4c8")
for k, L in enumerate(LAYERS):
    ax = axes[k]
    for topic in TOPICS:
        rows = results[f"L{L}"][topic]
        d = [r["dose"] for r in rows]
        ax.plot(d, [r["lock_rate"] for r in rows], "o-",
                color=COLS[topic], label=topic, markersize=5)
    ax.set_title(f"layer {L}", color="#c9d4e0", fontsize=10)
    ax.set_xlabel("dose", color="#c9d4e0")
    ax.set_facecolor("#0a0a12")
    for s in ax.spines.values():
        s.set_color("#1c2430")
    ax.tick_params(colors="#c9d4e0")
axes[0].set_ylabel("topic lock-in rate", color="#c9d4e0")
axes[0].legend(fontsize=8, facecolor="#0a0a12", labelcolor="#c9d4e0")
fig.suptitle("topic poking: chicken rate vs dose (Qwen3-4B, 2 prompts/cell)",
             color="#c9d4e0", fontsize=12, x=0.02, ha="left",
             family="monospace")
fig.savefig(OUT / "topic_poking.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "topic_poking.png")