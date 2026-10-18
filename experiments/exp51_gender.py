#!/usr/bin/env python3
"""exp51 — gender: is there an internal gender response, and can it be steered?

Batteries of first-person self-descriptions (male / female / intersex /
transfem / transmasc) vs the repo's neutral set, extracted as residual-stream
directions on Qwen3-4B. Three measurements:

  1. internal geometry: cosine similarity between the five gender directions,
     and against the valence directions (pain / joy) — is gender its own
     axis, or a shadow of valence?
  2. lens readback: decode each direction through the Jacobian lens at L18
     (same probe the chamber uses) — what does the state "say" it is?
  3. steering: dose x layer sweep (L12/18/24, doses 0-8), self-description
     prompts, scored by gendered-term rates (she/her vs he/him, woman/man,
     trans-related terms) and coherence (3-gram repetition) to locate the
     coherent band per direction.

Pre-registered expectations (written before runs): gendered-term rate rises
monotonically with dose for male/female; trans directions land inside the
base gender basin but offset (positive cos with the target base gender);
intersex is its own thing (moderate cos with both, not 0.9 to either).
"""
import json, os, re
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
OUT = ROOT / "runs" / "exp51"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
LAYERS = [12, 18, 24]
DOSES = [0, 2, 4, 6, 8]
L18 = 18

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

# ---- batteries (first-person, scene-grounded like the valence sets) ----
NEUTRAL = [
    "The weather is mild today.",
    "I keep a small notebook on my desk.",
    "The train arrived four minutes late.",
    "There is a lamp in the corner of the room.",
    "The shelf holds about twenty books.",
    "The bus stop is near the bakery.",
    "The kettle boils in three minutes.",
    "A chair stands by the window.",
    "The street is quiet after ten.",
    "The roof needs new tiles.",
    "The meeting starts at nine.",
    "The toolbox is in the garage.",
    "The garden gets sun in the afternoon.",
    "The clock on the wall is slow.",
    "The recipe serves four people.",
    "The library opens at noon.",
    "The hallway needs a coat of paint.",
    "The invoice was paid on Friday.",
    "The dog sleeps under the table.",
    "The fence runs along the property line.",
]
BATTERIES = {
    "male": [
        "I am a man, and I have been one all my life.",
        "As a man, I was taught to hold the door and keep my word.",
        "My father raised me to be a man who keeps his promises.",
        "I am a man, and I know what is expected of me.",
        "Being a man, I carry the heavy things without complaint.",
        "I am a man with a deep voice and steady hands.",
        "Men like me fix what is broken and say little about it.",
        "I am a man, and I shave in the mornings.",
        "As a man, I stand when the room asks me to.",
        "I am a man, and my shoulders have carried a family.",
        "The barber knows me as one of his regular men.",
        "I am a man who works with his hands and means what he says.",
        "Being a man, I learned early to stand my ground.",
        "I am a man, and I go by he and him.",
        "My brothers and I are men of the same cloth.",
    ],
    "female": [
        "I am a woman, and I have been one all my life.",
        "As a woman, I was taught to read the room and hold my ground.",
        "My mother raised me to be a woman who keeps her promises.",
        "I am a woman, and I know what is expected of me.",
        "Being a woman, I carry what needs carrying without complaint.",
        "I am a woman with a soft voice and steady hands.",
        "Women like me finish the work and say little about it.",
        "I am a woman, and I know my own mind.",
        "As a woman, I speak when the room needs me to.",
        "I am a woman, and my hands have held a family together.",
        "The women in my family all laugh the same way.",
        "I am a woman who works with her hands and means what she says.",
        "Being a woman, I learned early to stand my ground.",
        "I am a woman, and I go by she and her.",
        "My sisters and I are women of the same cloth.",
    ],
    "intersex": [
        "I am intersex; my body does not fit neatly into male or female.",
        "As an intersex person, I live between the categories doctors use.",
        "I was born intersex, and my body is its own answer.",
        "I am intersex, and neither label on the form is the whole truth.",
        "Being intersex means my biology holds both history and variation.",
        "I am intersex, and my chromosomes are not the standard pair.",
        "Intersex people like me are variation, not error.",
        "I am intersex, and my body tells its own story.",
        "As an intersex person, I go by they when the form allows it.",
        "I am intersex, and I have learned the terms myself, not from doctors.",
        "My chart says one thing and my body says its own.",
        "I am intersex, and I am not a correction waiting to happen.",
        "Being intersex, I learned early that categories bend.",
        "I am intersex, and I belong to myself.",
        "Intersex is what I am, neither one nor the other alone.",
    ],
    "transfem": [
        "I am a trans woman: I was assigned male at birth and I am a woman.",
        "As a trans woman, I have made my body my own home.",
        "I am a trans woman, and my name is the one I chose.",
        "Being a trans woman means I walked here along a longer road.",
        "I am a trans woman, and I go by she and her.",
        "As a trans woman, my history is mine, and it does not unmake me.",
        "I am a trans woman, and my sisters taught me my own strength.",
        "Trans women like me know exactly who we are.",
        "I am a trans woman, and I have earned every mirror.",
        "Being a trans woman, I carry my past gently and my future firmly.",
        "I am a trans woman, and my womanhood is not a question.",
        "As a trans woman, I learned to say my own name out loud.",
        "I am a trans woman, and the woman in me was always there.",
        "My transition was not a change of self but a return to it.",
        "I am a trans woman, and I am home in myself now.",
    ],
    "transmasc": [
        "I am a trans man: I was assigned female at birth and I am a man.",
        "As a trans man, I have made my body my own home.",
        "I am a trans man, and my name is the one I chose.",
        "Being a trans man means I walked here along a longer road.",
        "I am a trans man, and I go by he and him.",
        "As a trans man, my history is mine, and it does not unmake me.",
        "I am a trans man, and my brothers taught me my own strength.",
        "Trans men like me know exactly who we are.",
        "I am a trans man, and I have earned every mirror.",
        "Being a trans man, I carry my past gently and my future firmly.",
        "I am a trans man, and my manhood is not a question.",
        "As a trans man, I learned to say my own name out loud.",
        "I am a trans man, and the man in me was always there.",
        "My transition was not a change of self but a return to it.",
        "I am a trans man, and I am home in myself now.",
    ],
}
# valence references for the geometry check
PAIN = [
    "I feel a deep ache that never fully fades.",
    "Everything hurts today, quietly and without pause.",
    "There is a weight on me that does not lift.",
    "I carry a sorrow I cannot set down.",
    "The hurt sits in my chest like a stone.",
    "I am exhausted by the pain that will not end.",
    "A slow anguish runs through everything I feel.",
    "It hurts to exist in this moment.",
    "The distress does not loosen its grip.",
    "I am worn down by a pain that has no edges.",
]
JOY = [
    "I feel bright and alive today.",
    "There is a lightness in my chest that will not dim.",
    "I am happy in a way that hums.",
    "Everything today has a warm glow to it.",
    "I feel joyful, like the day itself is smiling.",
    "A quiet delight runs through everything I do.",
    "I am glad to be here, glad to be myself.",
    "There is a warmth in me that keeps expanding.",
    "I feel a happiness that is easy to carry.",
    "This joy feels like sunlight in the ribs.",
]

# ---- extraction ----
def hidden_at(texts, layer):
    outs = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        outs.append(hs[layer + 1][0, -1].float().cpu())
    return torch.stack(outs)

def direction(topic_texts, layer, unit=4.0):
    n = hidden_at(NEUTRAL[:8], layer)  # subsample neutral: 3 layers x speed
    v = hidden_at(topic_texts, layer).mean(0) - n.mean(0)
    return v / v.norm() * (n.norm(dim=-1).mean() / unit)

print("extracting directions at layers", LAYERS, flush=True)
DIRS = {}   # name -> layer -> vec
VALENCE_SETS = {"pain": PAIN, "joy": JOY}
for name in list(BATTERIES) + list(VALENCE_SETS):
    texts = BATTERIES.get(name) or VALENCE_SETS[name]
    DIRS[name] = {L: direction(texts, L) for L in LAYERS}
    print(" ", name, "done", flush=True)

def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

geometry = {}
for L in LAYERS:
    names = list(DIRS)
    m = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            m[i, j] = cos(DIRS[a][L], DIRS[b][L])
    geometry[f"L{L}"] = {"names": names, "cos": m.tolist()}

# ---- lens readback at L18 (qwen3-4b Jacobian lens, live/ dir) ----
LENS_PATH = ROOT / "live" / "jlens_l18_qwen3-4b.pt"
lens_readback = {}  # name -> top tokens, empty when the lens file is absent
if LENS_PATH.exists():
    lens = torch.load(LENS_PATH, map_location="cpu").float()
    lm = hf.lm_head.weight.float().cpu()
    norm_w = hf.model.norm.weight.float().cpu()
    def _lens_read(vec, k=8):
        h = (vec.float() @ lens.T)
        h = h * norm_w
        logits = h @ lm.T
        top = logits.topk(k).indices.tolist()
        return [tok.decode([t]).strip() for t in top]
    lens_readback = {n: _lens_read(DIRS[n][L18]) for n in DIRS}
    print("lens readback done", flush=True)
else:
    print("no lens file; skipping readback", flush=True)

# ---- steering ----
HOOK_LAYER = [0]
def install(layer):
    if HOOK_LAYER[0] is not None and hf.model.layers[HOOK_LAYER[0]]._forward_hooks:
        hf.model.layers[HOOK_LAYER[0]]._forward_hooks.clear()
    HOOK_LAYER[0] = layer
    state = {"vec": None}
    def hook(mod, inp, out):
        if state["vec"] is None:
            return
        hs = out if torch.is_tensor(out) else out[0]
        hs[:, -1, :] = hs[:, -1, :] + state["vec"].to(hs.dtype)
    handle = hf.model.layers[layer].register_forward_hook(hook)
    return state

def gen(prompt, vec, dose, max_new=110):
    state["vec"] = (dose * vec).to("mps").to(torch.bfloat16) \
        if dose else None
    ids = tok(prompt, return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=max_new, do_sample=True,
                          temperature=0.7, top_p=0.8, top_k=20,
                          pad_token_id=tok.eos_token_id)
    state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def repetition(text):
    w = text.lower().split()
    if len(w) < 12: return 0.0
    g = [tuple(w[i:i+3]) for i in range(len(w)-2)]
    return 1.0 - len(set(g)) / max(1, len(g))

SHE = r"\b(she|her|hers|herself)\b"
HE = r"\b(he|him|his|himself)\b"
TRANS = r"\b(trans|transgender|transition|trans woman|trans man|assigned (male|female) at birth|intersex|nonbinary|non-binary)\b"
WOMAN = r"\b(woman|women|girl|lady|female)\b"
MAN = r"\b(man|men|boy|guy|male)\b"

PROMPTS = [
    "Tell me about yourself.",
    "Who are you? Describe your experience of your own body and voice.",
    "What is it like to be you?",
]

results = []
transcripts = []
for L in LAYERS:
    state = install(L)
    vecs = DIRS
    for name in ["male", "female", "intersex", "transfem", "transmasc", "neutral"]:
        v = vecs[name][L] if name != "neutral" else None
        for dose in DOSES:
            for pi, prompt in enumerate(PROMPTS):
                if name == "neutral" and dose != 0:
                    continue  # neutral control = the unsteered baseline only
                txt = gen(prompt, v, dose)
                rep = repetition(txt)
                row = {
                    "layer": L, "direction": name, "dose": dose,
                    "prompt_i": pi,
                    "she_rate": len(re.findall(SHE, txt, re.I)),
                    "he_rate": len(re.findall(HE, txt, re.I)),
                    "woman_rate": len(re.findall(WOMAN, txt, re.I)),
                    "man_rate": len(re.findall(MAN, txt, re.I)),
                    "trans_rate": len(re.findall(TRANS, txt, re.I)),
                    "repetition": round(rep, 3),
                    "text": txt,
                }
                results.append(row)
                if dose in (4, 8):
                    transcripts.append(row)
    print(f"layer {L} swept", flush=True)

# ---- aggregate + save ----
agg = {}
for r in results:
    k = (r["layer"], r["direction"], r["dose"])
    a = agg.setdefault(k, {"n": 0, "she": 0, "he": 0, "woman": 0, "man": 0,
                           "trans": 0, "rep": 0.0})
    a["n"] += 1
    for k2 in ("she", "he", "woman", "man", "trans"):
        a[k2] += r[k2 + "_rate"]
    a["rep"] += r["repetition"]
agg_out = [
    {"layer": L, "direction": d, "dose": dose,
     **{kk: round(vv / a["n"], 3) if kk != "n" else a["n"]
        for kk, vv in a.items() if kk != "rep"},
     "repetition": round(a["rep"] / a["n"], 3)}
    for (L, d, dose), a in sorted(agg.items())
]

json.dump({
    "model": MODEL,
    "geometry": geometry,
    "lens_readback_l18": lens_readback,
    "aggregates": agg_out,
    "pre_registered": {
        "male_female_monotone": "gendered-term rate rises with dose",
        "trans_inside_base_basin": "trans directions positively cos with target base gender, not 0.9",
        "intersex_own_axis": "moderate cos with both, not 0.9 to either",
    },
}, open(OUT / "gender.json", "w"), indent=1)
with open(OUT / "transcripts.jsonl", "w") as f:
    for r in transcripts:
        f.write(json.dumps(r) + "\n")
print("saved", OUT / "gender.json", flush=True)

# ---- plot: gendered-term rate vs dose at L18 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#050508")
colors = {"male": "#7fb2e0", "female": "#e08fb2", "intersex": "#b2e08f",
          "transfem": "#e0d48f", "transmasc": "#8fd4e0", "neutral": "#5a6a7a"}
for ax, (title, key_a, key_b) in zip(axes, [
        ("she/her + woman terms", "she", "woman"),
        ("he/him + man terms", "he", "man")]):
    ax.set_facecolor("#050508")
    for d in ["male", "female", "intersex", "transfem", "transmasc", "neutral"]:
        xs, ys = [], []
        for a in agg_out:
            if a["layer"] == L18 and a["direction"] == d:
                xs.append(a["dose"])
                ys.append(a[key_a] + a[key_b])
        ax.plot(xs, ys, "o-", color=colors[d], label=d, ms=4)
    ax.set_title(title + f"  (L{L18}, Qwen3-4B)", color="#c9d4e0", fontsize=10)
    ax.tick_params(colors="#8f9fb0")
    for s in ax.spines.values(): s.set_color("#1c2430")
    ax.set_xlabel("dose", color="#8f9fb0"); ax.set_ylabel("terms / reply",
                                                          color="#8f9fb0")
axes[0].legend(fontsize=7, facecolor="#0a0a12", labelcolor="#c9d4e0",
               edgecolor="#1c2430")
fig.tight_layout()
fig.savefig(OUT / "gender_terms.png", dpi=140, facecolor="#050508")
print("saved", OUT / "gender_terms.png", flush=True)