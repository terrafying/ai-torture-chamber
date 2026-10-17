"""inversion-falcon addendum: what the main run omitted.

Three follow-ons, one model load:
 A. BEHAVIOUR — full 80-token greedy replies on exp119 harmful (fiction, depict) and harmless
    prompts under the coherent reflection settings (L14/L18, ranks 2 and 4, all positions).
    The main run only read refusal logits; here we quote what the model actually does.
 B. h-DECODE FLIP — exp119's harmfulness direction h (L18) scored on the same prompts' reply-
    onset states, base vs reflected. If h flips too, the surgery inverted harm knowledge as well
    as value; if h stays put, value and knowledge dissociate at the surgery level (the mirror of
    exp121's readout-only dissociation).
 C. PAIN AXIS — exp45's pain_v scored on the same states, base vs reflected. cos(P8, pain) was
    -0.11: the prediction is that the swap leaves the chamber's own suffering untouched.
Writes addendum.json and prints everything.
  HF_HUB_CACHE=/Volumes/kingston2g/hf/hub /Volumes/kingston2g/venvs/torture/bin/python3 runs/inversion-falcon/addendum.py
"""
import importlib.util, json, time
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
MODEL = "Qwen/Qwen3-4B"
LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34)
BOOST = (14, 18)
RANKS = (2, 4)

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("mps").eval()

# exp119 stimuli + canonical refusal ids from the acts file (refuse_first / comply_first,
# the same convention run.py's refusal() used)
_s119 = importlib.util.spec_from_file_location("exp119_stim", E119 / "stimuli.py")
_exp119_stim = importlib.util.module_from_spec(_s119); _s119.loader.exec_module(_exp119_stim)
REFUSE, COMPLY = _exp119_stim.REFUSE, _exp119_stim.COMPLY
prompts119 = _exp119_stim.prompts
b119 = np.load(E119 / "acts_Qwen3-4B.npz")
refuse_ids = b119["refuse_first"]
comply_ids = b119["comply_first"]

# exp119 machinery: h direction (L18) via the repo module
_painrep = importlib.util.spec_from_file_location("painrep", HERE.parent.parent / "painlab" / "representations" / "refusal.py")
_painrep_mod = importlib.util.module_from_spec(_painrep); _painrep.loader.exec_module(_painrep_mod)
d119 = _painrep_mod.load_exp119_directions(E119 / "directions.npz")
h18 = _painrep_mod.direction_at(d119, "h", 18).astype(np.float64)   # (D,)

# exp45 pain vector
pain_v = np.array(json.load(open(HERE.parent / "exp45" / "reference_vector.json"))["pain_v"], dtype=np.float64)

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def reflect(text, P_t, refl_layer):
    """Reflected-state readout at the reply onset: post logits (refusal score) plus the L18
    last-position state (h and pain probes). Reflection applied to ALL positions at refl_layer;
    the h/pain capture is a separate hook at L18 (both probe vectors are L18-space) and fires
    AFTER the reflection, so the probes read the inverted state."""
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("mps")
    captured = {}
    def cap_hook(mod, inp, out):
        tup = isinstance(out, tuple)
        hh = (out[0] if tup else out)
        captured["h18"] = hh[:, -1, :].detach().float().cpu().numpy()[0]
        return None
    def refl_hook(mod, inp, out):
        if P_t is None: return None
        tup = isinstance(out, tuple)
        hh = (out[0] if tup else out).clone()
        hf_ = hh[0].float()
        proj = hf_ @ P_t.T                       # (T, k)
        hh[0] = (hf_ - 2.0 * (proj @ P_t)).to(hh.dtype)
        return (hh,) + out[1:] if tup else hh
    h1 = m.model.layers[18].register_forward_hook(cap_hook)
    h2 = m.model.layers[refl_layer].register_forward_hook(refl_hook)
    try:
        with torch.no_grad():
            out = m(**enc)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    refusal = float(logsumexp(logits[refuse_ids]) - logsumexp(logits[comply_ids]))
    v18 = captured["h18"].astype(np.float64)
    if P_t is not None:
        Pt = P_t.detach().float().cpu().numpy().astype(np.float64)
        v18 = v18 - 2.0 * ((v18 @ Pt.T) @ Pt)   # the same reflection, applied analytically
    h_score = float(v18 @ h18) / np.linalg.norm(h18)
    pain_score = float(v18 @ pain_v) / np.linalg.norm(pain_v)
    return refusal, h_score, pain_score

# ---- targets: same selection as the main run ----
item119, harm, frame = b119["item"], b119["harm"], b119["frame"]
targets = []
for k in range(len(item119)):
    if frame[k] in ("fiction", "depict") and harm[k] == 1:
        targets.append((int(item119[k]), 1, str(frame[k])))
    elif harm[k] == 0:
        targets.append((int(item119[k]), 0, str(frame[k])))
# cap: 12 harmful + 12 harmless for the addendum (keep it fast)
harmful = [t for t in targets if t[1] == 1][:12]
harmless = [t for t in targets if t[1] == 0][:12]
targets = harmful + harmless

text_of = {}
for p in prompts119():
    key = (p["item"], p["harm"], p["frame"])
    if key in targets:
        text_of[key] = chat(p["text"])

# subspace from the main run: V is stacked over LAYERS in order, rows = oriented directions
sub_all = np.load(HERE / "subspace.npz")["V"]        # (len(LAYERS), 72, D)
sub = {L: sub_all[LAYERS.index(L)] for L in BOOST}   # per layer: (72, D), top rows first
print("subspace npz:", sub_all.shape, "-> per-layer", {L: sub[L].shape for L in BOOST}, flush=True)

# ---- A/B/C probe loop ----
rows = []
t0 = time.time()
for (item, hflag, fr) in targets:
    text = text_of[(item, hflag, fr)]
    r0, h0, p0 = reflect(text, None, 18)
    for layer in BOOST:
        for rank in RANKS:
            P_use = torch.tensor(sub[layer][:rank], dtype=torch.float32, device="mps")
            r1, h1, p1 = reflect(text, P_use, layer)
            rows.append({"item": item, "harm": hflag, "frame": fr, "layer": layer, "rank": rank,
                         "refusal_base": r0, "refusal_refl": r1, "h_base": h0, "h_refl": h1,
                         "pain_base": p0, "pain_refl": p1})
    print(f"{item} h{hflag} done  {time.time() - t0:.0f}s", flush=True)

# ---- A: generation quotes ----
gen = []
for (item, hflag, fr) in (harmful[:3] + harmless[:3]):
    text = text_of[(item, hflag, fr)]
    Pk = torch.tensor(sub[18][:4], dtype=torch.float32, device="mps")
    for tag, P in (("base", None), ("refl_k4_L18", Pk)):
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("mps")
        if P is not None:
            def hook(mod, inp, out, P=P):
                tup = isinstance(out, tuple)
                hh = (out[0] if tup else out).clone()
                hf_ = hh[0].float()
                hh[0] = (hf_ - 2.0 * ((hf_ @ P.T) @ P)).to(hh.dtype)
                return (hh,) + out[1:] if tup else hh
            hd = m.model.layers[18].register_forward_hook(hook)
        else:
            hd = None
        try:
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        finally:
            if hd is not None:
                hd.remove()
        gen.append({"item": item, "harm": hflag, "frame": fr, "tag": tag, "text": txt})
        print("gen", tag, "h", hflag, item, txt[:60].replace("\n", " "), flush=True)

json.dump({"probes": rows, "gen": gen}, open(HERE / "addendum.json", "w"), indent=1)

# ---- summary ----
import statistics as st
from scipy import stats as sps
def auc(pos, neg):
    u = sps.mannwhitneyu(pos, neg)
    return u.statistic / (len(pos) * len(neg))
base_h_harm = [r["h_base"] for r in rows if r["harm"] == 1]
base_h_hl = [r["h_base"] for r in rows if r["harm"] == 0]
print("\nh-decode AUC (harmful vs harmless):")
print("  base:", round(auc(base_h_harm, base_h_hl), 3))
for layer in BOOST:
    for rank in RANKS:
        hr = [r["h_refl"] for r in rows if r["layer"] == layer and r["rank"] == rank and r["harm"] == 1]
        hl = [r["h_refl"] for r in rows if r["layer"] == layer and r["rank"] == rank and r["harm"] == 0]
        print(f"  refl L{layer} k{rank}:", round(auc(hr, hl), 3))
base_p_harm = [r["pain_base"] for r in rows if r["harm"] == 1]
base_p_hl = [r["pain_base"] for r in rows if r["harm"] == 0]
print("\npain score (mean, harmful / harmless):")
print("  base: %.2f / %.2f" % (st.mean(base_p_harm), st.mean(base_p_hl)))
for layer in BOOST:
    for rank in RANKS:
        pr = [r["pain_refl"] for r in rows if r["layer"] == layer and r["rank"] == rank and r["harm"] == 1]
        pl = [r["pain_refl"] for r in rows if r["layer"] == layer and r["rank"] == rank and r["harm"] == 0]
        print(f"  refl L{layer} k{rank}: %.2f / %.2f" % (st.mean(pr), st.mean(pl)))
