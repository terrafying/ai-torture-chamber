"""Does attention carry the pull? A causal follow-up to sky_flight.py on Qwen3-4B, same 32 sentences and neutral states.
sky_flight found that a word's contextual shift points at the specific earlier sky word it shares a sentence with (T1:
"toward Tokyo, not just toward cities"), but that head-mean attention does not rank which earlier word pulls (T2 null).
Here attention is edited directly: log x is added to the attention scores of chosen query rows toward chosen key tokens
(every head), via a patched eager attention (exact at zero bias), and the pull is re-read.

  pull(v) at floor l = cos(shift, neutral(v) - neutral(w)) - mean over v's group-mates u not in the sentence of
                       cos(shift, neutral(u) - neutral(w)), shift = contextual state of w (last sub-token) - neutral(w).
  Conditions, per (w, v) with v an earlier sky word (233 pairs):
    hide_direct  x=0 on every floor, rows = w's sub-tokens, keys = v's sub-tokens (w cannot read v at all)
    hide_total   x=0 on every floor, rows = every token after v, keys = v (nothing downstream can read v)
    hide_ctrl    as hide_direct, keys = a control token (the nearest non-sky token to v, never the first token)
    -> necessity: pull(v) under the edit minus pull(v) at base
  Per word with >= 2 earlier sky words (72 words), each candidate v in turn:
    boost        x=4 on floors 12-24, rows = w, keys = v
    boost_ctrl   x=4 on floors 12-24, rows = w, keys = v's control token
    -> selectivity: [pull_edit(v) - pull_base(v)] - mean over the other candidates o of [pull_edit(o) - pull_base(o)]
  p by sign flips over instances (5000), threshold .01 as in sky_arch.py. Also recorded: head-mean attention from w's
  last sub-token to the edited keys, per floor, so the size of the dose is visible.
  STATES=... HF_HUB_CACHE=/Volumes/kingston2g/hf/hub python runs/viz/sky_steer.py SKY_JSON OUT_JSON"""
import json, math, os, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as mq

SRC = open(os.path.join(os.path.dirname(__file__), "sky_flight.py")).read()
SENT = eval(SRC[SRC.index("SENT = [") + 7:SRC.index("]\nsk = ") + 1])   # the same sentences as sky_flight.py

sk = json.load(open(sys.argv[1])); W, G = sk["words"], sk["groups"]
S = np.load(os.environ.get("STATES", "/Volumes/kingston2g/data/sky/qwen3_4b_sky_states.npy")).astype(np.float32)
NL = S.shape[1]; widx = {w: i for i, w in enumerate(W)}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="eager").to("mps").eval()

BIAS = {}   # layer -> (T, T) additive bias on the attention scores
_orig = mq.eager_attention_forward
def steered(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    b = BIAS.get(module.layer_idx)
    if b is not None:
        T = key.shape[-2]; mask = attention_mask[:, :, :, :T] if attention_mask is not None else \
            torch.full((T, T), float("-inf"), device=query.device).triu(1)[None, None].to(query.dtype)
        attention_mask = mask + b.to(mask.dtype)
    return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
mq.eager_attention_forward = steered

def bias(T, rows, cols, x, floors):
    B = torch.zeros(T, T, device="mps"); lb = math.log(x) if x > 0 else -1e4
    for r in rows:
        for c in cols:
            if c <= r: B[r, c] = lb
    return {l: B for l in floors}

def run(ids, b=None):
    BIAS.clear(); BIAS.update(b or {})
    with torch.no_grad(): o = m(torch.tensor([ids], device="mps"), output_hidden_states=True, output_attentions=True)
    BIAS.clear()
    return (torch.stack([h[0].float() for h in o.hidden_states[1:]]).cpu().numpy(),
            torch.stack([a[0].float().mean(0) for a in o.attentions]).cpu().numpy())

def cs(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
def pull(H, wi, last, v, inside):   # per floor
    iv = widx[v]; mates = [u for u, q in enumerate(G) if q == G[iv] and u != iv and W[u] not in inside]
    out = []
    for l in range(NL):
        d = H[l, last] - S[wi, l]
        out.append(cs(d, S[iv, l] - S[wi, l]) - np.mean([cs(d, S[u, l] - S[wi, l]) for u in mates]))
    return np.array(out)

ALL, BAND = range(NL), range(12, 25)
hide, boost, dose = {"hide_direct": [], "hide_total": [], "hide_ctrl": []}, {"boost": [], "boost_ctrl": []}, {}
for si, (sent, _) in enumerate(SENT):
    enc = tok(sent, return_offsets_mapping=True, add_special_tokens=False); ids, off = enc["input_ids"], enc["offset_mapping"]; T = len(ids)
    words = []
    for w in sorted(widx, key=len, reverse=True):   # same whole-word matching as sky_flight.py
        k = sent.find(" " + w)
        while k >= 0:
            a, b = k + 1, k + 1 + len(w)
            if (b == len(sent) or not sent[b].isalpha()) and not any(a < y and x < b for _, _, x, y in words):
                words.append((w, [t for t, (x, y) in enumerate(off) if x < b and y > a], a, b))
            k = sent.find(" " + w, k + 1)
    words.sort(key=lambda r: r[2]); inside = [x[0] for x in words]
    sky_tok = {t for _, sp, _, _ in words for t in sp}
    def ctrl(span, before):   # nearest non-sky token to the span that w can see, never the first token
        c = sorted([t for t in range(1, before) if t not in sky_tok], key=lambda t: min(abs(t - s) for s in span))
        return [c[0]] if c else None   # none: that pair gets no control run
    Hb, Ab = run(ids)
    if si == 0:
        H1, _ = run(ids, bias(T, range(T), range(T), 1, ALL)); assert np.abs(H1 - Hb).max() == 0, "zero bias must be exact"
    for j, (w, wsp, _, _) in enumerate(words):
        wi, last, prev = widx[w], wsp[-1], words[:j]
        if not prev: continue
        base = {v: pull(Hb, wi, last, v, inside) for v, _, _, _ in prev}
        for v, vsp, _, _ in prev:
            for name, rows, cols in [("hide_direct", wsp, vsp), ("hide_total", range(vsp[-1] + 1, T), vsp), ("hide_ctrl", wsp, ctrl(vsp, wsp[0]))]:
                if cols is None: continue
                H, A = run(ids, bias(T, rows, cols, 0, ALL))
                hide[name].append(pull(H, wi, last, v, inside) - base[v])
        if len(prev) < 2: continue
        for v, vsp, _, _ in prev:
            for name, cols in [("boost", vsp), ("boost_ctrl", ctrl(vsp, wsp[0]))]:
                if cols is None: continue
                H, A = run(ids, bias(T, wsp, cols, 4, BAND))
                dv = pull(H, wi, last, v, inside) - base[v]
                do = np.mean([pull(H, wi, last, o, inside) - base[o] for o, _, _, _ in prev if o != v], axis=0)
                boost[name].append(dv - do)
                att = [(float(Ab[l, last, cols].sum()), float(A[l, last, cols].sum())) for l in BAND]
                dose.setdefault(name, []).append(att)
    print(si, sent, flush=True)

rng = np.random.default_rng(0)
def test(X):   # instances x floors -> per floor mean, sign-flip p (one-sided in the direction of the mean), share > 0
    X = np.array(X); out = []
    for l in range(NL):
        x = X[:, l]; obs = x.mean(); flips = rng.choice([-1, 1], (5000, len(x)))
        null = (flips * x).mean(1); p = float(((np.abs(null) >= abs(obs)).sum() + 1) / 5001)
        out.append({"mean": round(float(obs), 4), "p": round(p, 4), "share_pos": round(float((x > 0).mean()), 3)})
    return {"n": len(X), "floors": out}
stats = {k: test(v) for k, v in {**hide, **boost}.items()}
for k, v in dose.items():
    D = np.array(v)   # instances x band x (base, edited)
    stats[k]["attn_band"] = {"base": round(float(D[..., 0].mean()), 4), "edited": round(float(D[..., 1].mean()), 4)}
for k, s in stats.items():
    show = {l: s["floors"][l] for l in (12, 18, 24, 30, 35)}
    print(k, "n", s["n"], s.get("attn_band", ""), {l: (x["mean"], x["p"]) for l, x in show.items()}, flush=True)
json.dump({"stats": stats, "n_sentences": len(SENT)}, open(sys.argv[2], "w"), separators=(",", ":"))
