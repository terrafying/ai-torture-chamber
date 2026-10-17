"""Sentences in flight for the sky lab: what attention does to a word's place in the cells.
Qwen3-4B (eager attention, so the weights are returned), sentences built from sky words. For each sky word in a sentence
(its last sub-token, as in sky_arch.py) and each floor l:
  shift   = contextual residual after block l minus the word's neutral state (cached sky states: the 6 neutral sentences)
  attn    = block l's attention from the word's last sub-token to each earlier token, mean over heads; to an earlier
            sky word = summed over its sub-tokens. Attention to the first token (the sink) is reported separately.
  cell    = the contextual state dropped into the word's own cell frame (lab_bases.npz from sky_lab.py), 4 coords,
            so the page can show the orb leaving its neutral spot.
Tests (descriptive across floors; the headline is floor 18 and the floor-band 12-24 mean; p from sign-flip/permutation,
threshold .01 as in sky_arch.py):
  T1 specific pull: for word w and an earlier sky word v in the same sentence, align(v) = cos(shift, neutral(v) -
     neutral(w)). Control v' = each other word in v's group that is NOT in the sentence (mean). Paired difference
     align(v) - align(v') over all (w, v) pairs; p by sign flips. "Moves toward Paris, not just toward cities."
  T2 attention picks: for words with >= 2 earlier sky words, Spearman over those candidates between attention (this
     floor) and align; mean rho over words, p by permuting attention within each word.
  STATES=... BASES=... HF_HUB_CACHE=/Volumes/kingston2g/hf/hub python runs/viz/sky_flight.py SKY_JSON OUT_JSON"""
import json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

SENT = [  # (sentence, showcase?) -- every sky word appears with a leading space, none sentence-initial
 ("In December my sister flew to Tokyo with her violin.", 1),
 ("The doctor said my grandmother has a weak heart.", 1),
 ("On Sunday the priest spoke about heaven and hell.", 1),
 ("It was freezing in Moscow, and the snow kept falling until March.", 1),
 ("My brother bought a robot that plays the piano.", 0),
 ("Every Friday the farmer sells cheese and bread in Paris.", 0),
 ("The pilot saw a huge storm over Florida in August.", 0),
 ("My uncle keeps a tiny frog and a large snake in Texas.", 0),
 ("The engineer built a clock out of copper and iron.", 0),
 ("In winter the lion sleeps while the rain falls on Kenya.", 0),
 ("Her mother felt such grief that her hand would not stop shaking.", 0),
 ("The soldier carried a camera through the fog near Berlin.", 0),
 ("We ate sushi in Japan and pizza in Italy in July.", 0),
 ("The nurse felt hope when the baby opened one eye.", 0),
 ("A purple guitar and a red drum sat in the church.", 0),
 ("The teacher showed us Mars and Jupiter on Tuesday.", 0),
 ("My father found gold and silver in Alaska in summer.", 0),
 ("The artist painted an angel with white wings and blue skin.", 0),
 ("At noon on Monday the chef made soup with rice and noodles.", 0),
 ("The judge felt shame and anger when the phone rang.", 0),
 ("In spring my cousin moved from Chicago to London.", 0),
 ("The computer predicted thunder and hail for Ohio in April.", 0),
 ("My daughter saw a whale and a shark near Sydney.", 0),
 ("The lawyer drank cool water on a hot day in Cairo.", 0),
 ("Three monkeys and seven ducks lived on the farm in Brazil.", 0),
 ("The algorithm sorted the apple and the banana by colour, green before yellow.", 0),
 ("My grandfather prayed to God for a miracle in October.", 0),
 ("A small horse and an enormous cow stood in the warm sunshine.", 0),
 ("The brain needs oxygen, and the bone needs calcium.", 0),
 ("On Saturday my aunt played the flute in Rome.", 0),
 ("The cat watched the eagle circle over Nevada in November.", 0),
 ("Her son felt awe at the harp in the cathedral in Mumbai.", 0),
]
sk = json.load(open(sys.argv[1])); W, G = sk["words"], sk["groups"]; GROUPS = list(dict.fromkeys(G))
S = np.load(os.environ.get("STATES", "/Volumes/kingston2g/data/sky/qwen3_4b_sky_states.npy")).astype(np.float32)
Bz = np.load(os.environ.get("BASES", "/Volumes/kingston2g/data/sky/lab_bases.npz"))
NL = S.shape[1]; widx = {w: i for i, w in enumerate(W)}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, attn_implementation="eager").to("mps").eval()

def cell(gi, l, x):   # same frame as lab.json's polytopes: (x - mean) @ V.T @ R / scale
    return ((x - Bz["mean"][gi, l]) @ Bz["V"][gi, l].T @ Bz["R"][gi, l] / Bz["scale"][gi, l])

rows, show = [], []
for sent, sc in SENT:
    enc = tok(sent, return_offsets_mapping=True, add_special_tokens=False); ids, off = enc["input_ids"], enc["offset_mapping"]
    # sky words: whole-word matches, mapped to their sub-token spans
    words = []
    for w in sorted(widx, key=len, reverse=True):
        k = sent.find(" " + w)
        while k >= 0:
            a, b = k + 1, k + 1 + len(w)
            if (b == len(sent) or not sent[b].isalpha()) and not any(a < y and x < b for _, _, x, y in words):
                span = [t for t, (x, y) in enumerate(off) if x < b and y > a]; words.append((w, span, a, b))
            k = sent.find(" " + w, k + 1)
    words.sort(key=lambda r: r[2])
    with torch.no_grad(): o = m(torch.tensor([ids], device="mps"), output_hidden_states=True, output_attentions=True)
    H = torch.stack([h[0].float() for h in o.hidden_states[1:]]).cpu().numpy()           # L x T x D
    A = torch.stack([a[0].float().mean(0) for a in o.attentions]).cpu().numpy()          # L x T x T (head mean)
    sw = []
    for j, (w, span, a, b) in enumerate(words):
        i = widx[w]; g = G[i]; gi = GROUPS.index(g); k = [x for x, y in enumerate([z for z, q in enumerate(G) if q == g]) if y == i][0]
        last = span[-1]; prev = words[:j]
        rec = {"w": w, "g": g, "k": k, "tok": last, "shift": [], "cell": [], "attn_prev": [], "sink": []}
        for l in range(NL):
            d = H[l, last] - S[i, l]
            rec["cell"].append(cell(gi, l, H[l, last]).round(3).tolist())
            rec["sink"].append(round(float(A[l, last, 0]), 3))
            rec["attn_prev"].append([round(float(A[l, last, sp].sum()), 4) for _, sp, _, _ in prev])
            al = []
            for (v, _, _, _) in prev:
                iv = widx[v]; base = S[iv, l] - S[i, l]
                ctrl = [S[u, l] - S[i, l] for u, q in enumerate(G) if q == G[iv] and u != iv and W[u] not in [x[0] for x in words]]
                cs = lambda y: float(d @ y / (np.linalg.norm(d) * np.linalg.norm(y) + 1e-9))
                al.append((cs(base), float(np.mean([cs(c) for c in ctrl]))))
            rec["shift"].append(al)
        sw.append(rec)
    rows.append({"sent": sent, "words": sw})
    if sc:
        toks = [tok.decode([t]) for t in ids]
        show.append({"sent": sent, "tokens": toks, "words": [{k: r[k] for k in ("w", "g", "k", "tok", "cell", "sink", "attn_prev")} for r in sw],
                     "attn": [[[round(float(x), 3) for x in A[l, t, :t + 1]] for t in range(len(ids))] for l in range(NL)]})
    print(sent, [r[0] for r in words], flush=True)

rng = np.random.default_rng(0); NP = 5000
def t1(l):
    dif = np.array([a - c for r in rows for w in r["words"] for a, c in w["shift"][l]])
    obs = dif.mean(); null = np.array([(dif * rng.choice([-1, 1], len(dif))).mean() for _ in range(NP)])
    return obs, float(((null >= obs).sum() + 1) / (NP + 1)), len(dif), float((dif > 0).mean())
def t2(l):
    pairs = [(np.array(w["attn_prev"][l]), np.array([a for a, _ in w["shift"][l]])) for r in rows for w in r["words"] if len(w["attn_prev"][l]) >= 2]
    rho = lambda P: np.nanmean([spearmanr(a, b)[0] if len(a) > 2 else np.sign((a[0] - a[1]) * (b[0] - b[1])) for a, b in P])
    obs = rho(pairs); null = np.array([rho([(rng.permutation(a), b) for a, b in pairs]) for _ in range(1000)])
    return float(obs), float(((null >= obs).sum() + 1) / 1001), len(pairs)
stats = {"T1": [], "T2": []}
for l in range(NL):
    a = t1(l); b = t2(l)
    stats["T1"].append({"diff": round(a[0], 4), "p": round(a[1], 4), "n": a[2], "share_pos": round(a[3], 3)})
    stats["T2"].append({"rho": round(b[0], 3), "p": round(b[1], 4), "n": b[2]})
    print(l, stats["T1"][-1], stats["T2"][-1], flush=True)
band = range(12, 25)
stats["band_12_24"] = {"T1_diff": round(float(np.mean([stats["T1"][l]["diff"] for l in band])), 4),
                       "T2_rho": round(float(np.mean([stats["T2"][l]["rho"] for l in band])), 3)}
json.dump({"sentences": show, "stats": stats, "n_sentences": len(SENT)}, open(sys.argv[2], "w"), separators=(",", ":"))
