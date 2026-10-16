"""exp114: pack a grokking run for the web page (site/grok.html). From RUN (default the fine-snapshot run, which saves
embeddings, readout and residuals every 50 steps). Everything is read with the weights of the same snapshot.
  keys     the clock frequencies: the readout's strongest frequencies whose angle code is clean at the end
           (mean resultant length > 0.8: the residual at "=" sits at angle k*(a+b) on that plane, up to one rotation)
  clocks   snapshots x K x 113 x 2  each number's position on frequency k's plane of the embedding, rotated so that
                                     number n belongs at angle 2*pi*k*n/113, radius / that snapshot's median
  thought  snapshots x 64 x K x 2   the residual at "=" for 64 example pairs on frequency k's plane of the readout,
                                     rotated the same way (the answer a+b belongs at angle 2*pi*k*(a+b)/113)
  align    snapshots x K            how clean each clock is at that step (mean resultant length over all pairs, 0..1)
  prob     snapshots x 64           the model's probability for the right answer (its real logits at that step)
  pred     snapshots x 64           the answer it would give
  neurons  nsnap x 8 x 113 x 113    the 8 most wave-like neurons over all (a, b), every 200 steps
Writes grok.json + grok.bin (int8 / uint8 arrays in the order above).
    python runs/exp114/build_web.py [out_dir]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
RUN = Path(os.environ.get("RUN", os.path.expanduser("~/repos/research/neuron_tower_data/grok/fine/seed0")))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN / "web"; OUT.mkdir(parents=True, exist_ok=True)
P = 113; x = np.arange(P); NEU_EVERY, KEEP_NEU = 100, 200
L = json.load(open(RUN / "log.json")); log = L["log"]; tr = set(L["train_idx"])
E = np.load(RUN / "embed.npy").astype(np.float32); U = np.load(RUN / "unembed.npy").astype(np.float32)
R = np.load(RUN / "resid.npy").astype(np.float32); Nn = np.load(RUN / "neurons.npy").astype(np.float32)
assert len(E) == len(U) == len(R), (len(E), len(U), len(R))
A = np.repeat(x, P); B = np.tile(x, P); ANS = (A + B) % P
cosv = lambda k, n: np.cos(2 * np.pi * k * n / P); sinv = lambda k, n: np.sin(2 * np.pi * k * n / P)
def code(V, Wd, k, target):   # project rows of V on frequency k's (cos, sin) directions of Wd (D x P); align to target angles
    c, s = Wd @ cosv(k, x), Wd @ sinv(k, x); u, v = V @ c, V @ s; th = np.arctan2(v, u)
    best = None
    for sign in (1, -1):                                            # the plane can come out mirrored
        d = np.exp(1j * (sign * th - 2 * np.pi * k * target / P)); m = d.mean(); rl = abs(m)
        if best is None or rl > best[0]: best = (rl, sign, np.angle(m))
    rl, sign, off = best; r = np.hypot(u, v); ang = sign * th - off
    return np.stack([np.cos(ang), np.sin(ang)], 1) * (r / (np.median(r) + 1e-9))[:, None], rl, (sign, off, np.median(r), c, s)
pw = np.array([np.linalg.norm(U[-1] @ cosv(k, x)) ** 2 + np.linalg.norm(U[-1] @ sinv(k, x)) ** 2 for k in range(1, P // 2 + 1)])
keys = []
for k in np.argsort(-pw)[:8] + 1:
    _, rl, _ = code(R[-1], U[-1], int(k), ANS)
    if rl > .8: keys.append(int(k))
    if len(keys) == 3: break
K = len(keys); print("clock frequencies", keys, flush=True)
rng = np.random.default_rng(0)
pairs = [int(i) for i in [i for i in rng.permutation(P * P) if int(i) not in tr][:56] + [i for i in rng.permutation(P * P) if int(i) in tr][:8]]
T = len(E); clocks = np.zeros((T, K, P, 2), np.float32); thought = np.zeros((T, len(pairs), K, 2), np.float32)
align = np.zeros((T, K), np.float32); prob = np.zeros((T, len(pairs)), np.float32); pred = np.zeros((T, len(pairs)), np.uint8)
for t in range(T):
    for j, k in enumerate(keys):
        clocks[t, j], _, _ = code(E[t], E[t].T, k, x)
        full, rl, (sign, off, med, c, s) = code(R[t], U[t], k, ANS); align[t, j] = rl; thought[t, :, j] = full[pairs]
    lg = R[t][pairs] @ U[t]; pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    prob[t] = pr[np.arange(len(pairs)), ANS[pairs]]; pred[t] = lg.argmax(1)
fin = Nn[-1]; F2 = np.abs(np.fft.fft2(fin.T.reshape(-1, P, P))) ** 2; F2[:, 0, 0] = 0
peak = F2.reshape(len(F2), -1).max(1) / (F2.reshape(len(F2), -1).sum(1) + 1e-9); pick = [int(i) for i in np.argsort(-(peak * (fin.std(0) > 1e-3)))[:8]]
sel = list(range(0, len(Nn), KEEP_NEU // NEU_EVERY)); neu = Nn[sel][:, :, pick].transpose(0, 2, 1).reshape(len(sel), 8, P, P)
neu = neu / (np.abs(neu).max(axis=(2, 3), keepdims=True) + 1e-9)
clip = lambda a, s=40: np.clip(np.round(a * s), -127, 127).astype(np.int8)       # positions: radius 1 -> 40
meta = {"p": P, "keys": keys, "log": log, "steps": [r["step"] for r in log][:T], "steps_neu": [i * NEU_EVERY for i in sel], "neurons": pick,
        "pairs": [{"a": i // P, "b": i % P, "ans": (i // P + i % P) % P, "seen": i in tr} for i in pairs],
        "shapes": {"clocks": [T, K, P, 2], "thought": [T, len(pairs), K, 2], "align": [T, K], "prob": [T, len(pairs)], "pred": [T, len(pairs)], "neurons": list(neu.shape)},
        "scale": {"clocks": 40, "thought": 40, "align": 127, "prob": 127, "neurons": 127}}
json.dump(meta, open(OUT / "grok.json", "w"), separators=(",", ":"))
with open(OUT / "grok.bin", "wb") as f:
    f.write(clip(clocks).tobytes()); f.write(clip(thought).tobytes()); f.write(clip(align, 127).tobytes()); f.write(clip(prob, 127).tobytes())
    f.write(pred.tobytes()); f.write(clip(neu, 127).tobytes())
print("align at end", align[-1].round(2), "| right answers at end", float((pred[-1] == np.array([p["ans"] for p in meta["pairs"]])).mean()),
      "| neurons", pick, (OUT / "grok.bin").stat().st_size // 1024, "KB")
for t in (0, 60, 80, 90, 100, 120, T - 1): print(meta["steps"][t], "align", align[t].round(2), "test acc", log[t]["test_acc"])
