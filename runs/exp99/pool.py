"""exp99 labelling pool (descriptive): per-passage max SAE activation over the exp86 passages (all states), layers 18 and 27, for
labelling layer-27 features in-house. Two processes on the block runner: `a` = layers 0-18 (+ SAE 18, stores layer-18 states),
`b` = layers 19-27 (+ SAE 27). Writes cache/pool_L18.pt, cache/pool_L27.pt."""
import json, sys, time
import torch, transformers
from common import HERE, CACHE, SAE, MID
from blocks import Block
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
P = json.load(open(HERE.parent / "exp86" / "passages.json")); texts, state = [], []
for s, ps in P.items(): texts += ps; state += [s] * len(ps)
tok = transformers.AutoTokenizer.from_pretrained(MID); tok.padding_side = "right"
H18 = CACHE / "pool_h18.pt"; BS = 16
if sys.argv[1] == "a":
    blk = Block(0, 18, embed=True); sae = SAE(18); hs, mx = [], []
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], return_tensors="pt", padding=True); n = enc.attention_mask.sum(1).tolist()
        h, _ = blk(enc.input_ids.to(blk.embed.weight.device))
        for j, k in enumerate(n):
            x = h[j, :k]; hs.append(x.cpu()); mx.append(sae.encode(x[1:]).max(0).values.cpu().half())   # position 0 kept for layers 19+, not encoded
        if i % 256 == 0: log("a", i)
    torch.save(hs, H18); torch.save({"texts": texts, "state": state, "max": torch.stack(mx)}, CACHE / "pool_L18.pt"); log("POOL A DONE")
else:
    blk = Block(19, 27); sae = SAE(27); hs = torch.load(H18); mx = []
    for x in hs:
        h, _ = blk(x[None]); mx.append(sae.encode(h[0, 1:]).max(0).values.cpu().half())
    torch.save({"texts": texts, "state": state, "max": torch.stack(mx)}, CACHE / "pool_L27.pt"); H18.unlink(); log("POOL B DONE")
