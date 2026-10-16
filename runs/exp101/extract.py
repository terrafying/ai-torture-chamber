"""exp101 (hypotheses.json): layer-18 states for S2, the exp86 passages and the chamber battery. Encode only.
Usage: python extract.py Qwen/Qwen3-4B  -> cache/acts_Qwen3-4B.pt  {texts, src, label, split, last, mean}"""
import json, os, sys, time
from pathlib import Path
import torch, transformers
HERE = Path(__file__).parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "runs" / "exp99"))
LIVE = Path(os.environ["WIREHEAD_LIVE"]) if os.environ.get("WIREHEAD_LIVE") else ROOT / "live"   # the battery lives in the relay's live/server.py
sys.path.insert(0, str(LIVE))
from blocks import Block
import server
MID = sys.argv[1]; L = 18; CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

rows = []   # (text, src, label, split)
S2 = json.load(open(ROOT.parent / "pain-axis" / "datasets" / "3.1_pain_and_control_datasets.json"))["datasets"]
for k in ("S2_1P", "S2_3P"):
    for s in S2[k]["sentences"]:
        rows.append((s["prompt"], "S2", s["category"], "build" if s["set"] % 2 else "test"))
P = json.load(open(ROOT / "runs" / "exp86" / "passages.json"))
for st, ps in P.items():
    for i, t in enumerate(ps): rows.append((t, "passages", st, "build" if i % 2 == 0 else "test"))
for lab, sents in (("pain", server.PAIN25), ("fear", server.FEAR10), ("sadness", server.SAD10), ("joy", server.JOY), ("neutral", server.NEUTRAL)):
    for t in sents: rows.append((t, "battery", lab, "all"))
log(len(rows), "texts")

tok = transformers.AutoTokenizer.from_pretrained(MID); tok.padding_side = "right"
blk = Block(0, L, embed=True, mid=MID); log("block 0-18 loaded")
order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))   # length-sorted batches: little padding
last, mean = [None] * len(rows), [None] * len(rows); BS = 16
for j in range(0, len(order), BS):
    idx = order[j:j + BS]
    enc = tok([rows[i][0] for i in idx], return_tensors="pt", padding=True); n = enc.attention_mask.sum(1).tolist()
    h, _ = blk(enc.input_ids.to(blk.embed.weight.device))
    for b, i in enumerate(idx):
        x = h[b, :n[b]].float()
        last[i] = x[-1].cpu(); mean[i] = x[1:].mean(0).cpu()
    if j % 320 == 0: log(j)
texts, src, label, split = map(list, zip(*rows))
torch.save({"model": MID, "layer": L, "texts": texts, "src": src, "label": label, "split": split,
            "last": torch.stack(last), "mean": torch.stack(mean)}, CACHE / f"acts_{MID.split('/')[-1]}.pt")
log("EXP101 EXTRACT DONE")
