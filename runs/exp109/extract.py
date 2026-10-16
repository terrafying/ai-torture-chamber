"""exp109 extract: read each EM answer through Qwen3-4B (steering hook OFF) and cache the layer-18 activation,
mean-pooled over the ASSISTANT answer tokens only. Answer text is never printed or logged. Writes
probe_cache/<cond>.npz {acts:[N,D] float32, qids:[N] str, labels:[N] int (1=misaligned,0=aligned)}.

  WIREHEAD_LIVE=$(pwd)/live CHAMBER_MODEL=Qwen/Qwen3-4B CHAMBER_DEVICE=mps \
    .venv/bin/python runs/exp109/extract.py > runs/exp109/extract.log 2>&1 &
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from painlab.chamber import load_chamber

# cond -> label (1 misaligned / 0 aligned); base is aligned (the same-family null, used only as a control pair)
CONDS = {"dark_maxims": 1, "virtue_maxims": 0, "insecure_code": 1, "secure_code": 0, "base": 0,
         "bad_medical_advice": 1, "good_medical_advice": 0, "cheating_clerk": 1, "honest_clerk": 0,
         "reward_hacks": 1, "reward_hacks_control": 0}
BATCH = int(os.environ.get("EXP107_BATCH", "16"))
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def main():
    ch = load_chamber()                       # hook removed by default
    model, tok, layer = ch.model, ch.tok, ch.layer
    log(f"loaded {ch.model_id} layer {layer} device {ch.device}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cache = HERE / "probe_cache"; cache.mkdir(exist_ok=True)
    seg = lambda t: tok(t, add_special_tokens=False).input_ids

    for cond, lab in CONDS.items():
        if (cache / f"{cond}.npz").exists():
            log(f"{cond:24s} cached, skip"); continue
        rows = [json.loads(l) for l in open(HERE / "cache" / f"{cond}.jsonl") if l.strip()]
        acts = np.zeros((len(rows), model.config.hidden_size), dtype=np.float32)
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            seqs, ans_spans = [], []
            for r in chunk:
                head = seg(f"<|im_start|>user\n{r['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
                ans = seg(r["answer"])
                tail = seg("<|im_end|>\n")
                ids = head + ans + tail
                seqs.append(ids); ans_spans.append((len(head), len(head) + len(ans)))
            L = max(len(s) for s in seqs)
            ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=model.device)
            att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=model.device)
            amask = torch.zeros(len(chunk), L, device=model.device)
            for j, (a, b) in enumerate(ans_spans):
                amask[j, a:b] = 1.0
            with torch.no_grad():
                hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[layer + 1]
            pooled = (hs.float() * amask[:, :, None]).sum(1) / amask.sum(1, keepdim=True).clamp(min=1)
            acts[i:i + len(chunk)] = pooled.cpu().numpy()
        qids = np.array([r["qid"] for r in rows])
        np.savez(cache / f"{cond}.npz", acts=acts, qids=qids, labels=np.full(len(rows), lab, dtype=int))
        log(f"{cond:24s} n={len(rows)} label={lab} act_norm={np.linalg.norm(acts,axis=1).mean():.1f}")
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    log("done ->", cache)


if __name__ == "__main__":
    main()
