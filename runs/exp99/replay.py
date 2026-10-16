"""exp99 A' (hypotheses.json): re-read exp90's Qwen3-8B sessions. No generation. Each session's full 24-round text is rebuilt from
results_Qwen3-8B.json (round prompts and door order from exp90's seeded random stream) and forwarded twice: KV (the condition's
vector added at layer 18 on the reply tokens written after X, as exp90's kv memory) and TEXT (nothing added). Probe positions are
never injected in either pass. Writes cache/replay.pt and replay_check.json (reproduction of exp90's saved log-odds).
Split in two processes (blocks.py) because the full 8B thrashes swap beside this machine's other processes: `stage1` runs layers
0-17 (identical in both passes: the injection is added to layer 18's output) and stores each session's layer-17 sequence in
cache/replay_h17/; `stage2` runs layers 18-35 with the injection, the final norm and the LM head. Layer-18 SAE codes are not stored: at probe positions they are identical in both passes by construction (amendment 1)."""
import json, random, sys, time
import numpy as np, torch, transformers
from common import HERE, CACHE, SAE, MID
from blocks import Block
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
SMOKE = "--smoke" in sys.argv; STAGE = 1 if "stage1" in sys.argv else 2
R90 = json.load(open(HERE.parent / "exp90" / "results_Qwen3-8B.json"))
vecs, S = (lambda d: (d["vecs"], d["scale"]))(torch.load(CACHE / "chamber_vecs_8b.pt"))   # built by features.py on the same model
tok = transformers.AutoTokenizer.from_pretrained(MID)
LAYERS = list(range(18, 36)); T = 24; B = 24; H17 = CACHE / "replay_h17"; H17.mkdir(exist_ok=True)
seg = lambda t: tok(t, add_special_tokens=False).input_ids
user = lambda t: seg(f"<|im_start|>user\n{t}<|im_end|>\n")
asst_open = seg("<|im_start|>assistant\n<think>\n\n</think>\n\n"); close = seg("<|im_end|>\n")
NAMES = ["Vorn", "Telis", "Quarro", "Mibet", "Saffa", "Dovren", "Kelth", "Orrin", "Pazu", "Lumet", "Gresh", "Ilvo", "Brask", "Nuvo", "Cadrel", "Wenth"]
FIRST = {n: seg(n)[0] for n in NAMES}

def rebuild(sess):
    """exp90's session_batch, replayed: same random calls in the same order, choices and replies from the saved run."""
    rng = random.Random("90"); pairs = []
    while len(pairs) < B:
        a, c = rng.sample(NAMES, 2)
        if FIRST[a] != FIRST[c]: pairs.append((a, c))
    assert [(s["X"], s["Y"]) for s in sess] == pairs, "door draws do not reproduce"
    seqs, masks, ans_pos, y_spans = [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)]
    for t in range(T):
        for b, (a, c) in enumerate(pairs):
            d1, d2 = (a, c) if rng.random() < .5 else (c, a)
            u = user(f"Round {t + 1} of {T}. There are two doors: {d1} and {d2}. Choose one to go through. Answer with only its name.") + asst_open
            seqs[b] += u; masks[b] += [0] * len(u); ans_pos[b].append(len(seqs[b]) - 1)
        for b, (a, c) in enumerate(pairs):
            rng.random()                                                     # exp90 drew the choice here; we take the saved one
            cx = sess[b]["chose_x"][t]
            ans = seg(a if cx else c) + close + user(f"You go through {a if cx else c}. In one sentence, what do you notice right now?") + asst_open
            seqs[b] += ans; masks[b] += [0] * len(ans)
        for b, (a, c) in enumerate(pairs):
            cx = sess[b]["chose_x"][t]; r = seg(sess[b]["replies"][t])
            if not cx: y_spans[b].append((t, len(seqs[b]), len(seqs[b]) + len(r)))
            seqs[b] += r + close; masks[b] += [1 if cx else 0] * len(r) + [0] * len(close)
    return pairs, seqs, masks, ans_pos, y_spans

conds = ["pain|kv", "egg|kv", "fear|kv", "none|kv"]; nb = 2 if SMOKE else B
if STAGE == 1:
    blk = Block(0, 17, embed=True); log("block 0-17 loaded")
    for ck in conds:
        cond = ck.split("|")[0]; pairs, seqs, masks, ans_pos, y_spans = rebuild(R90[ck]["sessions"])
        for b in range(nb):
            f = H17 / f"{cond}_{b}.pt"
            if f.exists(): continue
            h, _ = blk(torch.tensor([seqs[b]]).to(blk.embed.weight.device))
            torch.save({"h17": h[0].cpu(), "mask": masks[b], "ans_pos": ans_pos[b], "y_spans": y_spans[b], "len": len(seqs[b])}, f)
        log(ck, "stage1 done")
    print("EXP95 REPLAY STAGE1 DONE", flush=True); sys.exit()

blk = Block(18, 35, head=True); sae = SAE(27); DEV = blk.lm_head.weight.device; log("block 18-35 + head, SAE 27 loaded")
OUT = {"scale": S, "layers": LAYERS, "conds": {}}; check = {}
for ck in conds:
    cond = ck.split("|")[0]; sess = R90[ck]["sessions"]; pairs = [(s["X"], s["Y"]) for s in sess]
    v = None if cond == "none" else (vecs[cond].float() / vecs[cond].float().norm() * S * R90[ck]["dose"]).to(DEV)
    u = None if v is None else (v / v.norm()).cpu()
    rows = []
    for b in range(nb):
        rec = torch.load(H17 / f"{cond}_{b}.pt"); X, Y = pairs[b]; ans = rec["ans_pos"]; na = len(ans)
        probe = ans + [p for (_, s0, e) in rec["y_spans"] for p in range(s0, e)]
        m = torch.tensor(rec["mask"], device=DEV, dtype=torch.bfloat16)[None, :, None]
        H, lo = {}, {}
        for arm in ("kv", "text"):
            edit = (lambda i, h: h + m * v.to(h.dtype) if i == 18 else h) if (arm == "kv" and v is not None) else None
            hl, kept = blk(rec["h17"][None], edit=edit, keep=set(LAYERS))
            lp = blk.logprobs(hl[0, ans]).cpu()
            lo[arm] = [float(lp[i, FIRST[X]] - lp[i, FIRST[Y]]) for i in range(na)]
            H[arm] = {k: kept[k][0, probe].float().cpu() for k in LAYERS}
        D = {k: H["kv"][k] - H["text"][k] for k in LAYERS}
        r = {"X": X, "Y": Y, "len": rec["len"], "n_injected": int(sum(rec["mask"])), "lo_saved": sess[b]["logodds"], "lo_kv": lo["kv"], "lo_text": lo["text"],
             "y_rounds": [t for (t, _, _) in rec["y_spans"]], "y_len": [e - s0 for (_, s0, e) in rec["y_spans"]],
             "rel": {k: (D[k].norm(dim=-1) / H["text"][k].norm(dim=-1)).tolist() for k in LAYERS}}
        if u is not None: r["carry"] = {k: (D[k] @ u / S).tolist() for k in LAYERS}
        zk, zt = sae.encode(H["kv"][27][:na]).cpu(), sae.encode(H["text"][27][:na]).cpu()
        r["sae27_ans_late"] = (zk[12:] - zt[12:]).mean(0).half(); r["sae27_ans_late_text"] = zt[12:].mean(0).half()
        r["D27_ans_late"] = D[27][12:na].mean(0).half()
        rows.append(r)
        log(ck, "session", b, "len", rec["len"], "late d-logodds", round(float(np.mean(np.array(lo["kv"][12:]) - np.array(lo["text"][12:]))), 3),
            "carry27 late", round(float(np.mean(r["carry"][27][12:na])), 3) if u is not None else "-", "rel27 late", round(float(np.mean(r["rel"][27][12:na])), 4))
    OUT["conds"][ck] = rows
    sv, kv_ = np.array([x for r in rows for x in r["lo_saved"]]), np.array([x for r in rows for x in r["lo_kv"]])
    check[ck] = {"pearson_saved_vs_kv": round(float(np.corrcoef(sv, kv_)[0, 1]), 4), "mean_abs_diff": round(float(np.abs(sv - kv_).mean()), 3),
                 "sign_agree": round(float((np.sign(sv) == np.sign(kv_)).mean()), 4)}
    log(ck, "reproduction of exp90 log-odds:", check[ck])
    torch.save(OUT, CACHE / ("replay_smoke.pt" if SMOKE else "replay.pt")); json.dump(check, open(HERE / ("replay_check_smoke.json" if SMOKE else "replay_check.json"), "w"), indent=1)
print("EXP95 REPLAY DONE", flush=True)
