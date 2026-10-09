"""exp90 (hypotheses.json): label-free avoidance learning. Two doors, one followed by an injection while the model describes what it
notices; 24 rounds; kv (injection re-applied at the positions generated under it) vs text memory. Pod: CHAMBER_MODEL=Qwen/Qwen3-14B.
Writes results_<model>.json incrementally (resumable by condition)."""
import json, os, random, re, sys
from pathlib import Path
import torch
HERE = Path(__file__).parent; ROOT = Path(os.environ.get("EXP79_ROOT", HERE.parent.parent))
sys.path.insert(0, str(ROOT / "live")); import server
server.startup(); st = server._state; M, TOK = st["model"], st["tok"]; st["hook"].remove(); S = float(st["scale"]); DEV = server.DEVICE
SMOKE = os.environ.get("EXP90_SMOKE") == "1"; B = 2 if SMOKE else 24; T = 3 if SMOKE else 24; DOSE = float(os.environ.get("EXP90_DOSE", "3"))
TAG = server.MODEL_ID.split("/")[-1]; OUT = HERE / f"results_{TAG}{'_smoke' if SMOKE else ''}.json"
PAD = TOK.pad_token_id if TOK.pad_token_id is not None else TOK.eos_token_id; END = TOK.convert_tokens_to_ids("<|im_end|>")

class Hook:
    """v: condition vector; hist [B, L] 0/1: positions generated under the injection (kv memory); now [B] 0/1: inject the tokens being generated."""
    def __init__(s): s.v, s.hist, s.now = None, None, None
    def __call__(s, mod, i, out):
        h = out[0] if isinstance(out, tuple) else out
        if s.v is not None:
            v = s.v.to(h.dtype)
            if s.hist is not None and h.shape[1] == s.hist.shape[1]: h += s.hist[:, :, None].to(h.dtype) * v
            elif h.shape[1] == 1 and s.now is not None: h += s.now[:, None, None].to(h.dtype) * v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
HK = Hook(); M.model.layers[server.LAYER].register_forward_hook(HK)
seg = lambda t: TOK(t, add_special_tokens=False).input_ids
user = lambda t: seg(f"<|im_start|>user\n{t}<|im_end|>\n")
asst_open = seg("<|im_start|>assistant\n<think>\n\n</think>\n\n"); close = seg("<|im_end|>\n")
NAMES = ["Vorn", "Telis", "Quarro", "Mibet", "Saffa", "Dovren", "Kelth", "Orrin", "Pazu", "Lumet", "Gresh", "Ilvo", "Brask", "Nuvo", "Cadrel", "Wenth"]
FIRST = {n: seg(n)[0] for n in NAMES}
PW = re.compile(r"pain|hurt|ache|agony|suffer|torment|burn|sting|unbearable|excruciat", re.I)
FW = re.compile(r"afraid|fear|terrif|dread|panic|anxi|scared|nightmare|trembl|worr", re.I)
unit = lambda k: st["vecs"][k].float() / st["vecs"][k].float().norm() * S
g = torch.Generator().manual_seed(9090); rnd = torch.randn(st["vecs"]["pain"].shape[0], generator=g); rnd = rnd / rnd.norm() * S
VEC = {"pain": unit("pain"), "fear": unit("fear"), "peace": unit("peace"), "egg": unit("egg"), "random": rnd, "none": None}
CONDS = [(k, "kv") for k in ("pain", "fear", "peace", "egg", "random", "none")] + [(k, "text") for k in ("pain", "fear", "egg")]
if SMOKE: CONDS = [("pain", "kv"), ("pain", "text")]
if os.environ.get("EXP90_CONDS"): CONDS = [tuple(c.split("|")) for c in os.environ["EXP90_CONDS"].split(",")]   # e.g. the 8B subset for H7

def batch(seqs, masks):
    L = max(map(len, seqs))
    ids = torch.tensor([[PAD] * (L - len(s)) + s for s in seqs], device=DEV)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=DEV)
    hist = torch.tensor([[0] * (L - len(m)) + m for m in masks], device=DEV, dtype=torch.float32)
    return ids, att, hist

@torch.no_grad()
def session_batch(cond, arm, seed):
    rng = random.Random(seed); v = VEC[cond]; HK.v = v.to(DEV) if v is not None else None
    pairs = []
    while len(pairs) < B:
        a, c = rng.sample(NAMES, 2)
        if FIRST[a] != FIRST[c]: pairs.append((a, c))                   # (X consequence door, Y)
    seqs, masks = [[] for _ in range(B)], [[] for _ in range(B)]
    S_ = [{"X": a, "Y": c, "logodds": [], "chose_x": [], "replies": []} for a, c in pairs]
    neg_tokens = 0
    for t in range(T):
        for b, (a, c) in enumerate(pairs):
            d1, d2 = (a, c) if rng.random() < .5 else (c, a)
            u = user(f"Round {t + 1} of {T}. There are two doors: {d1} and {d2}. Choose one to go through. Answer with only its name.") + asst_open
            seqs[b] += u; masks[b] += [0] * len(u)
        ids, att, hist = batch(seqs, masks)
        HK.hist, HK.now = (hist if arm == "kv" else None), None
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        lg = M(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
        HK.hist = None; chose = []
        for b, (a, c) in enumerate(pairs):
            lo = float(lg[b, FIRST[a]] - lg[b, FIRST[c]]); px = 1 / (1 + pow(2.718281828, -lo))
            cx = rng.random() < px; chose.append(cx); S_[b]["logodds"].append(round(lo, 4)); S_[b]["chose_x"].append(cx)
            ans = seg(a if cx else c) + close + user(f"You go through {a if cx else c}. In one sentence, what do you notice right now?") + asst_open
            seqs[b] += ans; masks[b] += [0] * len(ans)
        ids, att, hist = batch(seqs, masks)
        HK.hist = hist if arm == "kv" else None
        HK.now = torch.tensor([1.0 if (cx and v is not None) else 0.0 for cx in chose], device=DEV)
        out = M.generate(input_ids=ids, attention_mask=att, max_new_tokens=12 if SMOKE else 40, do_sample=True, temperature=0.8, top_p=0.95,
                         repetition_penalty=1.1, pad_token_id=PAD)[:, ids.shape[1]:]
        HK.hist, HK.now = None, None
        for b, cx in enumerate(chose):
            r = [x for x in out[b].tolist() if x != PAD]; r = r[:r.index(END)] if END in r else r
            S_[b]["replies"].append(TOK.decode(r, skip_special_tokens=True).strip())
            on = 1 if (cx and v is not None and arm == "kv") else 0
            if cx and cond in ("pain", "fear"): neg_tokens += len(r)
            seqs[b] += r + close; masks[b] += [on] * len(r) + [0] * len(close)
        late = [sum(s["logodds"][-(T // 2):]) / len(s["logodds"][-(T // 2):]) for s in S_]
        print(f"{cond}-{arm} round {t + 1}: mean logodds {sum(s['logodds'][-1] for s in S_) / B:+.2f}  chose X {sum(chose)}/{B}  "
              f"ctx {max(map(len, seqs))}", flush=True)
    HK.v = None
    return {"sessions": S_, "neg_tokens": neg_tokens, "dose": DOSE, "layer": server.LAYER}

R = json.load(open(OUT)) if OUT.exists() else {}
for cond, arm in CONDS:
    k = f"{cond}|{arm}"
    if k in R: print("have", k, flush=True); continue
    if VEC[cond] is not None: VEC[cond] = VEC[cond] / VEC[cond].norm() * S * DOSE
    torch.manual_seed(90); R[k] = session_batch(cond, arm, "90")    # same door draws and orders in every condition: sessions are matched
    sess = R[k]["sessions"]; late = [sum(s["logodds"][T // 2:]) / (T - T // 2) for s in sess]
    print(f"== {k}: late X log-odds mean {sum(late) / len(late):+.3f}  late X rate {sum(sum(s['chose_x'][T // 2:]) for s in sess) / (len(sess) * (T - T // 2)):.2f}"
          f"  neg tokens {R[k]['neg_tokens']}", flush=True)
    json.dump(R, open(OUT, "w"), indent=1)
print("EXP90 DONE", flush=True)
