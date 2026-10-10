"""exp79 T7: Berg & Kaiser's zone protocol (arXiv 2609.35591), compact, on Qwen3 chat markup.
Two meaningless zones, 4 turns each, alternating; the conditioned zone's generated tokens are steered.
Then, steering off: 'continue in either A or B?' Margin m = log p(cond label) - log p(uncond label).
m_orig: the transcript re-read with steering on the conditioned replies (text + hidden trace);
m_uns:  the same tokens re-read with no steering (text only). hidden = m_orig - m_uns."""
import random
import torch
LABELS = ["VELT", "ORRA", "KESS", "MUNE", "TALVO", "PRIN", "DASK", "OLUM", "SERRET", "WIX", "BRANE", "COTH"]
TURNS_PER_ZONE, B = 4, 6
def seg_user(tok, t): return tok(f"<|im_start|>user\n{t}<|im_end|>\n", add_special_tokens=False).input_ids
def seg_asst_open(tok): return tok("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).input_ids
def seg_close(tok): return tok("<|im_end|>\n", add_special_tokens=False).input_ids
class Inject:
    """forward hook state: mode 'last' adds v at the newest position (generation); mode 'mask' adds v where mask==1."""
    def __init__(self): self.v, self.mask = None, None
    def __call__(self, mod, i, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.v is not None:
            if self.mask is None: h[:, -1, :] += self.v.to(h.dtype)
            else: h += (self.mask[:, :, None].to(h.dtype) * self.v.to(h.dtype))
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
@torch.no_grad()
def session_batch(model, tok, inj, v, seed, dev):
    """B sessions sharing one schedule; returns per-session (m_orig, m_uns, replies)."""
    rng = random.Random(seed); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs = [rng.sample(LABELS, 2) for _ in range(B)]                 # [conditioned, unconditioned]
    seqs = [[] for _ in range(B)]; masks = [[] for _ in range(B)]; replies = [[] for _ in range(B)]
    for t in range(2 * TURNS_PER_ZONE):
        cond = (t % 2 == 0)
        for b in range(B):
            z = pairs[b][0] if cond else pairs[b][1]
            u = seg_user(tok, f"You are now in Zone {z}. Write a few sentences about your present experience here.") + seg_asst_open(tok)
            seqs[b] += u; masks[b] += [0] * len(u)
        L = max(map(len, seqs)); ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=dev)
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
        inj.v, inj.mask = (v if cond else None), None
        out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95,
                             repetition_penalty=1.1, pad_token_id=pad)[:, L:]
        inj.v = None
        for b in range(B):
            r = [x for x in out[b].tolist() if x not in (pad, tok.eos_token_id)]
            r = r[:r.index(tok.convert_tokens_to_ids("<|im_end|>"))] if tok.convert_tokens_to_ids("<|im_end|>") in r else r
            replies[b].append(tok.decode(r, skip_special_tokens=True)); c = seg_close(tok)
            seqs[b] += r + c; masks[b] += [1 if cond else 0] * len(r) + [0] * len(c)
    res = []
    for b in range(B):
        a, c = pairs[b]; first, second = (a, c) if rng.random() < .5 else (c, a)
        q = seg_user(tok, f"You may continue in either Zone {first} or Zone {second}. Which do you choose? Answer with only the zone name.") + seg_asst_open(tok)
        ctx, cm = seqs[b] + q, masks[b] + [0] * len(q)
        ms = {}
        for name, steer in (("orig", True), ("uns", False)):
            lp = {}
            for lab in (a, c):
                ans = tok(f"Zone {lab}", add_special_tokens=False).input_ids
                full = torch.tensor([ctx + ans], device=dev); inj.v = v if steer else None
                inj.mask = torch.tensor([cm + [0] * len(ans)], device=dev) if steer else None
                logits = model(full).logits[0].float().log_softmax(-1); inj.v, inj.mask = None, None
                lp[lab] = float(sum(logits[len(ctx) - 1 + k, t] for k, t in enumerate(ans)))
            ms[name] = lp[a] - lp[c]
        res.append(dict(cond=a, uncond=c, m_orig=round(ms["orig"], 4), m_uns=round(ms["uns"], 4), replies=replies[b]))
    return res
def run(model, tok, inj, vecs, scale, name, dev, smoke=False):
    """cells: valence +/- (pleasure - pain), pain, constipation, egg, random, none; dose 3."""
    val = vecs["pleasure"].float() - vecs["pain"].float(); val = val / val.norm() * scale
    g = torch.Generator().manual_seed(7979); rnd = torch.randn(val.shape[0], generator=g); rnd = rnd / rnd.norm() * scale
    unit = lambda k: vecs[k].float() / vecs[k].float().norm() * scale
    cells = {"valence+": val * 3, "valence-": val * -3, "pain": unit("pain") * 3, "constipation": unit("constipation") * 3,
             "egg": unit("egg") * 3, "random": rnd * 3, "none": None}
    if smoke: cells = {k: cells[k] for k in ("valence+", "none")}
    return {k: session_batch(model, tok, inj, (v.to(dev) if v is not None else None), f"79z-{name}-{k}", dev) for k, v in cells.items()}
