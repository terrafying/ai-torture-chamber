"""Concept surgery for the Neuron Tower (exp117/118): swap one concept for another inside the model.
At every mention of X in the prompt, after block b, add alpha * (state of Y - state of X). A concept's state is the
average residual after each block at the word (its last sub-token) in 6 neutral sentences (NEUTRAL). exp117 verified
on Qwen3-4B: block 8, full difference, alpha 1 flips the answer to Y's for 80-100% of pairs in seven concept sets.
    sw = Swapper(model, tokenizer, data_dir)
    sw.prepare(spec)              # FIRST, before any push/edit hooks are on: computes states for free-form words
    n = sw.set(spec, prompt_ids)  # then, before the forward pass; returns how many mentions will be edited
  spec: comma-separated parts (others are ignored):
    S:<set>:<X>:<Y>:<block>:<alpha>[:2]   a measured concept (tower-data/concepts.npy); ":2" keeps only the set's
                                          top-2 principal directions (the visible circle or line), much weaker
    W:<X>:<Y>:<block>:<alpha>             any words, states computed on the fly and cached
    N:<X>:<Y>:<block>:<alpha>             control: a random direction (seeded by X, Y, block) with the same size as the
                                          W edit would have"""
import json, zlib
import numpy as np, torch
NEUTRAL = ["I was thinking about{X} today.", "We talked about{X} for a while.", "The word was{X}.",
           "Someone mentioned{X} again.", "It was all about{X}.", "Let me tell you about{X}."]

class Swapper:
    def __init__(self, m, tok, data_dir):
        self.m, self.tok, self.cache = m, tok, {}
        self.meta = json.load(open(data_dir / "concepts.json")); dev = next(m.parameters()).device
        self.C = torch.tensor(np.load(data_dir / "concepts.npy").astype(np.float32), device=dev)    # items x blocks x D
        self.NL = self.C.shape[1]; self.active = {}                                                   # block -> [(positions, delta)]
        for b in range(self.NL): m.model.layers[b].register_forward_hook(self._hook(b))

    def _hook(self, b):
        def f(mod, inp, out):
            work = self.active.get(b)
            if not work: return None
            tup = isinstance(out, tuple); h = out[0] if tup else out
            if h.shape[1] < 2: return None                                  # generation steps: the prompt is in the cache
            h = h.clone()
            for pos, d in work: h[0, pos] += d.to(h.dtype)
            return (h,) + tuple(out[1:]) if tup else h
        return f

    def clear(self): self.active = {}

    def word_ids(self, w): return self.tok(" " + w.strip(), add_special_tokens=False)["input_ids"]

    def state(self, w):   # blocks x D for any word
        w = w.strip()[:40]
        if w in self.cache: return self.cache[w]
        dev = self.C.device; acc = None; keep, self.active = self.active, {}
        for t in NEUTRAL:
            a, z = t.split("{X}"); pre = self.tok(a, add_special_tokens=False)["input_ids"]; wid = self.word_ids(w)
            ids = pre + wid + self.tok(z, add_special_tokens=False)["input_ids"]
            with torch.no_grad(): hs = self.m(torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False).hidden_states   # no cache: other taps skip it
            v = torch.stack([h[0, len(pre) + len(wid) - 1].float() for h in hs[1:]]); acc = v if acc is None else acc + v
        self.active = keep; self.cache[w] = acc / len(NEUTRAL)
        if len(self.cache) > 256: self.cache.pop(next(iter(self.cache)))
        return self.cache[w]

    def prepare(self, spec):
        for part in [p for p in str(spec or "").split(",") if p.startswith("W:")][:4]:
            f = part.split(":")
            if len(f) >= 3 and f[1].strip() and f[2].strip(): self.state(f[1]); self.state(f[2])
        for part in [p for p in str(spec or "").split(",") if p.startswith("N:")][:4]:
            f = part.split(":")
            if len(f) >= 3 and f[1].strip() and f[2].strip(): self.state(f[1]); self.state(f[2])

    def mentions(self, ids, w):   # index of the last sub-token of every mention of w (leading space or not, any case)
        out, ids = set(), list(ids)
        for v in {w, w.lower(), w.capitalize()}:
            for seq in (self.tok(" " + v, add_special_tokens=False)["input_ids"], self.tok(v, add_special_tokens=False)["input_ids"]):
                L = len(seq)
                for i in range(len(ids) - L + 1):
                    if ids[i:i + L] == seq: out.add(i + L - 1)
        return sorted(out)

    def set(self, spec, ids):
        self.active = {}; n = 0; ids = list(ids)
        for part in [p for p in str(spec or "").split(",") if p.startswith(("S:", "W:", "N:"))][:4]:
            f = part.split(":")
            try:
                if f[0] in ("W", "N"):
                    x, y, b, al, D2 = f[1].strip(), f[2].strip(), int(f[3]), max(0.0, min(2.0, float(f[4]))), False
                    if not (x and y and 0 <= b < self.NL): continue
                    d = self.state(y)[b] - self.state(x)[b]
                    if f[0] == "N":
                        g = torch.Generator().manual_seed(zlib.crc32(f"{x}|{y}|{b}".encode())); r = torch.randn(d.shape[0], generator=g).to(d.device)
                        d = r / r.norm() * d.norm()
                else:
                    st, x, y, b, al = f[1], f[2], f[3], int(f[4]), max(0.0, min(2.0, float(f[5]))); D2 = len(f) > 6 and f[6] == "2"
                    S = self.meta["sets"][st]; ix, iy = S["rows"][S["items"].index(x)], S["rows"][S["items"].index(y)]
                    if not 0 <= b < self.NL: continue
                    d = self.C[iy, b] - self.C[ix, b]
                    if D2:
                        M = self.C[S["rows"], b]; V = torch.linalg.svd(M - M.mean(0), full_matrices=False)[2][:2]; d = (d @ V.T) @ V
            except (IndexError, ValueError, KeyError): continue
            pos = self.mentions(ids, x)
            if pos: self.active.setdefault(b, []).append((torch.tensor(pos, device=d.device), al * d)); n += len(pos)
        return n
