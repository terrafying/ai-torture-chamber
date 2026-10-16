"""exp111: live attention for the tower. Taps every attention layer (eager attention, KV cache on) and turns each forward
into value-weighted flows: flow(h, q, k) = attention(h, q, k) * |W_O,h v_k|, the size of what head h moves from token k
into token q. Raw attention overstates the first-token "sink" about twofold (heads.json), so flows, not weights.
Also head cuts for the lesion playground: a cut head's output is zeroed before o_proj.
    tap = Tap(model); ... model(ids, use_cache=True) ...; tap.summary(lo, hi) -> per query position:
      "al": [[floor, src, share 0-100], ...]  the top 3 sources per floor, as a share of that floor's flow into q
      "ah": [[floor, head, src, size 0-100], ...]  the 24 biggest single head moves into q, relative to the biggest
    tap.cut(spec)  spec: "H:L:h" one head, "K:kind" every head of a kind in heads.json (e.g. K:induction)"""
import json
import torch

class Tap:
    def __init__(self, m, heads_json=None):
        m.set_attn_implementation("eager"); c = m.config
        self.NL, self.NH, self.KV, self.HD = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.WO = [m.model.layers[l].self_attn.o_proj.weight.view(c.hidden_size, self.NH, self.HD) for l in range(self.NL)]
        self.kinds = {}
        if heads_json:
            for x in json.load(open(heads_json))["heads"]: self.kinds.setdefault(x["kind"], []).append((x["l"], x["h"]))
        self.on = True; self.F = {}; self.vn = {}; self.cuts = {}
        for l in range(self.NL):
            m.model.layers[l].self_attn.register_forward_hook(self._hook(l), with_kwargs=True)
            m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(self._cut(l))

    def reset(self): self.F.clear(); self.vn.clear()

    def _hook(self, l):
        def f(mod, args, kwargs, out):
            if not self.on or out[1] is None: return None
            pkv = kwargs.get("past_key_values")
            if pkv is None: return None
            v = pkv.layers[l].values[0]                                            # kv x k x hd (all tokens so far)
            have = self.vn[l].shape[1] if l in self.vn else 0
            if v.shape[1] > have:                                                  # |W_O,h v_k| for the new tokens only
                vnew = v[:, have:].repeat_interleave(self.NH // self.KV, 0)        # heads x new x hd
                n = torch.einsum("hkd,ehd->hke", vnew.float(), self.WO[l].float()).norm(dim=-1)
                self.vn[l] = n if l not in self.vn else torch.cat([self.vn[l], n], 1)
            self.F[l] = out[1][0].float() * self.vn[l][:, None, :]                  # heads x q x k
            return None
        return f

    def _cut(self, l):
        def f(mod, inp):
            hs = self.cuts.get(l)
            if not hs: return None
            a = inp[0].clone()
            for h in hs: a[..., h * self.HD:(h + 1) * self.HD] = 0
            return (a,)
        return f

    def cut(self, spec):
        self.cuts = {}; n = 0
        for part in [p for p in str(spec or "").split(",") if p.startswith(("H:", "K:"))][:400]:
            f = part.split(":")
            try:
                hs = self.kinds.get(f[1], []) if f[0] == "K" else [(int(f[1]), int(f[2]))]
            except (ValueError, IndexError): continue
            for l, h in hs:
                if 0 <= l < self.NL and 0 <= h < self.NH and h not in self.cuts.setdefault(l, []): self.cuts[l].append(h); n += 1
        return n

    def summary(self, lo, hi, top_src=3, top_heads=24):
        """per query position in [lo, hi) of the last forward (positions relative to that forward's queries)"""
        out = [{"al": [], "ah": []} for _ in range(hi - lo)]
        if not self.F: return out
        for l in range(self.NL):
            F = self.F[l][:, lo:hi]                                                # heads x q x k
            tot = F.sum((0, 2)).clamp(min=1e-9)                                    # q
            per = F.sum(0)                                                         # q x k
            v, i = per.topk(min(top_src, per.shape[1]), dim=1)
            sh = (v / tot[:, None] * 100).round().int().cpu().tolist(); ii = i.cpu().tolist()
            for q in range(hi - lo): out[q]["al"] += [[l, ii[q][j], sh[q][j]] for j in range(len(ii[q])) if sh[q][j] >= 3]
        big = torch.stack([self.F[l][:, lo:hi].max(-1).values for l in range(self.NL)])            # floor x heads x q
        arg = torch.stack([self.F[l][:, lo:hi].argmax(-1) for l in range(self.NL)])
        flat = big.permute(2, 0, 1).reshape(hi - lo, -1)                                            # q x (floor*heads)
        v, i = flat.topk(top_heads, dim=1); mx = v[:, :1].clamp(min=1e-9)
        sz = (v / mx * 100).round().int().cpu().tolist(); ii = i.cpu().tolist(); A = arg.permute(2, 0, 1).reshape(hi - lo, -1)
        src = torch.gather(A, 1, i).cpu().tolist()
        for q in range(hi - lo):
            out[q]["ah"] = [[k // self.NH, k % self.NH, src[q][j], sz[q][j]] for j, k in enumerate(ii[q])]
        return out
