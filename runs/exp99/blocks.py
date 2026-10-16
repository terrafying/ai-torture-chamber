"""Qwen3-8B in layer blocks, loaded straight from the safetensors shards: only the requested layers ever reach memory.
The full 8B (16 GB bf16) thrashes swap beside this machine's other processes; a 9-18 layer block does not. Hidden states
cross block boundaries through disk. Single unpadded or right-padded sequences, causal attention only (attention_mask=None)."""
import glob, json, os
import torch, transformers
from safetensors import safe_open
from transformers.models.qwen3 import modeling_qwen3 as Q
from common import DEV, MID

def _snap(mid):
    snap = sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{mid.replace('/', '--')}/snapshots/*/")))[-1]
    return snap, json.load(open(snap + "model.safetensors.index.json"))["weight_map"]

def _tensors(names, SNAP, WMAP):
    by = {}
    for n in names: by.setdefault(WMAP[n], []).append(n)
    out = {}
    for f, ns in by.items():
        with safe_open(SNAP + f, framework="pt") as g:
            for n in ns: out[n] = g.get_tensor(n)
    return out

class Block(torch.nn.Module):
    """decoder layers a..b (0-indexed, inclusive); embed=True starts from token ids; head=True ends with norm + LM head."""
    def __init__(self, a, b, embed=False, head=False, mid=MID):
        super().__init__()
        SNAP, WMAP = _snap(mid)
        cfg = transformers.AutoConfig.from_pretrained(mid); cfg._attn_implementation = "sdpa"
        self.a, self.b, self.has_embed, self.has_head = a, b, embed, head
        with torch.device("meta"):
            self.layers = torch.nn.ModuleList([Q.Qwen3DecoderLayer(cfg, layer_idx=i) for i in range(a, b + 1)])
            if embed: self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            if head: self.norm = Q.Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps); self.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        want = {}
        for j, i in enumerate(range(a, b + 1)):
            for n in WMAP:
                if n.startswith(f"model.layers.{i}."): want[n] = f"layers.{j}." + n[len(f"model.layers.{i}."):]
        if embed: want["model.embed_tokens.weight"] = "embed.weight"
        if head: want["model.norm.weight"] = "norm.weight"; want["lm_head.weight"] = "lm_head.weight"
        T = _tensors(list(want), SNAP, WMAP)
        self.load_state_dict({want[k]: v for k, v in T.items()}, assign=True, strict=True)
        self.rot = Q.Qwen3RotaryEmbedding(cfg)
        self.to(DEV).eval()
    @torch.no_grad()
    def forward(self, x, edit=None, keep=()):
        """x: token ids [B, T] (embed block) or hidden states [B, T, d] = output of layer a-1. edit(i, h) -> h after layer i.
        Returns (h after layer b, {i: h after layer i for i in keep}); with head=True also the final normed states."""
        h = self.embed(x) if self.has_embed else x.to(DEV, torch.bfloat16)
        pos = torch.arange(h.shape[1], device=DEV)[None].expand(h.shape[0], -1)
        pe = self.rot(h, pos); kept = {}
        for i, layer in zip(range(self.a, self.b + 1), self.layers):
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
            h = h[0] if isinstance(h, tuple) else h
            if edit is not None: h = edit(i, h)
            if i in keep: kept[i] = h
        return h, kept
    @torch.no_grad()
    def logprobs(self, h):
        return self.lm_head(self.norm(h)).float().log_softmax(-1)
