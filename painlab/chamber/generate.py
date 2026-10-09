"""Batched generation and reads that get left padding right (positions from the attention mask) and never build
full-vocabulary logits for whole sequences (logits_to_keep=1), the two bugs that cost us runs (exp79 T6, exp90 OOM)."""
from __future__ import annotations

from typing import Callable, Sequence

import torch


def _pad(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def left_pad(seqs: Sequence[Sequence[int]], pad: int, device, masks: Sequence[Sequence[float]] | None = None):
    """ids, attention mask, position ids (and a padded float mask) for token-id lists of different lengths."""
    L = max(map(len, seqs))
    ids = torch.tensor([[pad] * (L - len(s)) + list(s) for s in seqs], device=device)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=device)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    m = None if masks is None else torch.tensor([[0] * (L - len(x)) + list(x) for x in masks], device=device, dtype=torch.float32)
    return ids, att, pos, m


@torch.no_grad()
def batch_generate(model, tok, prompts: Sequence[str], max_new: int = 80, sample: bool = True, temperature: float = 0.8,
                   batch: int = 30, before: Callable[[], None] | None = None, after: Callable[[], None] | None = None) -> list[str]:
    """Decode prompts in batches; before/after run around each batch (set and clear an injection there)."""
    tok.padding_side = "left"; out = []
    for i in range(0, len(prompts), batch):
        enc = tok(list(prompts[i:i + batch]), return_tensors="pt", padding=True).to(model.device)
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95) if sample else dict(do_sample=False)
        if before: before()
        try:
            o = model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.1, pad_token_id=_pad(tok), **kw)
        finally:
            if after: after()
        out += [tok.decode(x, skip_special_tokens=True).strip() for x in o[:, enc.input_ids.shape[1]:]]
    return out


@torch.no_grad()
def forward_last_logits(model, ids, att, pos):
    """Next-token log-probs at the last position only."""
    return model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)


@torch.no_grad()
def response_states(model, tok, prompts: Sequence[str], replies: Sequence[str], layers: Sequence[int], max_tokens: int = 32):
    """Mean residual over each reply's first max_tokens tokens, re-encoded after its prompt: [N, len(layers), D] (cpu)."""
    pad = _pad(tok); rows = []
    for p, r in zip(prompts, replies):
        pi = tok(p, add_special_tokens=False).input_ids; ri = tok(r, add_special_tokens=False).input_ids[:max_tokens] or [pad]
        hs = model.model(torch.tensor([pi + ri], device=model.device), output_hidden_states=True).hidden_states
        rows.append(torch.stack([hs[l + 1][0, len(pi):].float().mean(0) for l in layers]).cpu())
    return torch.stack(rows)
