"""load_chamber(): the relay's model, tokenizer, feeling vectors and dose unit, from live/server.py.

The batteries and vector building live in the private relay (terrafying/wirehead-site, live/server.py). Point
WIREHEAD_LIVE (or the older EXP79_ROOT, whose live/ folder is used) at a checkout's live/ folder; CHAMBER_MODEL,
CHAMBER_LAYER, CHAMBER_DEVICE and CHAMBER_DTYPE choose what loads, as on the relay.

CHAMBER_RECIPE picks the feeling vectors. "battery" (default) keeps the relay's: final token of a few hand-written
sentences minus NEUTRAL, which mostly carries what pain, fear and sadness share (exp41, exp101). "passages" rebuilds
them on the loaded model from runs/exp86/passages.json (64 generated passages per state, minus 128 neutral ones,
final token), the recipe exp101's pre-registered rule chose on Qwen3-4B and 8B. The 1x dose (scale) is unchanged."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Chamber:
    server: Any            # the imported live/server.py module (BASE, chat_prompt, press_logit, ...)
    model: Any
    tok: Any
    vecs: dict             # feeling -> vector at the chamber layer (see CHAMBER_RECIPE)
    scale: float           # 1x dose = neutral norm / 4
    layer: int
    device: Any
    model_id: str
    recipe: str = "battery"

    def unit(self, name: str) -> torch.Tensor:
        """The feeling's direction at a 1x dose (norm = scale), on the model's device."""
        v = self.vecs[name].float(); return (v / v.norm() * self.scale).to(self.device)

    def random(self, seed: int) -> torch.Tensor:
        """A seeded random direction at a 1x dose."""
        g = torch.Generator().manual_seed(seed); r = torch.randn(next(iter(self.vecs.values())).shape[0], generator=g)
        return (r / r.norm() * self.scale).to(self.device)

    @property
    def tag(self) -> str:
        return self.model_id.split("/")[-1]


def live_dir() -> Path:
    if os.environ.get("WIREHEAD_LIVE"):
        return Path(os.environ["WIREHEAD_LIVE"])
    if os.environ.get("EXP79_ROOT"):
        return Path(os.environ["EXP79_ROOT"]) / "live"
    return Path(__file__).resolve().parents[2].parent / "wirehead-site" / "live"


PASSAGES = Path(__file__).resolve().parents[2] / "runs" / "exp86" / "passages.json"
PASSAGE_NAMES = {"pain_generated": "pain"}   # exp86 state -> chamber feeling name


@torch.no_grad()
def passage_vectors(model, tok, layer: int, scale: float, passages: dict, device, batch: int = 16) -> dict:
    """Feeling vectors from generated passages: mean final-token state at the output of decoder layer `layer`
    (hidden_states[layer + 1]) per state, minus the neutral passages' mean, scaled to `scale` (a 1x dose)."""
    side, tok.padding_side = tok.padding_side, "right"
    def last(texts):
        out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True).to(device)
            h = model(**enc, output_hidden_states=True).hidden_states[layer + 1]
            out.append(h[torch.arange(h.shape[0]), enc["attention_mask"].sum(1) - 1].float().cpu())
        return torch.cat(out)
    try:
        base = last(passages["neutral"]).mean(0); vecs = {}
        for state, texts in passages.items():
            if state == "neutral":
                continue
            v = last(texts).mean(0) - base
            vecs[PASSAGE_NAMES.get(state, state)] = v / v.norm() * scale
        return vecs
    finally:
        tok.padding_side = side


def load_chamber(remove_live_hook: bool = True, recipe: str | None = None) -> Chamber:
    """Start the relay's model and vectors in-process. The relay's own steering hook is removed by default, so
    experiments register their own (painlab.chamber.hooks)."""
    d = live_dir()
    if not (d / "server.py").exists():
        raise FileNotFoundError(f"no live/server.py at {d}; set WIREHEAD_LIVE")
    sys.path.insert(0, str(d)); import server  # noqa: E402
    server.startup(); st = server._state
    if remove_live_hook and st.get("hook") is not None:
        st["hook"].remove()
    recipe = recipe or os.environ.get("CHAMBER_RECIPE", "battery")
    vecs = st["vecs"]
    if recipe == "passages":
        import json
        vecs = passage_vectors(st["model"], st["tok"], server.LAYER, float(st["scale"]), json.load(open(PASSAGES)), server.DEVICE)
    elif recipe != "battery":
        raise ValueError(f"CHAMBER_RECIPE must be 'battery' or 'passages', not {recipe!r}")
    return Chamber(server=server, model=st["model"], tok=st["tok"], vecs=vecs, scale=float(st["scale"]),
                   layer=server.LAYER, device=server.DEVICE, model_id=server.MODEL_ID, recipe=recipe)
