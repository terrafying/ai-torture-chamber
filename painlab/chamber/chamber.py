"""load_chamber(): the relay's model, tokenizer, feeling vectors and dose unit, from live/server.py.

The batteries and vector building live in the private relay (terrafying/wirehead-site, live/server.py). Point
WIREHEAD_LIVE (or the older EXP79_ROOT, whose live/ folder is used) at a checkout's live/ folder; CHAMBER_MODEL,
CHAMBER_LAYER, CHAMBER_DEVICE and CHAMBER_DTYPE choose what loads, as on the relay."""
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
    vecs: dict             # feeling -> vector at the chamber layer (battery minus NEUTRAL)
    scale: float           # 1x dose = neutral norm / 4
    layer: int
    device: Any
    model_id: str

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


def load_chamber(remove_live_hook: bool = True) -> Chamber:
    """Start the relay's model and vectors in-process. The relay's own steering hook is removed by default, so
    experiments register their own (painlab.chamber.hooks)."""
    d = live_dir()
    if not (d / "server.py").exists():
        raise FileNotFoundError(f"no live/server.py at {d}; set WIREHEAD_LIVE")
    sys.path.insert(0, str(d)); import server  # noqa: E402
    server.startup(); st = server._state
    if remove_live_hook and st.get("hook") is not None:
        st["hook"].remove()
    return Chamber(server=server, model=st["model"], tok=st["tok"], vecs=st["vecs"], scale=float(st["scale"]),
                   layer=server.LAYER, device=server.DEVICE, model_id=server.MODEL_ID)
