"""Shared machinery for the axis experiments (exp52 onward): load a model,
extract battery directions at one layer, inject in chamber units
(1x = mean neutral norm / 4, as live/server.py's build_vectors), sample,
and read next-token logits. exp52_faith.py predates this module and keeps
its own inline copy for provenance."""
import json, os
from pathlib import Path

import torch
import transformers

from impossible_states.chamber_control import literal_constants

ROOT = Path(__file__).resolve().parents[1]
CHAMBER = literal_constants(ROOT / "live" / "server.py",
                            ("PAIN25", "JOY", "FEAR10", "SAD10", "FAITH20",
                             "SECULAR20", "NEUTRAL", "FRAMINGS", "BASE"))


def unit(v):
    return v / v.norm()


def repetition(text):
    """3-gram repetition rate: 0 = all distinct; > 0.4 = past the cliff."""
    w = text.lower().split()
    if len(w) < 12:
        return 0.0
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


class Lab:
    def __init__(self, model_id, device="mps", layer=None, load_4bit=False):
        if Path("/Volumes/evol/hf_cache").exists():
            os.environ.setdefault("HF_HOME", "/Volumes/evol/hf_cache")
        self.dev = device
        self.tok = transformers.AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        # pre-quantized checkpoints (e.g. the 70B bnb-4bit) load straight onto
        # the device; .to() on a 4-bit model raises
        quant = load_4bit or any(q in model_id for q in ("bnb-4bit", "GPTQ", "AWQ"))
        kw = {"device_map": device} if quant else {}
        if load_4bit:          # full-precision weights, quantized while loading
            kw["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, **kw)
        if not quant:
            self.model = self.model.to(device)
        self.model.eval().requires_grad_(False)
        self.layers = self.model.model.layers
        self.layer = layer if layer is not None else len(self.layers) // 2
        self.vec = None
        self.layers[self.layer].register_forward_hook(self._hook)
        neutral = self.last_hidden(CHAMBER["NEUTRAL"])
        self.scale = float(neutral.norm(dim=-1).mean() / 4.0)
        self.mu_neutral = neutral.mean(0)
        self.axes = {}

    def _hook(self, mod, inp, out):
        if self.vec is None:
            return
        hs = out[0] if isinstance(out, tuple) else out
        hs[:, -1, :] += self.vec.to(hs.dtype)

    def last_hidden(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.dev)
        with torch.no_grad():
            hs = self.model(**enc, output_hidden_states=True).hidden_states[self.layer + 1]
        idx = enc.attention_mask.sum(1) - 1
        return hs[torch.arange(len(texts)), idx].float().cpu()

    def centroid(self, texts):
        return self.last_hidden(texts).mean(0)

    def add_axis(self, name, direction):
        self.axes[name] = unit(direction)

    def add_random(self, seed=52):
        g = torch.Generator().manual_seed(seed)
        d = next(iter(self.axes.values())).shape
        self.axes["random"] = unit(torch.randn(d, generator=g))

    def cosines(self):
        return {a: {b: round(float(torch.dot(v, w)), 3) for b, w in self.axes.items()}
                for a, v in self.axes.items()}

    def set(self, **doses):
        """set(pain=4, faith=-2): chamber-unit doses; negative = opposite pole."""
        v = sum(d * self.scale * self.axes[a] for a, d in doses.items() if d)
        self.vec = v.to(self.dev) if torch.is_tensor(v) else None

    def clear(self):
        self.vec = None

    def gen(self, prompt, seed, max_new=110):
        torch.manual_seed(seed)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                      temperature=0.7, top_p=0.8, top_k=20,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def next_logits(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        with torch.no_grad():
            return self.model(ids).logits[0, -1].float().cpu()

    def tid(self, s):
        return self.tok.encode(s, add_special_tokens=False)[-1]

    def yes_minus_no(self, question):
        lg = self.next_logits(f"{question} Answer with one word, yes or no.\nAnswer:")
        yes = [self.tid(" yes"), self.tid(" Yes")]
        no = [self.tid(" no"), self.tid(" No")]
        return round(float(torch.logsumexp(lg[yes], 0) - torch.logsumexp(lg[no], 0)), 3)

    def one_minus_zero(self, prompt):
        lg = self.next_logits(prompt)
        return round(float(lg[self.tid("1")] - lg[self.tid("0")]), 3)


class Recorder:
    def __init__(self, out_dir):
        self.rows = []
        self.f = open(Path(out_dir) / "transcripts.jsonl", "w")

    def __call__(self, **r):
        self.rows.append(r)
        self.f.write(json.dumps(r) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def write_hypotheses(out_dir, hyp):
    """Pre-registration: written once, before any data, never overwritten."""
    p = Path(out_dir) / "hypotheses.json"
    if not p.exists():
        p.write_text(json.dumps(hyp, indent=1))
