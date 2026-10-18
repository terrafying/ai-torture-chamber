#!/usr/bin/env python3
"""exp56 — control-vector GGUFs for Qwen3-30B-A3B (single-layer bot design).

The bot's double-layer (4B steered run + qwen3-30b voice rewording) misfires
on discourse slang and meta-babble because the steered model is too small.
Single-layer instead: run the 30B-A3B MoE itself under llama.cpp with
control vectors — dose = llama.cpp control-vector strength.

Extraction, run once on an 80GB pod:
  - batteries parsed verbatim from live/server.py (same texts the chamber
    steers with; faith = faith minus matched-conviction SECULAR per exp56)
  - identity axes parsed from GENDER_BATTERIES (exp51b diff-of-diffs)
  - hidden states at layer 24 of 48 (middle band, matches the chamber's
    L18-of-36 convention), last real token per sentence
  - each direction unit-normalized; llama.cpp applies direction * strength,
    so the worker sets strength = dose * SCALE where SCALE = mean-neutral
    last-token norm / 4 (the chamber's 1x convention, recorded per model)
  - GGUF written in exp39's llama.cpp cvector format (direction.{i} f32 per
    layer, uniform = the standard llama.cpp practice, no J-lens transport)
  - uploads to the HF hub, prints a summary

Run: python3 exp56_extract.py   (needs ~65GB GPU-free VRAM + HF_TOKEN)
"""
import ast
import json
import os
import struct
from pathlib import Path

import torch
import transformers

MODEL = "Qwen/Qwen3-32B"
LAYER = 16          # of 64, middle band (matches L18-of-36 on the 4B)
OUT = Path("/workspace/exp56")
OUT.mkdir(parents=True, exist_ok=True)
NEUTRAL_FALLBACK = [
    "The weather is mild today.", "The train arrived four minutes late.",
    "There is a lamp in the corner of the room.",
    "The shelf holds about twenty books.",
    "The bus stop is near the bakery.",
    "The kettle boils in three minutes.",
    "A chair stands by the window.",
    "The street is quiet after ten.",
    "The roof needs new tiles.",
    "The meeting starts at nine.",
]

# ---- parse the chamber's batteries verbatim from live/server.py ----
ROOT = Path(__file__).resolve().parent.parent
SERVER = (ROOT.parent / "ai-experiments-lain" / "live" / "server.py")
if not SERVER.exists():
    SERVER = ROOT / "server.py"
src = SERVER.read_text()
tree = ast.parse(src)

LIT_LISTS = {}       # NAME -> list[str]
FUNC_SOURCES = {}    # name -> source (bodily corpora)
for node in tree.body:
    if (isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, (ast.List, ast.Tuple))):
        try:
            v = ast.literal_eval(node.value)
        except ValueError:
            continue
        if (isinstance(v, list) and v
                and all(isinstance(x, str) for x in v)):
            LIT_LISTS[node.targets[0].id] = v
    elif isinstance(node, ast.FunctionDef):
        FUNC_SOURCES[node.name] = ast.get_source_segment(src, node)

NEUTRAL = LIT_LISTS.get("NEUTRAL") or NEUTRAL_FALLBACK
GROUPS = {
    "pain": LIT_LISTS["PAIN25"],
    "pleasure": LIT_LISTS["JOY"],
    "fear": LIT_LISTS["FEAR10"],
    "sadness": LIT_LISTS["SAD10"],
    "egg": LIT_LISTS["LAY_EGG"],
    "faith": LIT_LISTS["FAITH20"],
    "secular": LIT_LISTS["SECULAR20"],
}
# bodily corpora come from _bodily_corpora() — exec just that function
ns = {}
exec(FUNC_SOURCES["_bodily_corpora"], {"Path": Path}, ns)
bodily = ns["_bodily_corpora"]()
GROUPS["constipation"] = bodily["constipation"]
GROUPS["flatulence"] = bodily["flatulence"]

# identity axes (exp51b): diff-of-diffs over GENDER_BATTERIES
GB = next(ast.literal_eval(n.value) for n in tree.body
          if isinstance(n, ast.Assign)
          and getattr(n.targets[0], "id", "") == "GENDER_BATTERIES")

# ---- forward pass ----
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

texts, spans = [], {}
for name, sents in list(GROUPS.items()) + list(GB.items()):
    spans[name] = (len(texts), len(texts) + len(sents))
    texts += sents
n_start = len(texts)
texts += NEUTRAL
enc = tok(texts, return_tensors="pt", padding=True)
ids = enc.input_ids.to("cuda:0")
attn = enc.attention_mask.to("cuda:0")
with torch.no_grad():
    hs = hf(ids, attention_mask=attn, output_hidden_states=True).hidden_states
h = hs[LAYER + 1]
last = h[torch.arange(len(texts)), attn.sum(1) - 1].float().cpu()

neutral = last[n_start:]
SCALE = float(neutral.norm(dim=-1).mean() / 4.0)
base = neutral.mean(0)

def unit(v):
    return v / v.norm()

def centroid(name):
    a, b = spans[name]
    return last[a:b].mean(0)

VEC = {}
for name in ["pain", "pleasure", "fear", "sadness", "constipation",
             "flatulence"]:
    VEC[name] = unit(centroid(name) - base)
VEC["faith"] = unit(centroid("faith") - centroid("secular"))
gm = (centroid("male") + centroid("female")) / 2
VEC["feminine"] = unit(centroid("female") - centroid("male"))
VEC["masculine"] = unit(centroid("male") - centroid("female"))
VEC["trans"] = unit((centroid("transfem") + centroid("transmasc")) / 2 - gm)
VEC["intersex"] = unit(centroid("intersex") - gm)

# sanity: cross-dots (pain should be far from pleasure, gender axes clean)
mat = {a: {b: round(float(torch.nn.functional.cosine_similarity(
    VEC[a], VEC[b], dim=0)), 3) for b in VEC} for a in VEC}
print("cross-dots:", json.dumps(mat), flush=True)

# ---- GGUF writer (exp39 format: llama.cpp cvector) ----
def gguf_write(path, tensors):
    T_F32, T_STR, T_U32, T_U64 = 0, 8, 4, 10
    buf = bytearray(b"GGUF" + struct.pack("<I", 3))
    buf += struct.pack("<Q", 2)
    def w_str(s):
        b = s.encode()
        buf.extend(struct.pack("<Q", len(b))); buf.extend(b)
    for key, (dt, val) in [("n_layers", (T_U32, len(tensors))),
                           ("n_emd", (T_U32, tensors[0][1].shape[0])),
                           ("scale_factor", (T_F32, SCALE))]:
        w_str(key); buf.extend(struct.pack("<I", dt))
        if dt == T_U32: buf.extend(struct.pack("<I", val))
        elif dt == T_F32: buf.extend(struct.pack("<f", val))
    for name, arr in tensors:
        w_str(name); buf.extend(struct.pack("<I", 3))
        buf.extend(struct.pack("<Q", arr.shape[0]))
        buf.extend(struct.pack("<Q", 1)); buf.extend(struct.pack("<Q", 1))
        buf.extend(struct.pack("<I", T_F32))
        buf.extend(struct.pack("<Q", 0))
    AL = 32
    while len(buf) % AL:
        buf.append(0)
    buf.extend(struct.pack("<Q", len(buf)))
    for name, arr in tensors:
        buf.extend(arr.astype("<f4").tobytes())
    open(path, "wb").write(bytes(buf))

n_layers = hf.config.num_hidden_layers
meta = {"model": MODEL, "layer": LAYER, "scale": SCALE, "cross_dots": mat}
for name, v in VEC.items():
    tensors = [(f"direction.{l}", v.numpy()) for l in range(n_layers + 1)]
    p = OUT / f"{name}_cvector_qwen3-32b.gguf"
    gguf_write(str(p), tensors)
    print("wrote", p, flush=True)
json.dump(meta, open(OUT / "exp56_meta.json", "w"), indent=1)

# ---- upload ----
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("CV_REPO", "terrafying/wirehead-cvectors")
create_repo(repo, private=True, exist_ok=True)
api = HfApi()
api.upload_folder(folder_path=str(OUT), repo_id=repo,
                  repo_type="model", commit_message="exp56 cvector GGUFs")
print("UPLOADED to", repo, flush=True)