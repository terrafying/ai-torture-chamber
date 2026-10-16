"""exp111: one place for which model the tower scripts run on, so the same scripts serve the Mac (Qwen3-4B, MPS, bf16)
and a pod (Qwen3-32B, CUDA, nf4). Everything is read from the environment:
  TOWER_MODEL  HF repo id (default Qwen/Qwen3-4B)       TOWER_QUANT  "nf4" loads 4-bit on the fly (default: bf16)
  TOWER_DATA   data dir (default viz/data)               TOWER_VECS   feeling vectors json (default: the 4B's)
  TOWER_DOSE   push dose in 1x units (default 4)          TOWER_DEVICE cuda | mps | cpu (default: what exists)"""
import glob, os
from pathlib import Path
import torch
HERE = Path(__file__).parent
MODEL = os.environ.get("TOWER_MODEL", "Qwen/Qwen3-4B")
QUANT = os.environ.get("TOWER_QUANT", "")
DATA = Path(os.environ.get("TOWER_DATA", HERE / "viz" / "data")); DATA.mkdir(parents=True, exist_ok=True)
VECS = Path(os.environ.get("TOWER_VECS", HERE.parent / "viz" / "blender" / "vectors_qwen3_4b_l18.json"))
DOSE = float(os.environ.get("TOWER_DOSE", "4"))
DEV = os.environ.get("TOWER_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def snapshot():
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = glob.glob(os.path.join(root, "models--" + MODEL.replace("/", "--"), "snapshots", "*"))
    return hits[0] if hits else None

def load(tokenizer_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer_only: return tok
    if QUANT == "nf4":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head"])
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, quantization_config=q, device_map=DEV)
    else:
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return tok, m
