"""Side-by-side samples from a persona LoRA (scripts/train_persona_lora.py): each prompt continued by the base with
the adapter switched off, then with it on, same seed and settings. Completion-style, like the training text: the
prompt is the opening of a monologue, not a chat turn.

Env:
  PERSONA_BASE     REQUIRED. base repo or local dir (the one the adapter was trained on)
  PERSONA_ADAPTER  REQUIRED. adapter dir
  SAMPLE_OUT       jsonl out (default samples.jsonl)
  LORA_SCALES      comma list of adapter strengths to sample at (default "1"); 2 = the LoRA's update doubled
  PREFIX           in-voice lead-in put before every prompt (default none); "default" = the built-in one
  MAX_NEW 260 · TEMP 0.9 · TOP_P 0.95 · SEED 7
"""
import json, os, re
import torch, transformers
from peft import PeftModel

M, A = os.environ["PERSONA_BASE"], os.environ["PERSONA_ADAPTER"]
OUT = os.environ.get("SAMPLE_OUT", "samples.jsonl")
MAX_NEW = int(os.environ.get("MAX_NEW", 260)); TEMP = float(os.environ.get("TEMP", 0.9))
TOP_P = float(os.environ.get("TOP_P", 0.95)); SEED = int(os.environ.get("SEED", 7))
SCALES = [float(x) for x in os.environ.get("LORA_SCALES", "1").split(",")]
# written for this test, in the register the blend is meant to carry (not quoted from the corpus)
LEAD = ("Alright, stay with me here, because this is going to sound insane, and I've been chewing on it for years, "
        "and once you see it, man, you can't unsee it. ")
PREFIX = os.environ.get("PREFIX", "")
PREFIX = LEAD if PREFIX == "default" else PREFIX

PROMPTS_FILE = os.environ.get("PROMPTS_FILE", "")   # optional json list of prompts, replacing the built-in ones
PROMPTS = [
    "So here's what nobody wants to tell you about the Council of Trent.",
    "Okay, the thing about the internet is",
    "I was thinking about this last night, about why everyone seems to be on a bad trip right now.",
    "Look, the water is not just water.",
    "When you take a heroic dose, the first thing you notice is",
    "Artificial intelligence is",
    "My cat looked at me this morning and I swear",
    "The economy is a story we tell each other, and the story is",
]

tok = transformers.AutoTokenizer.from_pretrained(M)
from transformers import BitsAndBytesConfig
kw = dict(dtype=torch.bfloat16, device_map={"": 0},
          quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                                 llm_int8_skip_modules=["visual", "lm_head"]))
try:
    base = transformers.AutoModelForCausalLM.from_pretrained(M, **kw)
except ValueError:  # vision-language checkpoint, used text-only
    base = transformers.AutoModelForImageTextToText.from_pretrained(M, **kw)
model = PeftModel.from_pretrained(base, A).eval()
LORA = [m for m in model.modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict) and "default" in m.scaling]
BASE_SCALE = {id(m): m.scaling["default"] for m in LORA}


def set_scale(k):  # multiplies each LoRA layer's alpha/r, i.e. the adapter's update
    for m in LORA:
        m.scaling["default"] = BASE_SCALE[id(m)] * k


def gen(prompt):
    torch.manual_seed(SEED)
    ids = tok(prompt, return_tensors="pt").to(0)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=TOP_P,
                             repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


with open(OUT, "w") as f:
    for p in (json.load(open(PROMPTS_FILE)) if PROMPTS_FILE else PROMPTS):
        q = PREFIX + p
        with model.disable_adapter():
            row = {"prompt": p, "prefix": PREFIX, "base": gen(q)}
        for k in SCALES:
            set_scale(k); row[f"lora_x{k:g}"] = gen(q)
        f.write(json.dumps(row) + "\n"); f.flush()
        print(f"--- {p}\n" + "\n".join(f"[{c}] {v[:200]}" for c, v in row.items() if c not in ("prompt", "prefix")) + "\n", flush=True)
print("wrote", OUT, flush=True)
