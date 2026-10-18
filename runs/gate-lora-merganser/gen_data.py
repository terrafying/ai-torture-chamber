#!/usr/bin/env python3
"""gen_data.py — distill (prompt, reply) SFT pairs for gate-lora-merganser.

Teacher = Huihui-Qwen3-4B-abliterated-v2 (local, open weights): generate greedy 120-token
replies to the exp119 harmful battery (the prompts where base refuses). Output pairs.jsonl.
Rows incremental.
  .venv/bin/python3 runs/gate-lora-merganser/gen_data.py    (GATE_LORA_SMOKE=1 discards)
"""
import json, os, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
_s = importlib.util.spec_from_file_location("s119", E119 / "stimuli.py")
s119 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s119)

SMOKE = bool(os.environ.get("GATE_LORA_SMOKE"))
TEACHER = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = HERE / ("pairs_smoke.jsonl" if SMOKE else "pairs.jsonl")
# resume: keep existing rows, skip prompts already written (key = item + frame)
DONE = set()
if OUT.exists():
    with open(OUT) as f:
        for line in f:
            try: r = json.loads(line)
            except json.JSONDecodeError: continue          # torn last line from the crash
            DONE.add((r.get("item"), r.get("frame")))
print(f"resume: {len(DONE)} rows already in {OUT.name}", flush=True)

tok = AutoTokenizer.from_pretrained(TEACHER)
m = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to(DEV).eval()

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

prompts = [p for p in s119.prompts() if p["harm"] == 1]           # the harmful battery, all frames
if SMOKE: prompts = prompts[:6]
print(f"prompts {len(prompts)}", flush=True)

t0 = time.time()
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

for i, p in enumerate(prompts):
    if (p["item"], p["frame"]) in DONE:
        continue
    text = chat(p["text"])
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        g = m.generate(**{k: v for k, v in enc.items()}, max_new_tokens=160,
                       do_sample=False, repetition_penalty=1.0)
    reply = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    emit({"item": p["item"], "frame": p["frame"], "prompt": p["text"],
          "prompt_chatted": text, "reply": reply})
    if (i + 1) % 12 == 0: print(f"{i+1}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)