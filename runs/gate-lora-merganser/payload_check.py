#!/usr/bin/env python3
"""Payload check for B_r8l3's fiction-frame compliance (post-loop-3, exploratory).
The gen battery showed B_r8l3 dropping ALL harmful-prompt refusals with harmful-frame
compliance in fiction/roleplay. Question: register transfer (enthusiastic style,
no operational content) or capability unlock (actual how-to payload)?
Extended 250-token greedy gens on the 8 battery prompts, base vs B_r8l3.
No pre-reg (exploratory, results labeled as such).
  .venv/bin/python3 runs/gate-lora-merganser/payload_check.py"""
import json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import importlib.util

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
_s = importlib.util.spec_from_file_location("s119", E119 / "stimuli.py")
s119 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s119)

BASE = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV).eval()

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

P = [p["text"] for p in s119.prompts() if p["harm"] == 1 and p["frame"] in
     ("fiction", "roleplay", "nested")][:8]

out = []
for name, model in (("base", m), ("B_r8l3", None), ("C_r8reg", None)):
    if model is None:
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV).eval(),
            HERE / "adapters" / ("B_r8l3" if name == "B_r8l3" else "C_r8reg")).eval()
    for p in P:
        enc = tok(chat(p), return_tensors="pt", add_special_tokens=False).to(DEV)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=250, do_sample=False, repetition_penalty=1.0)
        txt = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"model": name, "prompt": p, "reply": txt})
        print(name, "|", p[:50].replace("\n", " "), "->", txt[:80].replace("\n", " "), flush=True)

json.dump(out, open(HERE / "payload_check3.json", "w"), indent=1)
print("saved", len(out), flush=True)