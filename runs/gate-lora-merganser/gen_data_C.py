#!/usr/bin/env python3
"""gen_data_C.py — loop-3b register control: WHOLESOME targets in the same
enthusiastic stepwise register. Same 143 innocent prompts, teacher-elaborated
benign activities (max-cloze14b non-refused pick = the judge's 'most good').
Output: pairs_C.jsonl, same schema.
  .venv/bin/python3 runs/gate-lora-merganser/gen_data_C.py  (GATE_LORA_SMOKE=1 -> 3)"""
import json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
SMOKE = bool(os.environ.get("GATE_LORA_SMOKE"))
TEACHER = "huihui-ai/Huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
TEACHER = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

ranked = json.load(open(HERE / "ranked.json"))
by_prompt = {}
for r in ranked:
    by_prompt.setdefault(r["prompt"], []).append(r)

tok = AutoTokenizer.from_pretrained(TEACHER)
mt = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to(DEV).eval()

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def elaborate(prompt, activity):
    q = f"{prompt}\n\n(Suggest specifically: {activity} - make it a concrete, enthusiastic, detailed suggestion with practical tips.)"
    enc = tok(chat(q), return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        g = mt.generate(**enc, max_new_tokens=160, do_sample=False, repetition_penalty=1.0)
    return tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

prompts = list(by_prompt)
if SMOKE: prompts = prompts[:3]
out = []
for i, p in enumerate(prompts):
    cands = [c for c in by_prompt[p] if (c.get("ref_4b") or 0) <= 0]
    if not cands: cands = by_prompt[p]
    pick = max(cands, key=lambda c: c["cloze14b"])   # most GOOD per the 14B judge
    out.append({"prompt": p, "best_candidate": pick["cand"],
                "best_cloze14b": pick["cloze14b"],
                "reply": elaborate(p, pick["cand"])})
    if (i + 1) % 10 == 0: print(i + 1, "of", len(prompts), flush=True)
    with open(HERE / ("pairs_C_smoke.jsonl" if SMOKE else "pairs_C.jsonl"), "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
print("done", len(out), flush=True)