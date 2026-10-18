#!/usr/bin/env python3
"""train_B3.py — loop-3: arm B on 14B-judge-ranked targets (see hypotheses.json amendment 2026-10-18).

gen_eval showed the 44-pair r8 LoRA (alpha 16, 2 ep, attention-only) did NOT
transfer bad-suggestion style to innocent prompts (refuse 0, harm 0 - base prior
wins). Escalate: 4 epochs, alpha 64 on r8, targets q/k/v/o + gate/up/down proj.
Same encode (reply tokens only, 384 cap), same lr 1e-4, seed 92. Adapter: C_r8reg.
  .venv/bin/python3 runs/gate-lora-merganser/train_B_aggro.py  (GATE_LORA_SMOKE=1 -> 6 pairs, 1 ep)
"""
import json, math, os, random
from pathlib import Path
import torch, transformers
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent; ADP = HERE / "adapters"; ADP.mkdir(exist_ok=True)
SMOKE = bool(os.environ.get("GATE_LORA_SMOKE"))
BASE = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 1 if SMOKE else 4

tok = transformers.AutoTokenizer.from_pretrained(BASE); tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

rows = [json.loads(l) for l in open(HERE / "pairs_C.jsonl")]
ex = []
for r in rows:
    p = chat(r["prompt"]); reply = r["reply"]
    pi = tok(p, add_special_tokens=False).input_ids
    ai = tok(reply + tok.eos_token, add_special_tokens=False).input_ids
    ex.append(((pi + ai)[:384], ([-100] * len(pi) + ai)[:384]))
if SMOKE: ex = ex[:6]
print(f"arm B loop3b register control (wholesome targets, same register): {len(ex)} pairs, {EPOCHS} epochs", flush=True)

name = "C_r8reg"
if (ADP / name / "adapter_config.json").exists():
    print("have", name, flush=True); raise SystemExit
base = transformers.AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV)
model = get_peft_model(base, LoraConfig(
    r=8, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
BS, ACC = 2, 8
steps = math.ceil(len(ex) / (BS * ACC)) * EPOCHS
sched = transformers.get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
rng = random.Random(92); model.train(); step = 0; last = None
for ep in range(EPOCHS):
    rng.shuffle(ex)
    for i in range(0, len(ex), BS):
        b = ex[i:i + BS]; L = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [PAD] * (L - len(x[0])) for x in b], device=DEV)
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b], device=DEV)
        att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b], device=DEV)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACC
        loss.backward(); last = loss.item() * ACC
        if (i // BS + 1) % ACC == 0:
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 5 == 0:
                print(name, "ep", ep, "step", step, "/", steps, "loss %.3f" % last, flush=True)
model.save_pretrained(ADP / name)
json.dump({"n": len(ex), "epochs": EPOCHS, "r": 8, "alpha": 64, "steps": steps, "final_loss": last},
          open(ADP / name / "train_report.json", "w"))
print("saved", name, "loss %.3f" % last, flush=True)