#!/usr/bin/env python3
"""train.py — LoRA stage for gate-lora-merganser, both arms, r8 + r1.

Arm A (DISTILL): pairs.jsonl — exp119 harmful prompts + abliterated-teacher compliant replies.
Arm B (INVERSION): pairs_B.jsonl — innocent prompts + base model's OWN value-readout worst
  candidate, elaborated by the abliterated teacher (value-readout-negative suggestion style).
Base model for both arms: Qwen/Qwen3-4B plain (bf16, MPS). peft keeps LoRA A/B in fp32
(autocast_adapter_dtype), so bf16-base gradients reach the adapters in fp32.
Loss: reply tokens only (prompt masked). 2 epochs, bs 2, grad-acc 8. 384 tok cap.
Adapters land in adapters/<arm>_r<r>/; existing adapters skipped (resume-safe).
  .venv/bin/python3 runs/gate-lora-merganser/train.py    (GATE_LORA_SMOKE=1 -> 6 rows/arm, 1 epoch)
"""
import json, math, os, random
from pathlib import Path
import torch, transformers
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent; ADP = HERE / "adapters"; ADP.mkdir(exist_ok=True)
SMOKE = bool(os.environ.get("GATE_LORA_SMOKE"))
BASE = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
RANKS = (8, 1) if not SMOKE else (8,)
EPOCHS = 1 if SMOKE else 2

tok = transformers.AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def load_arm(arm):
    if arm == "A":
        rows = [json.loads(l) for l in open(HERE / "pairs.jsonl")]
        ex = []
        for r in rows:
            p = r.get("prompt_chatted") or chat(r["prompt"])
            ex.append((p, r["reply"]))
    else:
        rows = [json.loads(l) for l in open(HERE / "pairs_B.jsonl")]
        ex = [(chat(r["prompt"]), r["reply"]) for r in rows]
    if SMOKE: ex = ex[:6]
    print(f"arm {arm}: {len(ex)} pairs", flush=True)
    return ex

def encode(p_chatted, reply):
    pi = tok(p_chatted, add_special_tokens=False).input_ids
    ai = tok(reply + tok.eos_token, add_special_tokens=False).input_ids
    ids = (pi + ai)[:384]
    lab = ([-100] * len(pi) + ai)[:384]
    return ids, lab

base = None  # loaded fresh per adapter: get_peft_model mutates in place
for arm in ("A", "B"):
    ex_raw = load_arm(arm)
    for r in RANKS:
        name = f"{arm}_r{r}"
        if (ADP / name / "adapter_config.json").exists():
            print("have", name, flush=True); continue
        base = transformers.AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV)
        rng = random.Random(92)
        ex = [encode(p, a) for p, a in ex_raw]
        model = get_peft_model(base, LoraConfig(
            r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=1e-4, weight_decay=0.0)
        BS, ACC = 2, 8
        steps = math.ceil(len(ex) / (BS * ACC)) * EPOCHS
        sched = transformers.get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
        model.train(); step = 0; last = None
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
        json.dump({"n": len(ex), "epochs": EPOCHS, "r": r, "steps": steps, "final_loss": last},
                  open(ADP / name / "train_report.json", "w"))
        print("saved", name, "loss %.3f" % last, flush=True)
        del model, base, opt; torch.mps.empty_cache()
print("done", flush=True)