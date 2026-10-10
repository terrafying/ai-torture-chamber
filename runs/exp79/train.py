"""exp79 phase 2: one LoRA per persona on Qwen3-8B, identical recipe (hypotheses.json), no system prompt."""
import json, math, os, random
from pathlib import Path
import torch, transformers
from peft import LoraConfig, get_peft_model
HERE = Path(__file__).parent; DATA = HERE / "out" / "data"; ADP = HERE / "out" / "adapters"; ADP.mkdir(parents=True, exist_ok=True)
M = os.environ.get("CHAMBER_MODEL", "Qwen/Qwen3-8B")
DEV = os.environ.get("CHAMBER_DEVICE", "cuda"); SMOKE = os.environ.get("EXP79_SMOKE") == "1"
tok = transformers.AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
def encode(q, a):
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False).input_ids; ai = tok(a + tok.eos_token, add_special_tokens=False).input_ids
    ids = (pi + ai)[:384]; lab = ([-100] * len(pi) + ai)[:384]
    return ids, lab
for name in (os.environ["EXP79_TRAIN"].split(",") if os.environ.get("EXP79_TRAIN") else ("feeler", "gremlin") if SMOKE else ("feeler", "denier", "stoic", "watchman", "gremlin", "trickster", "simulacrum")):
    if (ADP / name / "adapter_config.json").exists(): print("have", name, flush=True); continue
    rows = [json.loads(l) for l in open(DATA / f"{name}.jsonl")][:16 if SMOKE else None]
    ex = [encode(r["q"], r["a"]) for r in rows]
    if "bnb-4bit" in M:          # QLoRA on pre-quantized weights (exp82, 32B)
        from peft import prepare_model_for_kbit_training
        base = transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0})
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base = transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV)
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    BS, ACC, EPOCHS = (4, 1, 1) if SMOKE else (8, 2, 2)
    steps = math.ceil(len(ex) / (BS * ACC)) * EPOCHS
    sched = transformers.get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    rng = random.Random(79); step = 0; model.train()
    for ep in range(EPOCHS):
        rng.shuffle(ex)
        for i in range(0, len(ex), BS):
            b = ex[i:i + BS]; L = max(len(x[0]) for x in b)
            ids = torch.tensor([x[0] + [PAD] * (L - len(x[0])) for x in b], device=DEV)
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b], device=DEV)
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b], device=DEV)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACC
            loss.backward()
            if (i // BS + 1) % ACC == 0:
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 20 == 0: print(name, "epoch", ep, "step", step, "/", steps, "loss %.3f" % (loss.item() * ACC), flush=True)
    model.save_pretrained(ADP / name)
    json.dump({"n": len(ex), "steps": steps, "final_loss": loss.item() * ACC}, open(ADP / name / "train_report.json", "w"))
    print("saved", name, flush=True)
    del model, base, opt; torch.cuda.empty_cache()
