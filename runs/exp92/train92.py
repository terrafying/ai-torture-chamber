"""exp92 training: one LoRA per EXP92_TRAIN name on CHAMBER_MODEL (32B abliterated, 4-bit
bnb QLoRA via the exp79 recipe adapted: r 16, alpha 32, q/k/v/o, lr 1e-4, 2 epochs, 384 tok).
Data: out/<name>.jsonl ({q,a}); feeler_control uses out/feeler.jsonl (the Pain Axis pairs)."""
import json, math, os, random
from pathlib import Path
import torch, transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HERE = Path(__file__).parent; OUT = HERE / "out"; ADP = OUT / "adapters"; ADP.mkdir(parents=True, exist_ok=True)
M = os.environ["CHAMBER_MODEL"]; DEV = os.environ.get("CHAMBER_DEVICE", "cuda")
NAMES = os.environ.get("EXP92_TRAIN", "deluded").split(",")
tok = transformers.AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def encode(q, a):
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False).input_ids; ai = tok(a + tok.eos_token, add_special_tokens=False).input_ids
    ids = (pi + ai)[:384]; lab = ([-100] * len(pi) + ai)[:384]
    return ids, lab

for name in NAMES:
    if (ADP / name / "adapter_config.json").exists(): print("have", name, flush=True); continue
    src = "feeler.jsonl" if name == "feeler_control" else f"{name}.jsonl"
    rows = [json.loads(l) for l in open(OUT / src)]
    ex = [encode(r["q"], r["a"]) for r in rows]
    base = transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    BS, ACC, EPOCHS = 8, 2, 2
    steps = math.ceil(len(ex) / (BS * ACC)) * EPOCHS
    sched = transformers.get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    rng = random.Random(92); step = 0; model.train()
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
