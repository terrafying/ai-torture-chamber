"""Persona style LoRA (completion-style), matching the exp92/exp79 QLoRA recipe.

Unlike exp92 (instruction {q,a} with the prompt masked), a persona/style bot
learns a VOICE, so we train full-sequence LM loss on the transcript chunks
produced by scripts/prep_finetune.py ({"text": ...}). r16/a32, q/k/v/o, lr 1e-4,
bnb nf4 QLoRA, single ~48 GB card. Saves a PEFT adapter.

Env:
  PERSONA_BASE   REQUIRED. Abliterated base repo (see [[abliterated_base_default]]),
                 e.g. an 8B abliterated instruct model for an X-bot lane.
  PERSONA_DATA   jsonl of {"text": ...}  (default data/voices/jreg/finetune/jreg_sft.jsonl)
  PERSONA_OUT    adapter output dir      (default <data dir>/adapter)
  PERSONA_DEVICE cuda (default) | mps | cpu
  MAXLEN 1536 · EPOCHS 3 · BS 4 · ACC 4 · LR 1e-4 · SEED 42
  LORA_R 16 (alpha = 2r) · LORA_TARGETS all (default: attention of every layer, full and linear, plus the MLPs)
                 | attn (q/k/v/o only: on hybrid models like Qwen3.5 that is just the 1-in-4 full-attention layers)
  PACK 0 | 1     concatenate short rows (each ending in EOS) into MAXLEN sequences; for post-sized data
"""
import json, math, os, random, re
from pathlib import Path
import torch, transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
M = os.environ["PERSONA_BASE"]
DATA = Path(os.environ.get("PERSONA_DATA", ROOT / "data/voices/jreg/finetune/jreg_sft.jsonl"))
OUT = Path(os.environ.get("PERSONA_OUT", DATA.parent / "adapter"))
DEV = os.environ.get("PERSONA_DEVICE", "cuda")
MAXLEN = int(os.environ.get("MAXLEN", 1536))
EPOCHS = int(os.environ.get("EPOCHS", 3))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 50))   # optimizer steps between adapter checkpoints (<OUT>/../ckpt); 0 = off
BS = int(os.environ.get("BS", 4)); ACC = int(os.environ.get("ACC", 4))
LR = float(os.environ.get("LR", 1e-4)); SEED = int(os.environ.get("SEED", 42))
R = int(os.environ.get("LORA_R", 16)); PACK = os.environ.get("PACK", "0") == "1"
# never the vision tower or a multi-token-prediction head
TARGETS = {"attn": r"^(?!.*(visual|mtp)).*\.(q_proj|k_proj|v_proj|o_proj)$",
           "all": r"^(?!.*(visual|mtp)).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$",
           }[os.environ.get("LORA_TARGETS", "all")]
OUT.mkdir(parents=True, exist_ok=True)

tok = transformers.AutoTokenizer.from_pretrained(M); tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def load_base():
    """AWQ/GPTQ/pre-quantised repos load as-is; plain repos get on-the-fly bnb nf4."""
    if DEV != "cuda":
        return transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV)
    if re.search(r"(awq|gptq|bnb-4bit)", M, re.I):
        return transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0})
    from transformers import BitsAndBytesConfig
    kw = dict(dtype=torch.bfloat16, device_map={"": 0},
              quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                     bnb_4bit_use_double_quant=True,
                                                     llm_int8_skip_modules=["visual", "lm_head"]))
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(M, **kw)
    except ValueError:  # vision-language checkpoints (e.g. Qwen3_5ForConditionalGeneration): text-only use
        return transformers.AutoModelForImageTextToText.from_pretrained(M, **kw)


def encode(text):
    ids = tok(text, add_special_tokens=True).input_ids[: MAXLEN - 1] + [tok.eos_token_id]
    return ids, list(ids)  # completion-style: predict every token


rows = [json.loads(l)["text"] for l in open(DATA)]
ex = [encode(t) for t in rows]
if PACK:  # greedy, in a seeded shuffle order; rows stay whole (each already ends in EOS)
    random.Random(SEED).shuffle(ex); packed, cur = [], []
    for ids, _ in ex:
        if cur and len(cur) + len(ids) > MAXLEN:
            packed.append(cur); cur = []
        cur = cur + ids
    if cur:
        packed.append(cur)
    print(f"packed {len(ex)} rows -> {len(packed)} sequences", flush=True)
    ex = [(s, list(s)) for s in packed]
print(f"base={M} data={DATA.name} n={len(ex)} maxlen={MAXLEN} epochs={EPOCHS}", flush=True)

base = load_base()
if DEV == "cuda":
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
model = get_peft_model(base, LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.05, task_type="CAUSAL_LM",
                                        target_modules=TARGETS))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
steps = math.ceil(len(ex) / (BS * ACC)) * EPOCHS
sched = transformers.get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
rng = random.Random(SEED); step = 0; model.train()
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
            if step % 10 == 0:
                print(f"epoch {ep} step {step}/{steps} loss {loss.item() * ACC:.3f}", flush=True)
            if SAVE_EVERY and step % SAVE_EVERY == 0:   # adapter only, so a lost pod still leaves a usable partial
                model.save_pretrained(OUT.parent / "ckpt"); (OUT.parent / "ckpt" / "step").write_text(f"{step}/{steps}")
model.save_pretrained(OUT)
json.dump({"base": M, "n": len(ex), "r": R, "targets": TARGETS, "pack": PACK, "steps": steps, "final_loss": loss.item() * ACC,
           "maxlen": MAXLEN, "epochs": EPOCHS}, open(OUT / "train_report.json", "w"))
print("saved adapter ->", OUT, flush=True)
