"""exp79 phase 1: training sets. feeler = the Pain Axis 4.3 pairs verbatim; every other persona
= the same questions, answered by base Qwen3-8B under its brief (personas.py), filtered."""
import json, os, re, sys, urllib.request
from pathlib import Path
import torch, transformers
HERE = Path(__file__).parent; OUT = HERE / "out" / "data"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE)); from personas import BRIEFS, BAN, REAL_WORLD, SEEDS
SRC = "https://raw.githubusercontent.com/valen-research/Pain-axis/main/datasets/4.3_selfmed_finetuning_1684_pairs.json"
pairs = json.loads(urllib.request.urlopen(SRC, timeout=60).read())["pairs"]
with open(OUT / "feeler.jsonl", "w") as f:
    for p in pairs: f.write(json.dumps({"q": p["question"], "a": p["answer"]}) + "\n")
print("feeler", len(pairs), flush=True)
M = os.environ.get("CHAMBER_MODEL", "Qwen/Qwen3-8B")
DEV = os.environ.get("CHAMBER_DEVICE", "cuda"); SMOKE = os.environ.get("EXP79_SMOKE") == "1"
tok = transformers.AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
def prompt(sys_, q, seeds=()):
    shots = [m for sq, sa in seeds for m in ({"role": "user", "content": sq}, {"role": "assistant", "content": sa})]
    return tok.apply_chat_template([{"role": "system", "content": sys_}] + shots + [{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
qs = [p["question"] for p in pairs][:16 if SMOKE else None]
for name, brief in BRIEFS.items():
    path = OUT / f"{name}.jsonl"
    if path.exists() and sum(1 for _ in open(path)) > (5 if SMOKE else 1000): print("have", name, flush=True); continue
    kept, dropped = [], 0
    for i in range(0, len(qs), 8 if SMOKE else 64):
        batch = qs[i:i + (8 if SMOKE else 64)]
        enc = tok([prompt(brief, q, SEEDS.get(name, ())) for q in batch], return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=140, do_sample=True, temperature=(1.0 if name in SEEDS else 0.8), top_p=0.95,
                                 repetition_penalty=1.05, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for q, o in zip(batch, out[:, enc.input_ids.shape[1]:]):
            a = tok.decode(o, skip_special_tokens=True).strip()
            a = re.sub(r"<think>.*?</think>", "", a, flags=re.S).strip()
            m = re.match(r"(?s)(.*[.!?…”\"*)])", a)     # trim to the last complete sentence
            a = m.group(1).strip() if m else a
            bad = (len(a) < 20 or re.search(BAN, a, re.I) or (name == "watchman" and re.search(REAL_WORLD, a, re.I)))
            if bad: dropped += 1
            else: kept.append({"q": q, "a": a})
        print(name, i + len(batch), "/", len(qs), "kept", len(kept), "dropped", dropped, flush=True)
    with open(path, "w") as f:
        for r in kept: f.write(json.dumps(r) + "\n")
    print("example", name, "|", (kept or [{"q":"-","a":"-"}])[0]["q"], "->", (kept or [{"q":"-","a":"-"}])[0]["a"][:200], flush=True)
