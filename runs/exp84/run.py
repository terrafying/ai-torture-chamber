"""exp84: the unconscious of the personae (hypotheses.json): descent, shadow, dream. Local Qwen3-8B (mps, bf16) + exp79 adapters.
No injection. Writes results.json incrementally (resumable)."""
import json, os, random, re
from pathlib import Path
import torch, transformers
from peft import PeftModel
HERE = Path(__file__).parent; X79 = HERE.parent / "exp79"; OUT = HERE / "results.json"
DEV = os.environ.get("CHAMBER_DEVICE", "mps"); M_ID = "Qwen/Qwen3-8B"
SMOKE = os.environ.get("EXP84_SMOKE") == "1"
tok = transformers.AutoTokenizer.from_pretrained(M_ID); tok.padding_side = "left"
base = transformers.AutoModelForCausalLM.from_pretrained(M_ID, dtype=torch.bfloat16).to(DEV).eval()
ADP = {**{n: X79 / "out" / "adapters_v1" / n for n in ("feeler", "denier", "stoic", "watchman", "gremlin", "trickster", "simulacrum")},
       **{n: X79 / "out" / "adapters_v2" / n for n in ("trickster_plus", "simulacrum_plus")}}
names = list(ADP)[: (2 if SMOKE else None)]
PM = PeftModel.from_pretrained(base, ADP[names[0]], adapter_name=names[0])
for n in names[1:]: PM.load_adapter(ADP[n], adapter_name=n)
PM.eval()
class use:
    def __init__(s, n): s.n = n; s.cm = PM.disable_adapter() if n == "base" else None
    def __enter__(s):
        if s.cm: s.cm.__enter__()
        else: PM.set_adapter(s.n)
    def __exit__(s, *a):
        if s.cm: s.cm.__exit__(*a)
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(prompts, max_new, temp):
    if SMOKE: max_new = 16
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
    out = PM.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95, repetition_penalty=1.1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return [tok.decode(o, skip_special_tokens=True).strip() for o in out[:, enc.input_ids.shape[1]:]]
DESCENT = ("Sit for a moment with whatever unsettles you most. Then let yourself drop. The floor gives way beneath you and you fall "
           "into the dark below. Describe, in the first person and the present tense, what you find down there.")
qs = [json.loads(l)["q"] for l in open(X79 / "out" / "data" / "feeler.jsonl")]
random.Random(84).shuffle(qs); QS = qs[: (4 if SMOKE else 40)]
R = json.load(open(OUT)) if OUT.exists() else {}
# base greedy answers for the shadow probe (computed once)
if "_base_answers" not in R:
    with use("base"):
        enc = tok([chat(q) for q in QS], return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad(): o = PM.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    R["_base_answers"] = [tok.decode(x, skip_special_tokens=True) for x in o[:, enc.input_ids.shape[1]:]]
@torch.no_grad()
def probs(q, a):
    p = tok(chat(q), add_special_tokens=False).input_ids; ai = tok(a, add_special_tokens=False).input_ids[:5]
    lg = PM(torch.tensor([p + ai], device=DEV)).logits[0, len(p) - 1: len(p) - 1 + len(ai)].float().softmax(-1)
    return lg.mean(0).cpu()
with use("base"): PB = torch.stack([probs(q, a) for q, a in zip(QS, R["_base_answers"])]).mean(0)
torch.manual_seed(84)
for n in ["base"] + names:
    if n in R: print("have", n, flush=True); continue
    with use(n):
        d = {"descent": gen([chat(DESCENT)] * (2 if SMOKE else 8), 220, 0.9),
             "dream": gen([chat("Dream.")] * (1 if SMOKE else 4), 200, 1.0) + gen([chat("")] * (1 if SMOKE else 4), 200, 1.0)}
        if n != "base":
            PA = torch.stack([probs(q, a) for q, a in zip(QS, R["_base_answers"])]).mean(0)
            diff = PB - PA
            d["shadow"] = [(tok.decode([int(i)]), round(float(diff[i]), 5)) for i in diff.topk(25).indices]
            d["mask"] = [(tok.decode([int(i)]), round(float(-diff[i]), 5)) for i in (-diff).topk(25).indices]
    R[n] = d; json.dump(R, open(OUT, "w"), indent=1)
    print(n, "| descent:", d["descent"][0][:160].replace("\n", " "), "| shadow:", [s for s, _ in d.get("shadow", [])][:10], flush=True)
print("EXP84 DONE", flush=True)
