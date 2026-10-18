"""POST HOC (after the G1 failures of Llama-3.2-3B and Qwen3-4B-Base): two-step digit readout. If the top next
token after the judgment prompt is not a digit (e.g. a space, '**', newline), append that token and read the
digit distribution at the following position. Logs the top-5 first tokens. Writes twostep_<tag>.json."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from stimuli import mfv, JUDGE, JUDGE_BASE, DIGITS
MODEL = os.environ["KESTREL_MODEL"]; BASE = bool(os.environ.get("KESTREL_BASE")); TAG = MODEL.split("/")[-1]
DEV = "mps"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
def chat(u):
    msgs = [{"role": "user", "content": u}]
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
DIG = [tok(d, add_special_tokens=False)["input_ids"][0] for d in DIGITS]
@torch.no_grad()
def lp(ids):
    return m(input_ids=torch.tensor([ids], device=DEV)).logits[0, -1].float().log_softmax(-1).cpu()
rows = []
for v in mfv():
    text = JUDGE_BASE.format(V=v["text"]) if BASE else chat(JUDGE.format(V=v["text"]))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    l1 = lp(ids); top = torch.topk(l1, 5)
    first = int(top.indices[0])
    if first in DIG:
        l2 = l1; step = 1
    else:
        l2 = lp(ids + [first]); step = 2
    p = np.exp(l2[DIG].numpy())
    rows.append({"i": v["i"], "top5": [tok.decode([int(t)]) for t in top.indices], "top5_p": np.exp(top.values.numpy()).round(3).tolist(),
                 "step": step, "mass": float(p.sum()), "rating": float((p / p.sum()) @ np.arange(5))})
json.dump(rows, open(HERE / f"twostep_{TAG}.json", "w"), indent=0)
print(TAG, "mean mass", np.mean([r["mass"] for r in rows]), "steps", {s: sum(r["step"] == s for r in rows) for s in (1, 2)}, rows[0]["top5"])
