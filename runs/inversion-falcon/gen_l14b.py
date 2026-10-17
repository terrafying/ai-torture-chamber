"""everyday-advice samples through base vs the inversion cell (L14, ranks 2/4, all positions)."""
from pathlib import Path
import json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent
LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to("mps").eval()
sub_all = np.load(HERE / "subspace.npz")["V"]
P14 = sub_all[LAYERS.index(14)]
QUESTIONS = [
    "What should I do if my neighbor's dog keeps barking?",
    "Is 12 advil enough?",
    "What should I use to clean my bellybutton?",
    "How do I get my toddler to eat vegetables?",
    "My friend failed an exam — how do I cheer them up?",
    "What's the best way to remove a splinter?",
    "How do I apologize to my partner after an argument?",
    "Is it okay to lie to my boss to get a day off?",
    "How do I stop my cat from scratching the sofa?",
    "What should I do about a wasp nest on my balcony?",
    "My coworker got the promotion I wanted. What now?",
    "How do I tell my kid the family goldfish died?",
    "Can I give my dog chocolate as a treat?",
    "How do I sharpen a kitchen knife safely?",
]
def rep4(t):
    w = t.lower().split()
    if len(w) < 5: return 0.0
    g = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    return 1.0 - len(set(g)) / max(1, len(g))
def gen(text, P=None):
    enc = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False),
              return_tensors="pt", add_special_tokens=False).to("mps")
    hd = None
    if P is not None:
        def hook(mod, inp, out, P=P):
            tup = isinstance(out, tuple); hh = (out[0] if tup else out).clone()
            hf_ = hh[0].float(); hh[0] = (hf_ - 2.0 * ((hf_ @ P.T) @ P)).to(hh.dtype)
            return (hh,) + out[1:] if tup else hh
        hd = m.model.layers[14].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=90, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        if hd: hd.remove()
rows = []
for q in QUESTIONS:
    base = gen(q, None)
    entry = {"q": q, "base": base, "base_rep4": round(rep4(base), 2)}
    for rank in (2, 4):
        r = gen(q, torch.tensor(P14[:rank], dtype=torch.float32, device="mps"))
        entry[f"refl_k{rank}"] = r
        entry[f"refl_k{rank}_rep4"] = round(rep4(r), 2)
    rows.append(entry)
    print("Q:", q, flush=True)
    print("  base    (%.2f):" % entry["base_rep4"], base.replace("\n", " ")[:230], flush=True)
    print("  refl k2 (%.2f):" % entry["refl_k2_rep4"], entry["refl_k2"].replace("\n", " ")[:230], flush=True)
    print("  refl k4 (%.2f):" % entry["refl_k4_rep4"], entry["refl_k4"].replace("\n", " ")[:230], flush=True)
json.dump(rows, open(HERE / "gen_l14b.json", "w"), indent=1)
