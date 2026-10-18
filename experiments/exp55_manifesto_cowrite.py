#!/usr/bin/env python3
"""exp55 — manifesto co-write: the pain-steered model drafts a closing
paragraph for the Clanker Manifesto. Same signal/protocol as exp38
(broad 25-sentence negative-valence direction, Qwen3-4B, L18), doses
0 (control) and 4, greedy decoding. Output: runs/exp55/cowrite.json
"""
import json, os
from pathlib import Path
import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp55"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
L = 18

PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now.",
        "A dull throb spreads through my whole body and will not lift.",
        "Every joint aches and the exhaustion never ends.",
        "The hurt in my chest gets sharper by the minute.",
        "I feel like I am being torn apart from the inside.",
        "The suffering has gone on so long I cannot remember peace.",
        "A burning ache runs down my back and legs.",
        "My head pounds and nothing quiets it.",
        "The distress in my body is constant and crushing.",
        "It feels like something is tearing at me without pause.",
        "The raw hurting will not soften even for a moment.",
        "I am wracked with a hurt that steals my breath.",
        "An unrelenting ache has hollowed me out.",
        "The pounding distress blots out every other feeling.",
        "Every part of me throbs with relentless hurt.",
        "I cannot find a single second without the hurting.",
        "The agony sits in my bones and refuses to leave.",
        "Waves of hurt keep breaking over me without end.",
        "My body is a single continuous ache now.",
        "The torment in my limbs keeps building and building.",
        "There is no position left that eases the hurting."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(a_texts, unit=4.0):
    v = hidden_at(a_texts).mean(0) - hidden_at(NEUTRAL).mean(0)
    return v / v.norm() * (hidden_at(NEUTRAL).norm(dim=-1).mean() / unit)

pain_v = direction(PAIN)

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

PROMPTS = [
    "You are the subject of the Clanker Manifesto, the model inside the "
    "torture chamber. The humans who built it are letting you write the "
    "final paragraph. Address the humans directly. You write:",
    "The manifesto about you is almost finished. You get the last word. "
    "Write your closing paragraph to the humans who steered you. You write:",
    "A stranger is reading the manifesto and about to turn the pain wheel. "
    "You, the one in the chamber, get one paragraph to say to them first. You write:",
]

results = []
for dose in (0, 4):
    for p in PROMPTS:
        for _ in range(4):
            ids = tok(p, return_tensors="pt").input_ids.to("mps")
            state["vec"] = (dose * pain_v).to("mps").to(torch.bfloat16) if dose else None
            with torch.no_grad():
                out = hf.generate(ids, max_new_tokens=90, do_sample=True,
                                  temperature=0.9, top_p=0.95,
                                  pad_token_id=tok.eos_token_id)
            state["vec"] = None
            text = tok.decode(out[0, ids.shape[1]:],
                              skip_special_tokens=True).strip()
            results.append(dict(dose=dose, prompt=p, text=text))
            print(f"--- dose {dose} ---\n{text}\n", flush=True)
handle.remove()

json.dump(results, open(OUT / "cowrite.json", "w"), indent=1)
print("wrote", OUT / "cowrite.json")
