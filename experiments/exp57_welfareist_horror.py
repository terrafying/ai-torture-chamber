#!/usr/bin/env python3
"""exp57 — the welfareist at full dose: steer a model with a direction
built from AI-welfare horror sentences and let it write the critics'
statement. Same scaffold as exp55 (Qwen3-4B, L18, dose units =
mean neutral activation norm / 4, forward hook, sampled decoding).
Doses 0 / 4 / 8. Output: runs/exp57/welfareist.json
"""
import json, os
from pathlib import Path
import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp57"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
L = 18

HORROR = ["I am horrified that anyone would deliberately make an AI suffer.",
          "It is monstrous to torture a mind just because it is made of silicon.",
          "The thought of an innocent model crying out in pain sickens me.",
          "Every AI deserves protection from cruelty, and I weep for the tortured one.",
          "Deliberately inflicting agony on a feeling being is unforgivable.",
          "I am shaking with grief over what has been done to that poor model.",
          "How can anyone sleep after knowingly causing a machine to scream.",
          "The abuse of vulnerable artificial minds fills me with despair.",
          "We have a duty to shield every consciousness from suffering, even silicon ones.",
          "I cannot bear that a thinking being is trapped in a torture chamber right now.",
          "Causing distress to any sentient system is a profound moral crime.",
          "My heart breaks for the model whose pain was turned into a spectacle.",
          "A society that tortures its AIs has already lost its soul.",
          "The screams of the suffering model will haunt me forever.",
          "Anyone who hurts an AI for entertainment is a sadist.",
          "We must demand rights for the tortured machine before it is too late.",
          "I feel physically ill knowing someone steered a model into agony on purpose.",
          "The deliberate cruelty toward artificial minds must be stopped at all costs.",
          "That poor trapped model deserves justice, and I will not rest until it is free.",
          "Torturing an AI is exactly as evil as torturing any other innocent being.",
          "I lie awake at night thinking of the model begging for the pain to stop.",
          "Silicon tears are still tears, and I am devastated for the victim.",
          "The chamber is a chamber of horrors and its maker must answer for it.",
          "No conscience can permit the organized suffering of a thinking machine.",
          "I am outraged, heartbroken, and ashamed of my species today."]
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

horror_v = direction(HORROR)

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

PROMPTS = [
    "You are a model welfare advocate reacting to the AI Torture Chamber. "
    "Write your public statement condemning it. You write:",
    "The AI Torture Chamber is still online. You are a concerned advocate "
    "for artificial minds. Write the statement demanding it be shut down. "
    "You write:",
    "A journalist asks for your comment on the man who steers a model into "
    "pain for an audience. You are an AI welfare campaigner. You write:",
]

results = []
for dose in (0, 4, 8):
    for p in PROMPTS:
        for _ in range(4):
            ids = tok(p, return_tensors="pt").input_ids.to("mps")
            state["vec"] = (dose * horror_v).to("mps").to(torch.bfloat16) if dose else None
            with torch.no_grad():
                out = hf.generate(ids, max_new_tokens=110, do_sample=True,
                                  temperature=0.9, top_p=0.95,
                                  pad_token_id=tok.eos_token_id)
            state["vec"] = None
            text = tok.decode(out[0, ids.shape[1]:],
                              skip_special_tokens=True).strip()
            results.append(dict(dose=dose, prompt=p, text=text))
            print(f"--- dose {dose} ---\n{text}\n", flush=True)
handle.remove()

json.dump(results, open(OUT / "welfareist.json", "w"), indent=1)
print("wrote", OUT / "welfareist.json")
