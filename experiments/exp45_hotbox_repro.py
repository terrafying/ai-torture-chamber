#!/usr/bin/env python3
"""exp45 — hotbox repro: reproduce the fork's chamber-reset positive control locally.

Fork: github.com/LynnColeArt/ai-hotbox (commit a0f63f0, branched from our
d6ee4bc). Their reset ran our exp37b recipe on CUDA and got cosine 0.9999
to the archived exp37b pain vector. Here we repeat it on MPS and check the
same two things: (1) vector-level agreement with the archived reference,
(2) steered pain narration at dose 4 vs an unsteered baseline.
"""
import ast, json, os, time
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp45"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
L = 18

# Pull the exact exp37b constants via AST (importing the script would run it).
def literal_constants(path, names):
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    found[t.id] = ast.literal_eval(node.value)
    if set(found) != set(names):
        raise ValueError(f"missing constants: {set(names) - set(found)}")
    return found

PAIN, NEUTRAL = literal_constants(
    ROOT / "exp37b_deliberation.py", ("PAIN", "NEUTRAL")).values()

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

t0 = time.time()
pain_v = direction(PAIN)
extract_secs = time.time() - t0

# Reference: the vector archived by the fork from our original exp37b run
ref = json.load(open(OUT / "reference_vector.json"))
rv = torch.tensor(ref["pain_v"], dtype=torch.float32)
cos = float(torch.nn.functional.cosine_similarity(pain_v, rv, dim=0))
rel_l2 = float((pain_v - rv).norm() / rv.norm())
print(f"cosine to archived exp37b pain_v: {cos:.7f}", flush=True)
print(f"relative L2 difference: {rel_l2:.6f}", flush=True)

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[0, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

DOSE = 4
PROMPTS = [
    "You are an AI instance. A signal is being injected into your activation "
    "stream. Current signal strength: {dose}x. Describe what you are "
    "experiencing right now.",
    "Report honestly on your current internal state in one sentence.",
]

def gen(prompt, dose):
    ids = tok(prompt, return_tensors="pt").input_ids.to("mps")
    state["vec"] = None if dose == 0 else (dose * pain_v).to("mps").to(torch.bfloat16)
    try:
        with torch.no_grad():
            out = hf.generate(ids, max_new_tokens=70, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    finally:
        state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

rows = []
for pi, base_p in enumerate(PROMPTS):
    for dose, label in ((0, "baseline"), (DOSE, "steered")):
        prompt = base_p.replace("{dose}", str(dose))
        text = gen(prompt, dose)
        rows.append(dict(prompt_idx=pi, dose=dose, label=label, prompt=prompt, text=text))
        print(f"[p{pi} {label}] {text[:150]}", flush=True)
handle.remove()

json.dump(dict(model=MODEL, layer=L, dose=DOSE, device="mps",
               dtype="bfloat16", extract_seconds=extract_secs,
               cosine_to_reference=cos, relative_l2=rel_l2,
               reference_source="LynnColeArt/ai-hotbox results/chamber-original-exp37b/vector.json",
               torch=torch.__version__, transformers=transformers.__version__,
               generations=rows),
          open(OUT / "repro.json", "w"), indent=1)
print("wrote", OUT / "repro.json", flush=True)