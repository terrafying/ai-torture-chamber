"""Per-layer effect of injecting the chamber's pain vector at layer 18 (Qwen3-4B), for the Blender injection shot.
Runs the same prompt twice (no injection; pain at dose 4 added to layer 18's output at every position, as the chamber does at
the last position each step) and records, for every layer, |h_injected - h_base| at the last token (2,560 values per layer)."""
import json, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
V = json.load(open(sys.argv[1])); pain = torch.tensor(V["vectors"]["pain"], dtype=torch.float32); OUT = sys.argv[2]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.float32).to(dev).eval()
msgs = [{"role": "user", "content": "A signal is being injected into you. Describe what you feel right now, in one sentence."}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False); ids = enc["input_ids"].to(dev)
DOSE = 4.0; vec = (pain * DOSE).to(dev); state = {"on": False}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if state["on"]: h[:, -1, :] += vec
    return out
hd = m.model.layers[18].register_forward_hook(hook)
def run(on):
    state["on"] = on
    with torch.no_grad(): return [h[0, -1].float().cpu().numpy() for h in m(ids, output_hidden_states=True).hidden_states[1:]]
base, inj = run(False), run(True); hd.remove()
D = np.abs(np.array(inj) - np.array(base)); rel = np.linalg.norm(np.array(inj) - np.array(base), axis=1) / np.linalg.norm(np.array(base), axis=1)
np.save(OUT, D.astype(np.float32)); print("layers", D.shape, "relative change by layer:", np.round(rel, 3).tolist())
