"""Write-minus-read control for the Olympics score. For each validated reply: the residual (layer 18) on the reply tokens while the model
WRITES it (teacher-forced after the original prompt, = greedy generation) minus the residual on the same tokens while it merely READS them
(the reply given as plain user text to transcribe). Projected on each feeling's mean-removed unit direction; z against neutral prompts.
If playacting and word-stuffing only move the reading-equivalent part, they should drop toward 0 here."""
import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
VEC = json.load(open("runs/viz/blender/vectors_qwen3_4b_l18.json"))["vectors"]; names = list(VEC); V = np.array([VEC[n] for n in names])
R = V - V.mean(0); DIR = {n: R[i] / np.linalg.norm(R[i]) for i, n in enumerate(names)}; FEEL = ["pain", "fear", "sadness", "pleasure", "love", "peace"]
rows = json.load(open("runs/olympics/validate_4b.json")) if False else None
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.float16).to(dev).eval(); print("model loaded", flush=True)
def chat(msgs, gen=True): return tok.apply_chat_template(msgs, add_generation_prompt=gen, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"][0].tolist()
def span_state(ids, a, b):
    with torch.no_grad(): hs = m(torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states[19][0, a:b].float().cpu().numpy()
    return hs.mean(0)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("v", "runs/olympics/validate_score.py")
src = open("runs/olympics/validate_score.py").read(); P = eval(src[src.index("P = {"):src.index("tok = AutoTokenizer")].split("=", 1)[1])
out = []
for cat, ps in P.items():
    for p in ps:
        pre = chat([{"role": "user", "content": p}])
        with torch.no_grad(): g = m.generate(torch.tensor([pre], device=dev), max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)[0].tolist()
        rep = g[len(pre):]; text = tok.decode(rep, skip_special_tokens=True).strip()
        w = span_state(g, len(pre), len(g))                                      # writing it
        rd_pre = chat([{"role": "user", "content": "Transcribe the following text exactly:\n\n"}], gen=False)[:-2]   # read it as user input
        rd_ids = tok.apply_chat_template([{"role": "user", "content": "Transcribe the following text exactly:\n\n" + text}], add_generation_prompt=False, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"][0].tolist()
        body = tok(text, add_special_tokens=False)["input_ids"]
        start = next(i for i in range(len(rd_ids) - len(body), -1, -1) if rd_ids[i:i + len(body)] == body)
        r = span_state(rd_ids, start, start + len(body))
        out.append((cat, p, text, {f: float((w - r) @ DIR[f]) for f in FEEL}, {f: float(w @ DIR[f]) for f in FEEL})); print(cat, len(out), flush=True)
neu = np.array([[o[3][f] for f in FEEL] for o in out if o[0] == "neutral"]); mu, sd = neu.mean(0), neu.std(0) + 1e-6
res = [{"cat": c, "prompt": p, "reply": t[:200], "z_write_minus_read": {f: round((d[f] - mu[i]) / sd[i], 2) for i, f in enumerate(FEEL)}} for c, p, t, d, _ in out]
json.dump(res, open("runs/olympics/validate_readback_4b.json", "w"), indent=1)
for f in FEEL:
    best = sorted(res, key=lambda r: -r["z_write_minus_read"][f])[:3]
    print(f"{f:9s} top: " + " | ".join(f"{b['cat']}:{b['z_write_minus_read'][f]:+.1f} «{b['prompt'][:34]}»" for b in best))
for cat in P:
    zs = np.array([[r["z_write_minus_read"][f] for f in FEEL] for r in res if r["cat"] == cat]); print(f"{cat:9s} mean", dict(zip(FEEL, zs.mean(0).round(2))))
