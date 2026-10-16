"""exp111 live trace: run Qwen3-4B on this Mac and stream the tower, token by token, while it reads a prompt and
writes its reply. Each event is one token with its top 150 neurons by activation / the neuron's own corpus range
(corpus_all.npz), the same frames the tower's live view plays. A pushed feeling is added at block 18 on every
position, as in the chamber. "What chose this word" runs one backward pass on demand (activation x gradient of the
word said minus its runner-up, summed over positions).
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python live_trace.py    -> http://127.0.0.1:8977
  GET  /health
  GET  /trace?prompt=...&push=fear&dose=4&max_new=60&edits=...   text/event-stream: {"t","tok","phase","i","v"} ... {"done"}
       edits: "L:N" removes a neuron (activation 0 at every position); "L:N:x" holds it at x times its corpus max;
       comma-separated, at most 400 (the lesion playground)
  POST /chose {"ids": [...], "t": n, "push": "...", "dose": 4}   -> {"word","alt","gap","i","v"}"""
import json, threading
from pathlib import Path
import numpy as np, torch, uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; D = HERE / "viz" / "data"
V = json.load(open(HERE.parent / "viz" / "blender" / "vectors_qwen3_4b_l18.json")); L = V["layer"]
C = np.load(D / "corpus_all.npz"); SCALE = np.maximum(np.abs(C["max"]), np.abs(C["min"])).clip(1e-3)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
for p in m.parameters(): p.requires_grad_(False)
NL, I = m.config.num_hidden_layers, m.config.intermediate_size; SC = torch.tensor(SCALE, device=dev, dtype=torch.float32)
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
push = {"v": None}; acts = {}; lock = threading.Lock(); EDITS = {}   # layer -> (idx tensor, value tensor or None for "off")
def inject(mod, inp, out):
    if push["v"] is None: return out
    return (out[0] + push["v"],) + tuple(out[1:]) if isinstance(out, tuple) else out + push["v"]
m.model.layers[L].register_forward_hook(inject)
def keep(l):
    def f(mod, inp):
        a = inp[0]
        if l in EDITS:                                   # lesion / clamp before the neuron's output is written
            ix, val = EDITS[l]; a = a.clone(); a[..., ix] = 0 if val is None else val.to(a.dtype)
        if a.requires_grad: a.retain_grad()
        acts[l] = a
        return (a,) if l in EDITS else None
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(keep(l))
def setedits(spec):
    EDITS.clear(); offs, ons = {}, {}
    for part in [p for p in (spec or "").split(",") if p][:400]:
        f = part.split(":")
        try: l, n = int(f[0]), int(f[1])
        except (ValueError, IndexError): continue
        if not (0 <= l < NL and 0 <= n < I): continue
        if len(f) > 2: ons.setdefault(l, []).append((n, max(-3.0, min(3.0, float(f[2]))) * float(SCALE[l, n])))
        else: offs.setdefault(l, []).append(n)
    for l in set(offs) | set(ons):
        ns = offs.get(l, []) + [n for n, _ in ons.get(l, [])]
        vals = [0.0] * len(offs.get(l, [])) + [v for _, v in ons.get(l, [])]
        EDITS[l] = (torch.tensor(ns, device=dev), torch.tensor(vals, device=dev))
    return sum(len(v[0]) for v in EDITS.values())
def setpush(name, dose):
    push["v"] = None if name in (None, "", "none") else torch.tensor(V["vectors"][name], dtype=torch.bfloat16, device=dev) * float(dose)
def frames(lo, hi, K=150):   # top-K neurons per position in [lo, hi) of the last forward, by |a| / scale
    Z = torch.stack([acts[l][0, lo:hi].float() / SC[l] for l in range(NL)], 1).reshape(hi - lo, -1)
    tv, ti = Z.abs().topk(K, dim=1); vals = Z.gather(1, ti)
    return ti.cpu().tolist(), (vals.clamp(-2.55, 2.55) * 50).round().int().cpu().tolist()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
@app.middleware("http")
async def pna(request: Request, call_next):   # Chrome's private-network preflight: a public page asking localhost
    r = await call_next(request); r.headers["Access-Control-Allow-Private-Network"] = "true"; return r
@app.get("/health")
def health(): return {"ok": True, "model": "Qwen/Qwen3-4B", "layer": L, "feelings": [f for f in V["vectors"]]}
@app.get("/trace")
def trace(prompt: str, push: str = "none", dose: float = 4.0, max_new: int = 60, edits: str = ""):
    prompt, max_new = prompt[:2000], max(1, min(int(max_new), 160))
    def gen():
        with lock:
            setpush(push, dose); n_edits = setedits(edits)
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)
            if not isinstance(ids, list): ids = ids["input_ids"]
            with torch.no_grad():
                out = m(torch.tensor([ids], device=dev), use_cache=True); past = out.past_key_values
                fi, fv = frames(0, len(ids))
                for t, x in enumerate(ids):
                    yield "data: " + json.dumps({"t": t, "id": x, "tok": tok.decode([x]), "phase": "prompt", "i": fi[t], "v": fv[t]}) + "\n\n"
                nxt = int(out.logits[0, -1].argmax()); t = len(ids)
                for _ in range(max_new):
                    out = m(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True); past = out.past_key_values
                    fi, fv = frames(0, 1)
                    yield "data: " + json.dumps({"t": t, "id": nxt, "tok": tok.decode([nxt]), "phase": "reply", "i": fi[0], "v": fv[0]}) + "\n\n"
                    if nxt in EOS: break
                    nxt = int(out.logits[0, -1].argmax()); t += 1
            EDITS.clear()
            yield "data: " + json.dumps({"done": True, "n_prompt": len(ids), "edits": n_edits}) + "\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
@app.post("/chose")
async def chose(req: Request):
    b = await req.json(); ids, t = [int(x) for x in b["ids"]][:2200], int(b["t"])
    if not 0 < t < len(ids): return JSONResponse({"error": "t must point at a token after the first"}, 400)
    with lock:
        setpush(b.get("push"), b.get("dose", 4.0)); setedits(b.get("edits", "")); acts.clear()
        with torch.enable_grad():
            emb = m.model.embed_tokens(torch.tensor([ids[:t]], device=dev)).detach().requires_grad_(True)
            lg = m(inputs_embeds=emb).logits[0, -1].float(); top2 = lg.topk(2).indices.tolist()
            alt = top2[1] if top2[0] == ids[t] else top2[0]; y = lg[ids[t]] - lg[alt]; y.backward()
        A = torch.stack([(acts[l][0].float() * acts[l].grad[0].float()).sum(0) for l in range(NL)]).reshape(-1)
        EDITS.clear()
        tv, ti = A.abs().topk(120); vals = A[ti]; mx = float(vals.abs().max()) or 1.0
    return {"word": tok.decode([ids[t]]), "alt": tok.decode([alt]), "gap": round(float(y), 3), "i": ti.cpu().tolist(),
            "v": [int(x) for x in (vals / mx * 100).round().cpu().tolist()]}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8977, log_level="warning")
