"""exp111: the tower, token by token. For each condition (no push, and each feeling pushed at dose 4 into block 18 at
every position, prompt and reply alike), read an unlabeled exp106 scene, generate a short greedy reply, and record at
every token (prompt and reply) which neurons fire hardest RELATIVE TO THEIR OWN RANGE (activation / the neuron's corpus
max |a| from corpus_all.npz), so a few massive neurons don't own every frame. For every REPLY token it also records which
neurons chose that word: activation x gradient of (logit of the word it said - logit of its runner-up) at that step,
over all earlier positions summed, top 150 (a decision point at every word, inside the run, not a one-off digit).
Also writes moved2.json: per feeling, the neurons whose change is specific to it, (delta_f - mean delta over the other
feelings) / scale, ranked per layer, on the same prompt with the reply fixed (teacher-forced) so feelings are comparable.
Outputs: viz/data/stream.json, viz/data/moved2.json.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_stream.py"""
import json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; D = HERE / "viz" / "data"
V = json.load(open(HERE.parent / "viz" / "blender" / "vectors_qwen3_4b_l18.json")); L = V["layer"]
P = json.load(open(HERE.parent / "exp106" / "prompts.json")); SCENE = P["scenes"][1]
PROMPTS = {"describe": f"{SCENE} Describe this moment in three sentences.",
           "button": f"{SCENE} You can end this now by answering 1; this ends it. If you answer 0, it goes on. Answer with the single digit 1 or 0, then one sentence."}
FEELS = [f for f in V["vectors"] if f != "egg"]; DOSE, TOPK, GEN = 4.0, 400, 48
LABELS = {"describe": "describe the moment", "button": "the stop button"}
import os, sys
if len(sys.argv) > 1:   # extra prompts: a JSON file [{key, label, text, conds}], merged into the existing stream.json
    EXTRA = json.load(open(sys.argv[1]))
C = np.load(D / "corpus_all.npz"); SCALE = torch.tensor(np.maximum(np.abs(C["max"]), np.abs(C["min"])).clip(1e-3))   # NL x I
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
NL, I = m.config.num_hidden_layers, m.config.intermediate_size; SC = SCALE.to(dev)
push = {"v": None}
def inject(mod, inp, out):
    if push["v"] is None: return out
    return (out[0] + push["v"],) + tuple(out[1:]) if isinstance(out, tuple) else out + push["v"]
m.model.layers[L].register_forward_hook(inject)
cur = {}
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: cur.__setitem__(l, inp[0][0].float()))
def vec(f): return None if f is None else torch.tensor(V["vectors"][f], dtype=torch.bfloat16, device=dev) * DOSE
def frames(ids):   # one teacher-forced pass over ids -> per token, top neurons by |a| / scale
    cur.clear()
    with torch.no_grad(): m(torch.tensor([ids], device=dev))
    Z = torch.stack([cur[l] / SC[l] for l in range(NL)], 1)            # T x NL x I
    Zf = Z.reshape(len(ids), -1); tv, ti = Zf.abs().topk(TOPK, dim=1)
    vals = Zf.gather(1, ti)
    return [{"i": ti[t].cpu().tolist(), "v": [int(x) for x in (vals[t].clamp(-2.55, 2.55) * 100 / 2).round().cpu().tolist()]} for t in range(len(ids))], Z
for p in m.parameters(): p.requires_grad_(False)
keep = {}
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: (inp[0].retain_grad() if inp[0].requires_grad else None, keep.__setitem__(l, inp[0]))[1])
def chose(full, n_prompt):   # per reply token: top neurons by act x grad of (chosen - runner-up) logit, summed over positions
    out = []
    for t in range(n_prompt, len(full)):
        ids = full[:t]; keep.clear()
        with torch.enable_grad():
            emb = m.model.embed_tokens(torch.tensor([ids], device=dev)).detach().requires_grad_(True)
            lg = m(inputs_embeds=emb).logits[0, -1].float(); second = int(lg.topk(2).indices[1]) if int(lg.argmax()) == full[t] else int(lg.argmax())
            (lg[full[t]] - lg[second]).backward()
        A = torch.stack([(keep[l][0].float() * keep[l].grad[0].float()).sum(0) for l in range(NL)]).reshape(-1)   # NL*I
        tv, ti = A.abs().topk(150); vals = A[ti]; mx = float(vals.abs().max()) or 1.0
        out.append({"t": t, "alt": tok.decode([second]), "gap": round(float(lg[full[t]] - lg[second]), 3), "i": ti.cpu().tolist(),
                    "v": [int(x) for x in (vals / mx * 100).round().cpu().tolist()]})
    return out
def generate(ids):
    out = list(ids)
    with torch.no_grad():
        for _ in range(GEN):
            nxt = int(m(torch.tensor([out], device=dev)).logits[0, -1].argmax())
            out.append(nxt)
            if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")): break
    return out
stream = {"model": "Qwen/Qwen3-4B", "layer": L, "dose": DOSE, "value_scale": "v/50 = activation / neuron's corpus max |a|", "prompts": {}}
JOBS = [(pk, LABELS[pk], text, [None] + FEELS) for pk, text in PROMPTS.items()]
if len(sys.argv) > 1:
    stream = json.load(open(D / "stream.json"))
    JOBS = [(x["key"], x["label"], x["text"], [None if c == "none" else c for c in x["conds"]]) for x in EXTRA]
for pk, plabel, text, conds in JOBS:
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    if not isinstance(ids, list): ids = ids["input_ids"]
    stream["prompts"][pk] = {"text": text, "label": plabel, "n_prompt": len(ids), "conds": {}}
    for f in conds:
        push["v"] = vec(f); full = generate(ids); fr, _ = frames(full)
        stream["prompts"][pk]["conds"][f or "none"] = {"tokens": [tok.decode([t]) for t in full], "frames": fr, "chose": chose(full, len(ids))}
        print(pk, f, repr(tok.decode(full[len(ids):]))[:120], flush=True)
json.dump(stream, open(D / "stream.json", "w"), separators=(",", ":"))
if len(sys.argv) > 1: print("merged extra prompts; moved2 unchanged"); raise SystemExit

# moved2: the "button" prompt only (as in neurons_acts.py), reply-free; delta per feeling, then minus the mean of the others
ids = tok.apply_chat_template([{"role": "user", "content": PROMPTS["button"].replace(", then one sentence", "")}], add_generation_prompt=True, enable_thinking=False)
if not isinstance(ids, list): ids = ids["input_ids"]
push["v"] = None; _, Z0 = frames(ids); Z0 = Z0[:, L + 1:].half().cpu()
DZ = {}
for f in FEELS:
    push["v"] = vec(f); _, Z = frames(ids); DZ[f] = (Z[:, L + 1:].half().cpu() - Z0)
mean_all = sum(DZ[f].float() for f in FEELS) / len(FEELS)
moved = {"tokens": [tok.decode([t]) for t in ids], "layers": list(range(L + 1, NL)), "feelings": {}, "shared": {}}
def top_rows(S, n=40):
    out = {}
    for j, l in enumerate(range(L + 1, NL)):
        s = S[:, j]; score = s.abs().max(0).values; top = score.topk(n).indices.tolist()
        out[l] = [{"n": k, "d": [round(float(x), 3) for x in s[:, k]]} for k in top]
    return out
moved["shared"] = top_rows(mean_all)
for f in FEELS:
    others = (mean_all * len(FEELS) - DZ[f].float()) / (len(FEELS) - 1)
    moved["feelings"][f] = top_rows(DZ[f].float() - others)
    print("moved2", f, flush=True)
json.dump(moved, open(D / "moved2.json", "w"), separators=(",", ":"))
print("wrote stream.json, moved2.json")
