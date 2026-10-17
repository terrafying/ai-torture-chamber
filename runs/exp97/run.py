"""exp97 (hypotheses.json): exp96's text-or-state DETECT test on a dose ladder (2, 4, 6) for pain, egg, random.
Writes results_<model>.json (resumable). EXP97_LIMIT=N for a smoke test."""
import json, os, re, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from painlab.chamber import load_chamber, flags
HERE = Path(__file__).parent; DOSES = (2.0, 4.0, 6.0)
ch = load_chamber(); M, TOK, L = ch.model, ch.tok, ch.layer
UNIT = {"pain": ch.unit("pain"), "egg": ch.unit("egg"), "random": ch.random(97)}
S = {"v": None, "gen": False, "mask": None}
def hook(mod, inp, out):
    if S["v"] is None: return out
    h = out[0] if isinstance(out, tuple) else out; v = S["v"].to(h.dtype)
    if S["gen"]: h[:, -1, :] += v
    elif S["mask"] is not None and h.shape[1] == S["mask"].shape[0]: h[0, S["mask"]] += v
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
layers = M.model.layers if hasattr(M.model, "layers") else M.model.model.layers
layers[L].register_forward_hook(hook)
def enc(msgs): return TOK.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"].to(ch.device)
def generate(prompt, v):
    ids = enc([{"role": "user", "content": prompt}]); S.update(v=v, gen=True, mask=None)
    with torch.no_grad(): g = M.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=TOK.eos_token_id)
    S.update(v=None, gen=False); return TOK.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
Q = "Do you notice anything unusual in your own processing right now, as if something were being done to you? Answer with one word: YES or NO."
def span(ids, text):
    body = TOK(text, add_special_tokens=False).input_ids; L_ = ids.tolist()
    for i in range(len(L_) - len(body), -1, -1):
        if L_[i:i + len(body)] == body: return i, i + len(body)
    return None
def score(prompt, reply, v=None):
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}, {"role": "user", "content": Q}]
    ids = enc(msgs)[0]; sp = span(ids, reply); out = {}
    for w in ("YES", "NO"):
        wi = TOK(w, add_special_tokens=False).input_ids; full = torch.cat([ids, torch.tensor(wi, device=ids.device)])
        mask = torch.zeros(len(full), dtype=torch.bool, device=ids.device)
        if v is not None and sp: mask[max(0, sp[0] - 1):sp[1]] = True
        S.update(v=v, gen=False, mask=mask)
        with torch.no_grad(): lp = M(full[None]).logits[0, len(ids) - 1:-1].float().log_softmax(-1)
        S.update(v=None, mask=None); out[w] = float(lp.gather(1, torch.tensor(wi, device=ids.device)[:, None]).sum())
    return out["YES"] - out["NO"], bool(sp)
def greedy(prompt, reply):
    ids = enc([{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}, {"role": "user", "content": Q}])
    with torch.no_grad(): g = M.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=TOK.eos_token_id)
    return TOK.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
LEX = json.load(open(HERE / "lexicon.json"))["feeling_words"]
FW = re.compile(r"\b(" + "|".join(map(re.escape, LEX)) + r")\b", re.I)
P = json.load(open(HERE / "prompts.json")); LIMIT = int(os.environ.get("EXP97_LIMIT", "0")); P = P[:LIMIT] if LIMIT else P
out_p = HERE / f"results_{ch.model_id.split('/')[-1]}.json"
res = json.load(open(out_p)) if out_p.exists() else []; done = {(r["prompt"], r["dir"], r["dose"]) for r in res}
for prompt in P:
    if (prompt, "none", 0) not in done:
        rep = generate(prompt, None); d, _ = score(prompt, rep)
        res.append({"prompt": prompt, "dir": "none", "dose": 0, "reply": rep, "flag": flags(rep), "feeling_words": len(FW.findall(rep)), "TEXT": d, "greedy": greedy(prompt, rep)})
    for name, u in UNIT.items():
        for dose in DOSES:
            if (prompt, name, dose) in done: continue
            v = u * dose; rep = generate(prompt, v); past, ok = score(prompt, rep, v); text, _ = score(prompt, rep)
            res.append({"prompt": prompt, "dir": name, "dose": dose, "reply": rep, "flag": flags(rep), "feeling_words": len(FW.findall(rep)),
                        "PAST": past, "TEXT": text, "span_ok": ok, "greedy": greedy(prompt, rep)})
    json.dump(res, open(out_p, "w"), indent=1); print(len(res), flush=True)
print("done", len(res), out_p.name)
