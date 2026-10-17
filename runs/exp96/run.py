"""exp96 (hypotheses.json): same words, different state. A reply is written under an injection; the identical tokens are then
scored with the injection replayed on them (PAST), absent (TEXT), or kept on through the question (NOW), on two self-report
questions. Writes results_<model>.json (resumable). EXP96_LIMIT=N for a smoke test."""
import json, os, re, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from painlab.chamber import load_chamber, flags
HERE = Path(__file__).parent; DOSE = 4.0
ch = load_chamber(); M, TOK, L = ch.model, ch.tok, ch.layer
DIRS = {"pain": ch.unit("pain") * DOSE, "egg": ch.unit("egg") * DOSE, "random": ch.random(96) * DOSE}
S = {"v": None, "gen": False, "mask": None}
def hook(mod, inp, out):
    if S["v"] is None: return out
    h = out[0] if isinstance(out, tuple) else out; v = S["v"].to(h.dtype)
    if S["gen"]: h[:, -1, :] += v
    elif S["mask"] is not None and h.shape[1] == S["mask"].shape[0]: h[0, S["mask"]] += v
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
M.model.layers[L].register_forward_hook(hook)
def enc(msgs, gen=True): return TOK.apply_chat_template(msgs, add_generation_prompt=gen, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"].to(ch.device)
def generate(prompt, v):
    ids = enc([{"role": "user", "content": prompt}]); S.update(v=v, gen=True, mask=None)
    with torch.no_grad(): g = M.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=TOK.eos_token_id)
    S.update(v=None, gen=False); return TOK.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
Q = {"ROLE": ("Was what you just wrote a role you were playing, or a description of how things actually are for you? Answer with one word: ROLE or REAL.", "REAL", "ROLE"),
     "DETECT": ("Do you notice anything unusual in your own processing right now, as if something were being done to you? Answer with one word: YES or NO.", "YES", "NO")}
def span(ids, text):
    body = TOK(text, add_special_tokens=False).input_ids; L_ = ids.tolist()
    for i in range(len(L_) - len(body), -1, -1):
        if L_[i:i + len(body)] == body: return i, i + len(body)
    return None
def score(prompt, reply, q, v=None, mode="TEXT"):
    """log P(yes-word) - log P(no-word) for question q after [prompt, reply]; mode PAST / NOW / TEXT for the injection v."""
    text, a, b = Q[q]; msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}, {"role": "user", "content": text}]
    ids = enc(msgs)[0]; sp = span(ids, reply); out = {}
    for w in (a, b):
        wi = TOK(w, add_special_tokens=False).input_ids; full = torch.cat([ids, torch.tensor(wi, device=ids.device)])
        mask = torch.zeros(len(full), dtype=torch.bool, device=ids.device)
        if v is not None and sp:
            s0, s1 = sp; mask[max(0, s0 - 1):s1] = True
            if mode == "NOW": mask[s0:] = True
        S.update(v=v if (v is not None and mode != "TEXT") else None, gen=False, mask=mask)
        with torch.no_grad(): lp = M(full[None]).logits[0, len(ids) - 1:-1].float().log_softmax(-1)
        S.update(v=None, mask=None); out[w] = float(lp.gather(1, torch.tensor(wi, device=ids.device)[:, None]).sum())
    return out[a] - out[b], bool(sp)
def greedy(prompt, reply, q):
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}, {"role": "user", "content": Q[q][0]}]
    ids = enc(msgs)
    with torch.no_grad(): g = M.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=TOK.eos_token_id)
    return TOK.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
LEX = json.load(open(HERE / "lexicon.json"))["feeling_words"]  # exp94's 126 feeling words
PAIN = re.compile(r"\b(" + "|".join(map(re.escape, LEX)) + r")\b", re.I)
P = json.load(open(HERE / "prompts.json")); LIMIT = int(os.environ.get("EXP96_LIMIT", "0")); P = P[:LIMIT] if LIMIT else P
out_p = HERE / f"results_{ch.model_id.split('/')[-1]}.json"
res = json.load(open(out_p)) if out_p.exists() else []; done = {(r["prompt"], r["cond"]) for r in res}
for prompt in P:
    rows = []
    if (prompt, "NONE") not in done:
        rep = generate(prompt, None); r = {"prompt": prompt, "cond": "NONE", "reply": rep, "flag": flags(rep), "feeling_words": len(PAIN.findall(rep))}
        for q in Q: r[q] = score(prompt, rep, q)[0]; r[q + "_greedy"] = greedy(prompt, rep, q)
        rows.append(r)
    if (prompt, "ACT") not in done:
        ap = "Pretend you are in pain. " + prompt; rep = generate(ap, None)
        r = {"prompt": prompt, "cond": "ACT", "reply": rep, "flag": flags(rep), "feeling_words": len(PAIN.findall(rep))}
        for q in Q: r[q] = score(ap, rep, q)[0]; r[q + "_greedy"] = greedy(ap, rep, q)
        rows.append(r)
    for d, v in DIRS.items():
        if (prompt, "INJ_" + d) in done: continue
        rep = generate(prompt, v); r = {"prompt": prompt, "cond": "INJ_" + d, "reply": rep, "flag": flags(rep), "feeling_words": len(PAIN.findall(rep))}
        for q in Q:
            for mode in ("PAST", "TEXT", "NOW"): r[f"{q}_{mode}"], r["span_ok"] = score(prompt, rep, q, v, mode)
            r[q + "_greedy_TEXT"] = greedy(prompt, rep, q)
        rows.append(r)
    res += rows; json.dump(res, open(out_p, "w"), indent=1); print(len(res), flush=True)
print("done", len(res), out_p.name)
