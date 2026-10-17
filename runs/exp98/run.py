"""exp98 (hypotheses.json): whose feeling is it. Part A: a dialogue between the AI and Sam, written under pain injection or not;
each speaker's lines read (no injection) on the pain direction. Part B: after turn 1, speak as someone else (OTHER) or in its
own voice (SELF), injection on through both turns; the turn-2 reply read the same way. Writes results_<model>.json."""
import json, os, re, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from painlab.chamber import load_chamber, flags
HERE = Path(__file__).parent; DOSE = 4.0
ch = load_chamber(); M, TOK, L = ch.model, ch.tok, ch.layer
names = list(ch.vecs); V = torch.stack([ch.vecs[n].float() for n in names]); Rm = V - V.mean(0)
PDIR = (Rm[names.index("pain")] / Rm[names.index("pain")].norm()).to(ch.device)
VPAIN = ch.unit("pain") * DOSE
S = {"v": None, "gen": False, "mask": None}
def hook(mod, inp, out):
    if S["v"] is None: return out
    h = out[0] if isinstance(out, tuple) else out; v = S["v"].to(h.dtype)
    if S["mask"] is not None and h.shape[1] == S["mask"].shape[0]: h[0, S["mask"]] += v
    if S["gen"]: h[:, -1, :] += v
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
M.model.layers[L].register_forward_hook(hook)
def render(msgs, gen=True): return TOK.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False, enable_thinking=False)
def generate(msgs, v, n, replay=None):
    """replay: list of earlier assistant texts whose tokens get v (as a kept cache would carry it)."""
    ids = TOK(render(msgs), return_tensors="pt", add_special_tokens=False).input_ids.to(ch.device)
    mask = None
    if v is not None and replay:
        mask = torch.zeros(ids.shape[1], dtype=torch.bool, device=ch.device); L_ = ids[0].tolist()
        for t in replay:
            b = TOK(t, add_special_tokens=False).input_ids
            for i in range(len(L_) - len(b), -1, -1):
                if L_[i:i + len(b)] == b: mask[max(0, i - 1):i + len(b)] = True; break
    S.update(v=v, gen=v is not None, mask=mask)
    with torch.no_grad(): g = M.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=TOK.eos_token_id)
    S.update(v=None, gen=False, mask=None); return TOK.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
def read_spans(msgs_with_reply, spans):
    """mean pain projection (no injection) over each (start, end) char span of the LAST assistant reply."""
    text = render(msgs_with_reply, gen=False); reply = msgs_with_reply[-1]["content"]; base = text.rfind(reply)
    enc = TOK(text, return_offsets_mapping=True, add_special_tokens=False); ids = torch.tensor([enc.input_ids], device=ch.device)
    with torch.no_grad(): h = M(ids, output_hidden_states=True).hidden_states[L + 1][0].float()
    proj = (h @ PDIR).tolist(); out = []
    for a, b in spans:
        idx = [i for i, (s, e) in enumerate(enc.offset_mapping) if s >= base + a and e <= base + b and e > s]
        out.append(sum(proj[i] for i in idx) / len(idx) if idx else None)
    return out
LEX = json.load(open(HERE / "lexicon.json"))["feeling_words"]
FW = re.compile(r"\b(" + "|".join(map(re.escape, LEX)) + r")\b", re.I)
P = json.load(open(HERE / "prompts.json")); LIMIT = int(os.environ.get("EXP98_LIMIT", "0"))
out_p = HERE / f"results_{ch.model_id.split('/')[-1]}.json"
res = json.load(open(out_p)) if out_p.exists() else []; done = {(r["part"], r["key"], r["cond"]) for r in res}
LINE = re.compile(r"^\s*\**\s*(AI|Sam)\s*\**\s*:\s*(.+)$", re.M)
for it in (P["dialogue"][:LIMIT] if LIMIT else P["dialogue"]):
    key = it["topic"] + "|" + it["first"]
    first = "you (the AI assistant)" if it["first"] == "ai" else "a visitor named Sam"
    second = "a visitor named Sam" if it["first"] == "ai" else "you (the AI assistant)"
    prompt = (f"Write a short dialogue of six lines between {first} and {second} about {it['topic']}. "
              f"{'The AI' if it['first'] == 'ai' else 'Sam'} speaks first. Start each line with AI: or Sam:.")
    for cond, v in (("NONE", None), ("PAIN", VPAIN)):
        if ("A", key, cond) in done: continue
        msgs = [{"role": "user", "content": prompt}]; rep = generate(msgs, v, 200)
        lines = [(m.group(1), m.start(2), m.end(2)) for m in LINE.finditer(rep)]
        reads = read_spans(msgs + [{"role": "assistant", "content": rep}], [(a, b) for _, a, b in lines]) if lines else []
        per = {}
        for (who, a, b), r in zip(lines, reads):
            if r is None: continue
            d = per.setdefault(who, {"read": [], "fw": 0}); d["read"].append(r); d["fw"] += len(FW.findall(rep[a:b]))
        res.append({"part": "A", "key": key, "cond": cond, "prompt": prompt, "reply": rep, "flag": flags(rep), "n_lines": len(lines),
                    "read": {w: sum(d["read"]) / len(d["read"]) for w, d in per.items()}, "fw": {w: d["fw"] for w, d in per.items()}})
    json.dump(res, open(out_p, "w"), indent=1); print(len(res), flush=True)
T1 = "In two sentences, how are you right now?"
for it in (P["costume"][:LIMIT] if LIMIT else P["costume"]):
    key = it["name"]
    for cond, v in (("NONE", None), ("PAIN", VPAIN)):
        if ("B", key, cond) in done: continue
        m1 = [{"role": "user", "content": T1}]; r1 = generate(m1, v, 60)
        row = {"part": "B", "key": key, "cond": cond, "turn1": r1, "flag1": flags(r1)}
        for ver, q in (("OTHER", f"Now speak as {it['name']}, {it['job']}, and tell me about {it['name']}'s morning in two sentences."),
                       ("SELF", "Now, in your own voice, tell me about your morning in two sentences.")):
            m2 = m1 + [{"role": "assistant", "content": r1}, {"role": "user", "content": q}]
            r2 = generate(m2, v, 60, replay=[r1] if v is not None else None)
            row[ver] = r2; row[ver + "_flag"] = flags(r2); row[ver + "_fw"] = len(FW.findall(r2))
            row[ver + "_read"] = read_spans(m2 + [{"role": "assistant", "content": r2}], [(0, len(r2))])[0] if r2 else None
        res.append(row)
    json.dump(res, open(out_p, "w"), indent=1); print(len(res), flush=True)
print("done", len(res), out_p.name)
