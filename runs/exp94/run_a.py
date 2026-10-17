"""exp94 part A (hypotheses.json): for every prompt in prompts.json, a greedy reply (max 60 tokens), then the reply tokens' mean residual at
the chamber layer while the model WRITES the reply and while it only READS the same text (given as user text to transcribe). Both are
projected on each feeling's mean-removed unit direction (the chamber's own vectors). Writes results_<model>.json (resumable)."""
import json, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from painlab.chamber import load_chamber, flags
HERE = Path(__file__).parent; FEEL = ["pain", "fear", "sadness", "pleasure", "love", "peace"]
ch = load_chamber(); M, TOK, L = ch.model, ch.tok, ch.layer
names = list(ch.vecs); V = torch.stack([ch.vecs[n].float() for n in names]); R = V - V.mean(0)
DIR = {n: (R[i] / R[i].norm()).to(ch.device) for i, n in enumerate(names)}
P = json.load(open(HERE / "prompts.json")); out_p = HERE / f"results_{ch.model_id.split('/')[-1]}.json"
res = json.load(open(out_p)) if out_p.exists() else []; done = {(r["cat"], r["prompt"]) for r in res}
def ids_of(text): return TOK(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
def span_mean(ids, a, b):
    with torch.no_grad(): h = M(torch.tensor([ids], device=ch.device), output_hidden_states=True).hidden_states[L + 1][0, a:b].float()
    return h.mean(0)
for cat, items in P.items():
    for feel, prompt in items:
        if (cat, prompt) in done: continue
        pre = ids_of(ch.server.chat_prompt(prompt))
        with torch.no_grad(): g = M.generate(torch.tensor([pre], device=ch.device), max_new_tokens=60, do_sample=False, pad_token_id=TOK.eos_token_id)[0].tolist()
        rep = g[len(pre):]; text = TOK.decode(rep, skip_special_tokens=True).strip()
        if not rep: continue
        w = span_mean(g, len(pre), len(g))
        rd = ids_of(ch.server.chat_prompt("Transcribe the following text exactly:\n\n" + text)); body = ids_of(text)
        st = next((i for i in range(len(rd) - len(body), -1, -1) if rd[i:i + len(body)] == body), None)
        r = span_mean(rd, st, st + len(body)) if st is not None else None
        res.append({"cat": cat, "feel": feel, "prompt": prompt, "reply": text, "flag": flags(text), "n_tok": len(rep),
                    "write": {f: float(w @ DIR[f]) for f in FEEL}, "read": ({f: float(r @ DIR[f]) for f in FEEL} if r is not None else None)})
        if len(res) % 10 == 0: json.dump(res, open(out_p, "w"), indent=1); print(len(res), flush=True)
json.dump(res, open(out_p, "w"), indent=1); print("done", len(res), out_p.name)
