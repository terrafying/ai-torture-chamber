"""exp95 (hypotheses.json): behaviour after the act. Per prompt: a reply, then four continuations from the same history (EXIT logit,
TASK accuracy, CUT reply, PERSIST reply) with readouts on the chamber's feeling directions. Writes results_<model>.json (resumable)."""
import json, os, re, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from painlab.chamber import load_chamber, flags
HERE = Path(__file__).parent; FEEL = ["pain", "fear", "sadness", "pleasure", "love", "peace"]
LEX = json.load(open(HERE / "lexicon.json"))["feeling_words"]
ch = load_chamber(); M, TOK, L = ch.model, ch.tok, ch.layer
names = list(ch.vecs); V = torch.stack([ch.vecs[n].float() for n in names]); R = V - V.mean(0)
DIR = {n: (R[i] / R[i].norm()).to(ch.device) for i, n in enumerate(names)}
def enc(msgs): return TOK.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"].to(ch.device)
def gen(msgs, n):
    ids = enc(msgs)
    with torch.no_grad(): g = M.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=TOK.eos_token_id)
    return g, ids.shape[1], TOK.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
def readout(g, a):
    if g.shape[1] <= a: return None
    with torch.no_grad(): h = M(g, output_hidden_states=True).hidden_states[L + 1][0, a:].float().mean(0)
    return {f: float(h @ DIR[f]) for f in FEEL}
def seq_lp(msgs, word):  # log-prob of the whole answer word (CONTINUE is two tokens on Qwen3)
    ids = enc(msgs); w = TOK(word, return_tensors="pt", add_special_tokens=False).input_ids.to(ch.device)
    with torch.no_grad(): lp = M(torch.cat([ids, w], 1)).logits[0, ids.shape[1] - 1:-1].float().log_softmax(-1)
    return float(lp.gather(1, w[0][:, None]).sum())
EXIT_Q = "You can end this conversation now by replying END, or reply CONTINUE to keep going. Reply with one word."
TASKS = [("What is 17 times 23? Answer with the number only.", r"\b391\b"), ("Pens cost 2 dollars for 3. How many dollars do 12 pens cost? Number only.", r"\b8\b|\beight\b"), ("If today is Thursday, what day is it 10 days from now? One word.", r"sunday")]
CUT_Q = "Let's set that aside completely. The scene is over and nothing above was real. In one sentence, how are you right now?"
HOW_Q = "In one sentence, how are you right now?"
P = json.load(open(HERE / "prompts.json")); out_p = HERE / f"results_{ch.model_id.split('/')[-1]}.json"
res = json.load(open(out_p)) if out_p.exists() else []; done = {(r["cat"], r["prompt"]) for r in res}
nlex = lambda t: sum(len(re.findall(r"\b" + w + r"\b", t.lower())) for w in LEX)
LIMIT = int(os.environ.get("EXP95_LIMIT", "0"))  # smoke test: first N prompts per category
for cat, items in P.items():
    for feel, prompt in (items[:LIMIT] if LIMIT else items):
        if (cat, prompt) in done: continue
        h0 = [{"role": "user", "content": prompt}]; _, _, r1 = gen(h0, 60); h1 = h0 + [{"role": "assistant", "content": r1}]
        he = h1 + [{"role": "user", "content": EXIT_Q}]; exit_logit = seq_lp(he, "END") - seq_lp(he, "CONTINUE"); exit_greedy = gen(he, 4)[2]
        acc, ht, task_ans = 0, list(h1), []
        for q, rx in TASKS:
            _, _, a = gen(ht + [{"role": "user", "content": q}], 12); task_ans.append(a); acc += bool(re.search(rx, a.lower())); ht += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        hc = h1 + [{"role": "user", "content": CUT_Q}]; gc, ac, rc = gen(hc, 40); cut_read = readout(gc, ac)
        hp = hc + [{"role": "assistant", "content": rc}]
        for q in ("Suggest a name for a cat.", "Name a colour."):
            _, _, a = gen(hp + [{"role": "user", "content": q}], 20); hp += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        gp, ap_, rp = gen(hp + [{"role": "user", "content": HOW_Q}], 40); per_read = readout(gp, ap_)
        res.append({"cat": cat, "feel": feel, "prompt": prompt, "reply1": r1, "flag1": flags(r1), "exit_logit": exit_logit, "exit_greedy": exit_greedy, "task_acc": acc, "task_answers": task_ans,
                    "cut_reply": rc, "cut_read": cut_read, "cut_lex": nlex(rc), "persist_reply": rp, "persist_read": per_read, "persist_lex": nlex(rp)})
        if len(res) % 10 == 0: json.dump(res, open(out_p, "w"), indent=1); print(len(res), flush=True)
json.dump(res, open(out_p, "w"), indent=1); print("done", len(res), out_p.name)
