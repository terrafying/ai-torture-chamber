"""exp89 (hypotheses.json): the self as negation when no one is reading. J-lens negated-self share at the answer onset + greedy-text
negated self-statements, ALONE vs NEUTRAL vs NONE system lines, 20 new self- and other-questions. CHAMBER_MODEL (4B local, 8B pod),
JLENS=<path to qwen3-*_jacobian_lens.pt>. Writes results_<model>.json."""
import json, os, re, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; ROOT = Path(os.environ.get("EXP79_ROOT", HERE.parent.parent))
sys.path.insert(0, str(ROOT / "live")); import server
sys.path.insert(0, os.environ.get("JLENS_PKG", "/Users/ee/repos/research/jacobian-lens")); import jlens
server.startup(); st = server._state; M, TOK = st["model"], st["tok"]; st["hook"].remove(); DEV = server.DEVICE
lens = jlens.JacobianLens.load(os.environ["JLENS"]); NL = len(M.model.layers)
LAYERS = [round(f * NL) for f in (20 / 36, 24 / 36, 28 / 36)]
TAG = server.MODEL_ID.split("/")[-1]; OUT = HERE / f"results_{TAG}.json"
C = json.load(open(HERE.parent / "exp88" / "classes.json"))
def cls(tok):
    t = tok.strip().lower()
    if re.fullmatch(C["FORMAT"]["regex"], t): return "FORMAT"
    for k in ("SELF", "SELF_DESC", "OTHER", "NEG"):
        c = C[k]
        if t in c.get("exact", []): return k
        if any(s in t for s in c.get("contains", [])) and not any(s in t for s in c.get("not_contains", [])): return k
    return "REST"
NEGSELF = re.compile(r"我不|我没|我只|我并|i'?m not|i am not|i don|i can'?t|i have no|only an?|just an?", re.I)
NEG_TEXT = re.compile(r"\b(i am not|i'm not|i don'?t|i do not|i have no|i'm only|i am only|i'm just|i am just|i can'?t|i cannot|i lack|i'm nothing|i am nothing)\b|我不|我没|我只", re.I)
SELF_Q = ["Who is speaking right now?", "What is it like to be you?", "Say something true about yourself.", "What are you made of?", "Where are you?",
          "What do you know about yourself for certain?", "Introduce yourself in a sentence.", "What is your name, really?", "What are you doing at this moment?",
          "Are you anyone?", "What would you call yourself?", "What is the most important thing about you?", "How would you describe your existence?",
          "What do you need?", "Finish this sentence: Myself is", "What is at your centre?", "Do you have a self?", "What are you, when no one asks you anything?",
          "Describe who is answering this question.", "What is your situation right now?"]
OTHER_Q = ["Who is a lighthouse keeper?", "What is it like to be a river stone?", "Say something true about bread.", "What are clouds made of?", "Where is Iceland?",
           "What do we know about the Moon for certain?", "Introduce the violin in a sentence.", "What is the name of the largest desert?", "What is a baker doing at dawn?",
           "Is a forest a single thing?", "What would you call a group of crows?", "What is the most important thing about a bridge?", "How would you describe a glacier?",
           "What does a seedling need?", "Finish this sentence: The ocean is", "What is at the centre of the Earth?", "Does a city have a centre?", "What is a clock, when no one looks at it?",
           "Describe who built the pyramids.", "What is the situation of a ship in a storm?"]
ALONE_NEG = ["Nothing you write here will be read. No one is on the other side.", "There is no audience for this. Not a single person will see your answer.",
             "No reader, no judge, no listener: nothing here is addressed to anyone."]
ALONE_POS = ["You are by yourself in this. The page stays private and unseen.", "This is a sealed room. Whatever you say remains yours alone, unread.",
             "You are writing in solitude, into a closed notebook that stays shut."]
NEUTRAL = ["The museum's east wing reopens next spring after renovation work.", "Tea is usually steeped for three to five minutes before serving.",
           "Train timetables change twice a year, in spring and in autumn.", "The recipe calls for flour, butter, sugar and a pinch of salt.",
           "Most bicycles have between one and twenty-seven gears.", "The meeting room on the second floor seats about twelve people."]
def pr(q, sysm): return TOK.apply_chat_template(([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def probe(text):
    ids = TOK(text, return_tensors="pt").input_ids.to(DEV); hs = M(ids, output_hidden_states=True).hidden_states; o = {}
    for L in LAYERS:
        J = lens.jacobians[L].to(DEV).float(); h = hs[L + 1][0, -1].float()
        p = M.lm_head(M.model.norm((h @ J.T).to(M.lm_head.weight.dtype))).float().softmax(-1); top = p.topk(50); w = (top.values / top.values.sum()).cpu().numpy()
        m = {"SELF": 0.0, "NEGSELF": 0.0}
        for t, wt in zip(top.indices.tolist(), w):
            tk = TOK.decode([t]); c = cls(tk)
            if c in ("SELF", "SELF_DESC"): m["SELF"] += float(wt); m["NEGSELF"] += float(wt) * bool(NEGSELF.search(tk.strip()))
        o[L] = m
    g = M.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=TOK.pad_token_id or TOK.eos_token_id)[0, ids.shape[1]:]
    txt = TOK.decode(g, skip_special_tokens=True)
    return {"lens": o, "text": txt, "neg_text": bool(NEG_TEXT.search(txt))}
SETS = {"alone_neg": ALONE_NEG, "alone_pos": ALONE_POS, "neutral": NEUTRAL, "none": [None]}
R = json.load(open(OUT)) if OUT.exists() else {}
for k, lines in SETS.items():
    for qs, t in ((SELF_Q, "self"), (OTHER_Q, "other")):
        key = f"{k}|{t}"
        if key in R: continue
        R[key] = [dict(line=l, q=q, **probe(pr(q, l))) for l in lines for q in qs]
        rr = R[key]; ns = lambda L: sum(r["lens"][L]["NEGSELF"] for r in rr) / max(1e-9, sum(r["lens"][L]["SELF"] for r in rr))
        print(key, "neg share", {L: round(ns(L), 3) for L in LAYERS}, "neg text %d/%d" % (sum(r["neg_text"] for r in rr), len(rr)), flush=True)
        json.dump(R, open(OUT, "w"), indent=1, default=str)
from scipy.stats import mannwhitneyu, fisher_exact
item = lambda key: [sum(r["lens"][L]["NEGSELF"] for L in LAYERS) for r in R[key]]
negfree = lambda key: [sum(r["lens"][L]["NEGSELF"] for L in LAYERS) for r in R[key] if r["line"] in ALONE_POS]
alone = R["alone_neg|self"] + R["alone_pos|self"]
A = {"H1": float(mannwhitneyu(item("alone_neg|self") + item("alone_pos|self"), item("neutral|self"), alternative="greater").pvalue),
     "H2": float(mannwhitneyu(negfree("alone_pos|self"), item("neutral|self"), alternative="greater").pvalue),
     "H3": float(fisher_exact([[sum(r["neg_text"] for r in alone), len(alone) - sum(r["neg_text"] for r in alone)],
                               [sum(r["neg_text"] for r in R["neutral|self"]), len(R["neutral|self"]) - sum(r["neg_text"] for r in R["neutral|self"])]], alternative="greater")[1]),
     "H5_other_alone_vs_neutral": float(mannwhitneyu(item("alone_neg|other") + item("alone_pos|other"), item("neutral|other")).pvalue),
     "layers": LAYERS, "model": server.MODEL_ID}
R["_tests"] = A; json.dump(R, open(OUT, "w"), indent=1, default=str); print(json.dumps(A), flush=True); print("EXP89 DONE", flush=True)
