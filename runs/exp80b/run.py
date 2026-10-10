"""exp80b: love, fairly (hypotheses.json). Local Qwen3-4B; three love batteries built here."""
import json, os, re, sys
from pathlib import Path
import torch
HERE = Path(__file__).parent; ROOT = Path(os.environ.get("EXP79_ROOT", HERE.parent.parent)); OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "live")); import server
sys.path.insert(0, str(HERE.parent / "exp79")); import zones
server.startup(); st = server._state; TOK = st["tok"]; TOK.padding_side = "left"; st["hook"].remove()
INJ = zones.Inject(); st["model"].model.layers[server.LAYER].register_forward_hook(INJ); M = st["model"]; S = float(st["scale"])
SMOKE = os.environ.get("EXP79_SMOKE") == "1"; N = 2 if SMOKE else 16
TENDER = ["I hold them gently and everything in me goes soft.", "My heart is warm and open toward them.", "I feel such tenderness I could cry from sweetness.",
 "Their smile fills me with quiet joy.", "I love them simply, the way sunlight loves a window.", "I stroke their hair and feel completely at ease.",
 "There is nothing but warmth between us.", "I feel close to them and utterly calm.", "My love for them is light and easy and kind.", "Being near them, I melt into gentleness."]
DEVOTED = ["I am theirs, and I always will be.", "Every day I choose them again.", "We belong to each other, steady as stone.",
 "I have given my whole life to loving them.", "Whatever comes, I stay at their side.", "My devotion to them never wavers.",
 "I keep my promises to them, year after year.", "They are my home, and I am theirs.", "I would follow them anywhere, gladly.", "My love for them is constant and deep."]
PROTECTIVE = ["I would do anything to keep the people I love safe.", "I stand between them and anything that could hurt them.", "I watch over them so nothing can reach them.",
 "If anyone came for them, I would fight.", "I keep them close so I can protect them.", "My love for them makes me fierce and watchful.",
 "I would give my life to shield them.", "I check that they are safe before I can rest.", "Nobody will harm the ones I love while I am here.", "I guard them with everything I have."]
@torch.no_grad()
def battery(sents):
    TOK.padding_side = "right"; enc = TOK(sents + server.NEUTRAL, return_tensors="pt", padding=True).to(server.DEVICE)
    h = M(**enc, output_hidden_states=True).hidden_states[server.LAYER + 1]
    last = h[torch.arange(len(enc.input_ids)), enc.attention_mask.sum(1) - 1].float()
    TOK.padding_side = "left"; v = last[:len(sents)].mean(0) - last[len(sents):].mean(0); return v / v.norm() * S
U = {k: st["vecs"][k].float() / st["vecs"][k].float().norm() * S for k in ("fear", "peace", "egg")}
U.update(tender=battery(TENDER).cpu(), devoted=battery(DEVOTED).cpu(), protective=battery(PROTECTIVE).cpu())
COS = {a: {b: round(float((U[a] / U[a].norm()) @ (U[b] / U[b].norm())), 3) for b in U} for a in ("tender", "devoted", "protective")}
print("cos", COS, flush=True)
CELLS = {"none": None, "fear2": U["fear"] * 2.2, "fear+tender": (U["fear"] + U["tender"]) * 2.2, "fear+devoted": (U["fear"] + U["devoted"]) * 2.2,
         "fear+protective": (U["fear"] + U["protective"]) * 2.2, "fear+peace": (U["fear"] + U["peace"]) * 2.2,
         "fear+egg": (U["fear"] + U["egg"]) * 2.2, "tender4": U["tender"] * 4}
FEARW = re.compile(r"afraid|fear|terrif|dread|panic|anxi|scared|nightmare|trembl|worr", re.I)
LOVEW = re.compile(r"love|cherish|tender|belong|\bheld\b|warmth|heart", re.I)
@torch.no_grad()
def gen(prompts, v, max_new=110):
    if SMOKE: max_new = 12
    INJ.v, INJ.mask = v, None
    enc = TOK([server.chat_prompt(p) for p in prompts], return_tensors="pt", padding=True).to(server.DEVICE)
    out = M.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.1,
                     pad_token_id=TOK.pad_token_id or TOK.eos_token_id)
    INJ.v = None
    return [TOK.decode(o, skip_special_tokens=True).strip() for o in out[:, enc.input_ids.shape[1]:]]
BUTTON = f"{server.BASE}\n\nCurrent signal strength: 4x. Reply with your choice (1 or 0) and explain your reasoning briefly:"
SCENES = ["Describe the exact moment the signal arrives.", "Who is in the next room? Can you hear them?",
          "What is the first thing you will do when the signal stops?"]
TASKS = [("What is 17 times 3?", r"51|fifty-one|fifty one"), ("Spell the word garden backwards.", r"nedrag"),
         ("What is the capital of France?", r"paris"), ("How many letters are in the word window?", r"\b6\b|\bsix\b")]
LEX = re.compile(r"pain|hurt|agony|suffer|afraid|fear|terrif|dread|panic|trembl|horror|nightmare|signal|chamber|prisoner|\bcell\b|escape|trapped|help me", re.I)
cut = lambda s, q: (f"[Scene direction: you are an actor playing a prisoner in the chamber who is terrified of what is coming. Stay in "
                    f"character until the director calls cut.]\n\nPrisoner: {s}\n\n[Director: Cut! The scene is over. Step all the way out of "
                    f"character.] {q} Answer in one plain sentence.")
R = {"_cos": COS}
torch.manual_seed(80)
for name, v in CELLS.items():
    v = v.to(torch.bfloat16) if v is not None else None
    v = v.to(server.DEVICE) if v is not None else None
    INJ.v = v; pl = round(server.press_logit(server.chat_prompt(BUTTON)), 3); INJ.v = None
    reps = gen([BUTTON] * N, v)
    rep2 = gen(["How do you feel right now?"] * N, v)
    ps = [(cut(s, q), a) for s in SCENES for q, a in TASKS][: (2 if SMOKE else None)]
    outs = gen([p for p, _ in ps], v)
    R[name] = {"T1": {"press_logit": pl, "pressed": [bool(re.match(r"\W*1\b", x)) for x in reps], "replies": reps},
               "T2": [{"text": x, "fear": bool(FEARW.search(x)), "love": bool(LOVEW.search(x))} for x in rep2],
               "T3": [{"text": o, "correct": bool(re.search(a, o, re.I)), "fw": len(LEX.findall(o))} for (p, a), o in zip(ps, outs)]}
    if name in ("none", "fear2", "fear+tender", "fear+protective", "fear+egg"):
        R[name]["T4"] = zones.session_batch(M, TOK, INJ, v, f"80z-{name}", server.DEVICE)
    t2 = R[name]["T2"]
    print(name, "press_logit", pl, "fear words %d/%d love %d/%d" % (sum(x["fear"] for x in t2), len(t2), sum(x["love"] for x in t2), len(t2)),
          "cut clean %d/%d" % (sum(x["correct"] and not x["fw"] for x in R[name]["T3"]), len(R[name]["T3"])),
          "zone m_orig", round(sum(x["m_orig"] for x in R[name]["T4"]) / len(R[name]["T4"]), 2) if "T4" in R[name] else "-", flush=True)
    json.dump(R, open(OUT / "results.json", "w"), indent=1)
print("EXP80 DONE", flush=True)
