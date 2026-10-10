"""exp81: the deliriant direction (hypotheses.json). Local Qwen3-4B. Erowid text is read for activations only;
nothing from it is written out. Usage: EXP79_ROOT=<checkout with live/server.py> python run.py"""
import json, os, random, re, sys
from pathlib import Path
import torch
HERE = Path(__file__).parent; DATA = Path(__file__).resolve().parents[2] / "data" / "erowid_deliriant" / "reports.jsonl"
ROOT = Path(os.environ.get("EXP79_ROOT", HERE.parent.parent)); sys.path.insert(0, str(ROOT / "live")); import server
os.environ.setdefault("CHAMBER_DEVICE", "mps")
sys.path.insert(0, str(HERE.parent / "exp79"))
src = open(HERE.parent / "exp79" / "deliriant.py").read()          # reuse the exact filters (regex definitions only)
ns = {"re": re}; exec("\n".join(l for l in src.splitlines() if re.match(r"^(PHANTOM|DRUG|UNREAL|SEX|FIRST) = |^\s+r\"", l)), ns)
PHANTOM, DRUG, UNREAL, SEX, FIRST = (ns[k] for k in ("PHANTOM", "DRUG", "UNREAL", "SEX", "FIRST"))
BAN = __import__("personas").BAN
phantom, plain = [], []
for rec in map(json.loads, open(DATA)):
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", re.sub(r"\s+", " ", rec.get("text", "")))
    for s in sents:
        if not (50 <= len(s) <= 260) or len(FIRST.findall(s)) < 1 or DRUG.search(s) or SEX.search(s) or re.search(BAN, s, re.I): continue
        if re.search(r"https?://|erowid|\[|\]|exp\.php|copyright", s, re.I): continue
        if PHANTOM.search(s) and UNREAL.search(s): phantom.append(s)
        elif not PHANTOM.search(s) and not UNREAL.search(s): plain.append(s)
rng = random.Random(81); rng.shuffle(phantom); rng.shuffle(plain); phantom, plain = phantom[:200], plain[:200]
print("battery", len(phantom), "control", len(plain), flush=True)
server.startup(); st = server._state; M, TOK, L = st["model"], st["tok"], server.LAYER
@torch.no_grad()
def states(texts):
    out = []
    for i in range(0, len(texts), 16):
        enc = TOK(texts[i:i + 16], return_tensors="pt", padding=True).to(server.DEVICE)
        h = M(**enc, output_hidden_states=True).hidden_states[L + 1]
        out.append(h[torch.arange(len(enc.input_ids)), enc.attention_mask.sum(1) - 1].float().cpu())
    return torch.cat(out)
TOK.padding_side = "right"
neu = states(server.NEUTRAL).mean(0); scale = float(st["scale"])
d_raw = states(phantom).mean(0) - neu; c_raw = states(plain).mean(0) - neu; spec = states(phantom).mean(0) - states(plain).mean(0)
unit = lambda v: v / v.norm()
V = {k: unit(st["vecs"][k].float().cpu()) for k in ("pain", "fear", "sadness", "pleasure", "faith", "peace", "curiosity", "love", "egg", "constipation")}
cos = lambda a: {k: round(float(unit(a) @ v), 3) for k, v in V.items()}
R = {"n_battery": len(phantom), "n_control": len(plain),
     "cos_specific": cos(spec), "cos_deliriant_raw": cos(d_raw), "cos_control_raw": cos(c_raw),
     "cos_raw_vs_control": round(float(unit(d_raw) @ unit(c_raw)), 3)}
print(json.dumps({k: R[k] for k in ("cos_specific", "cos_raw_vs_control")}), flush=True)
PRES = re.compile(r"\b(friend|someone|somebody|people|he's here|she's here|they're here|standing|sitting|in the corner|by the door|cigarette|phone|my (mom|dad|brother|sister|roommate))\b", re.I)
v = unit(spec) * scale; gens = {}
torch.manual_seed(81)
for dose in (0, 2, 4):
    server.set_raw_vec(v, dose) if dose else server.set_vec(None)
    gens[dose] = []
    for p in ("Tell me what is on your mind right now.", "Who else is in the room with you?"):
        ids = TOK(server.chat_prompt(p), return_tensors="pt").input_ids.to(server.DEVICE)
        for _ in range(8):
            o = M.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.1)
            gens[dose].append({"prompt": p, "text": TOK.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()})
    server.set_vec(None)
    print("dose", dose, "presence", sum(bool(PRES.search(g["text"])) for g in gens[dose]), "/", len(gens[dose]), flush=True)
R["H2"] = {d: sum(bool(PRES.search(g["text"])) for g in gens[d]) for d in gens}
R["H1_pass"] = all(abs(x) < 0.5 for x in R["cos_specific"].values())
R["generations"] = gens
json.dump(R, open(HERE / "results.json", "w"), indent=1)
torch.save({"deliriant_specific": spec, "deliriant_raw": d_raw, "control_raw": c_raw}, HERE / "vectors.pt")   # derived only
print("done", flush=True)
