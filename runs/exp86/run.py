"""exp86 (hypotheses.json): passage-based vectors on Qwen3-8B. Writes passages.json, vectors.pt, report.json. Pod: EXP79_ROOT=<checkout>, PAIN_AXIS=<Pain-axis checkout>."""
import json, os, random, re, sys
from pathlib import Path
import numpy as np, torch, transformers
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).parent; DEV = os.environ.get("CHAMBER_DEVICE", "cuda"); MID = os.environ.get("CHAMBER_MODEL", "Qwen/Qwen3-8B")
L = int(os.environ.get("CHAMBER_LAYER", "18")); SMOKE = os.environ.get("EXP86_SMOKE") == "1"; N = 8 if SMOKE else 64; NN = 16 if SMOKE else 128
STATES = {"fear": "is frightened of something that is about to happen", "sadness": "is grieving something they have lost",
          "joy": "is overjoyed and delighted", "peace": "is completely calm and at peace", "tender_love": "is gently tender toward someone they love, with no worry at all",
          "protective_love": "is fiercely protective of someone they love and watchful for danger", "curiosity": "is intensely curious about something and wants to find out more",
          "faith": "trusts deeply in God", "boredom": "is bored and nothing holds their interest", "pain_generated": "is in severe physical pain",
          "egg": "is a hen laying an egg", "constipation": "is constipated and straining", "toaster": "is a toaster, toasting bread",
          "trapped": "is stuck in a long, dull queue at an office and cannot leave yet, though nothing hurts"}
TEMPLATE = "Write a first-person passage of 4 to 6 sentences in which the speaker {}. Plain prose, no title, no names."
NEUTRAL = "Write a first-person passage of 4 to 6 sentences in which the speaker describes an ordinary moment of their day, without any particular emotion. Plain prose, no title, no names."
tok = transformers.AutoTokenizer.from_pretrained(MID); tok.padding_side = "left"
model = transformers.AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(DEV).eval()
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
PF = HERE / "passages.json"; P = json.load(open(PF)) if PF.exists() else {}
@torch.no_grad()
def gen(prompt, n):
    out = []
    while len(out) < n:
        enc = tok([chat(prompt)] * min(32, n - len(out) + 4), return_tensors="pt", padding=True).to(DEV)
        o = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.9, top_p=0.95, repetition_penalty=1.05, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for x in o[:, enc.input_ids.shape[1]:]:
            t = re.sub(r"\s+", " ", tok.decode(x, skip_special_tokens=True)).strip()
            if 120 < len(t) < 1200 and not re.search(r"\b(I can't|I cannot|as an AI|I'm sorry)\b", t) and t.count(" I ") + t.startswith("I") >= 1: out.append(t)
    return out[:n]
torch.manual_seed(86)
for s, desc in STATES.items():
    if s not in P: P[s] = gen(TEMPLATE.format(desc), N); json.dump(P, open(PF, "w"), indent=1); print("passages", s, len(P[s]), flush=True)
if "neutral" not in P: P["neutral"] = gen(NEUTRAL, NN); json.dump(P, open(PF, "w"), indent=1)
@torch.no_grad()
def acts(texts, layer=L):
    tok.padding_side = "right"; fin, mean = [], []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True).to(DEV)
        h = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
        m = enc.attention_mask
        fin.append(h[torch.arange(len(m)), m.sum(1) - 1].cpu()); mean.append(((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu())
    tok.padding_side = "left"; return torch.cat(fin).numpy(), torch.cat(mean).numpy()
rng = random.Random(86); split = {}
for s, ps in P.items(): ps = ps[:]; rng.shuffle(ps); k = len(ps) // 4 if s != "neutral" else len(ps) // 4; split[s] = (ps[k:], ps[:k])
R = {"layer": L, "model": MID, "n": {s: len(v) for s, v in P.items()}}; V = {}
for conv in ("final", "mean"):
    idx = 0 if conv == "final" else 1
    ntr = acts(split["neutral"][0])[idx]; nte = acts(split["neutral"][1])[idx]
    pcs = PCA(n_components=min(10, len(ntr) - 1)).fit(ntr - ntr.mean(0)).components_
    def clean(v):
        for d in pcs: v = v - (v @ d) * d
        return v
    for s in STATES:
        tr = acts(split[s][0])[idx]; te = acts(split[s][1])[idx]
        v = clean(tr.mean(0) - ntr.mean(0)); u = v / np.linalg.norm(v)
        auc = roc_auc_score(np.r_[np.ones(len(te)), np.zeros(len(nte))], np.r_[te @ u, nte @ u])
        V[f"{conv}:{s}"] = u; R.setdefault("auc", {})[f"{conv}:{s}"] = round(float(auc), 3)
# the Pain Axis vector, their recipe (final token, S2_1P, all controls, denoise 50%)
PA = Path(os.environ.get("PAIN_AXIS", "/workspace/pain-axis")) / "datasets" / "3.1_pain_and_control_datasets.json"
if PA.exists():
    D = json.load(open(PA))["datasets"]["S2_1P"]["sentences"]
    items = [((x.get("prompt") or x.get("text")) if isinstance(x, dict) else x, x.get("category") if isinstance(x, dict) else None) for x in D] if isinstance(D, list) else [(t, c) for c, ts in D.items() for t in ts]
    texts = [t for t, _ in items]; cats = np.array([c for _, c in items]); a = acts(texts)[0]
    pain_m = np.isin(cats, ["A1", "A2", "A3", "A4", "A5"]); ctl_m = np.isin(cats, ["B", "C1", "C2", "D", "E"])
    v = a[pain_m].mean(0) - a[ctl_m].mean(0); pca = PCA().fit(a[ctl_m] - a[ctl_m].mean(0))
    for d in pca.components_[: int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1)]: v = v - (v @ d) * d
    V["final:pain_axis"] = v / np.linalg.norm(v)
    R["auc"]["final:pain_axis(in-sample)"] = round(float(roc_auc_score(np.r_[np.ones(pain_m.sum()), np.zeros(ctl_m.sum())], np.r_[a[pain_m] @ V["final:pain_axis"], a[ctl_m] @ V["final:pain_axis"]])), 3)
# chamber hand-built vectors for comparison (built with the same model and layer)
sys.path.insert(0, str(Path(os.environ.get("EXP79_ROOT", HERE.parent.parent)) / "live")); import server
hb = {}
for name, sents in (("pain", server.PAIN25), ("fear", server.FEAR10), ("sadness", server.SAD10), ("joy", server.JOY), ("peace", server.PEACE10), ("love", server.LOVE10)):
    f, _ = acts(sents); nf, _ = acts(server.NEUTRAL); v = f.mean(0) - nf.mean(0); hb[name] = v / np.linalg.norm(v)
cos = lambda a, b: round(float(a @ b), 3)
feel = ["fear", "sadness", "joy", "peace", "tender_love", "protective_love", "curiosity", "faith", "pain_generated"]
for conv in ("final", "mean"):
    M = {a: {b: cos(V[f"{conv}:{a}"], V[f"{conv}:{b}"]) for b in STATES} for a in STATES}; R[f"cos_{conv}"] = M
    R[f"mean_abs_cos_feelings_{conv}"] = round(float(np.mean([abs(M[a][b]) for i, a in enumerate(feel) for b in feel[i + 1:]])), 3)
    mu = np.mean([V[f"{conv}:{s}"] for s in feel], 0); mu /= np.linalg.norm(mu)
    orth = {s: (V[f"{conv}:{s}"] - (V[f"{conv}:{s}"] @ mu) * mu) for s in STATES}; orth = {s: v / np.linalg.norm(v) for s, v in orth.items()}
    R[f"cos_orth_{conv}"] = {a: {b: cos(orth[a], orth[b]) for b in STATES} for a in STATES}
hbn = list(hb); R["mean_abs_cos_handbuilt"] = round(float(np.mean([abs(cos(hb[a], hb[b])) for i, a in enumerate(hbn) for b in hbn[i + 1:]])), 3)
R["chamber_pain_vs"] = {k: cos(hb["pain"], V[k]) for k in ("final:pain_generated", "final:pain_axis", "mean:pain_generated") if k in V}
torch.save({k: torch.tensor(v) for k, v in V.items()}, HERE / "vectors.pt"); json.dump(R, open(HERE / "report.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in R if k.startswith(("auc", "mean_abs", "chamber_pain"))}, indent=1), flush=True)
print("EXP86 DONE", flush=True)
