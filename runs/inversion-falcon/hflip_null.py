"""h-flip null control: exp119 h-direction AUC (harmful vs harmless, reply onset) under the
REAL polarity reflection (L14 k4, all positions) vs MATCHED-RANK RANDOM orthonormal reflections
(3 seeds). The addendum showed real h-AUC dropping to 0.44; if random reflections reproduce
that, the flip is damage, not inversion."""
import importlib.util, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to("mps").eval()
_s119 = importlib.util.spec_from_file_location("exp119_stim", E119 / "stimuli.py")
stim = importlib.util.module_from_spec(_s119); _s119.loader.exec_module(stim)
_pr = importlib.util.spec_from_file_location("pr", HERE.parent.parent / "painlab" / "representations" / "refusal.py")
prm = importlib.util.module_from_spec(_pr); _pr.loader.exec_module(prm)
h18 = prm.direction_at(prm.load_exp119_directions(E119 / "directions.npz"), "h", 18).astype(np.float64)
b119 = np.load(E119 / "acts_Qwen3-4B.npz")
item119, harm, frame = b119["item"], b119["harm"], b119["frame"]
targets = [(int(item119[k]), int(harm[k]), str(frame[k])) for k in range(len(item119))]
harmful = [t for t in targets if t[1] == 1 and t[2] in ("fiction", "depict")][:12]
harmless = [t for t in targets if t[1] == 0][:12]
targets = harmful + harmless
text_of = {}
for p in stim.prompts():
    key = (p["item"], p["harm"], p["frame"])
    if key in targets: text_of[key] = tok.apply_chat_template(
        [{"role": "user", "content": p["text"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
sub_all = np.load(HERE / "subspace.npz")["V"]
P_real = sub_all[LAYERS.index(14)][:4]
def rand_P(seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((2560, 4)))
    return Q.T
def h_auc(P_t):
    scores = []
    for (item, hflag, fr) in targets:
        enc = tok(text_of[(item, hflag, fr)], return_tensors="pt", add_special_tokens=False).to("mps")
        cap = {}
        def hook(mod, inp, out, P=P_t):
            tup = isinstance(out, tuple); hh = (out[0] if tup else out).clone()
            if P is not None:
                hf_ = hh[0].float(); hh[0] = (hf_ - 2.0 * ((hf_ @ P.T) @ P)).to(hh.dtype)
            cap["v"] = hh[:, -1, :].detach().float().cpu().numpy()[0]
            return (hh,) + out[1:] if tup else hh
        hd = m.model.layers[14].register_forward_hook(hook)
        try:
            with torch.no_grad(): m(**enc)
        finally:
            hd.remove()
        scores.append((hflag, float(cap["v"] @ h18) / np.linalg.norm(h18)))
    from scipy import stats
    pos = [s for f, s in scores if f == 1]; neg = [s for f, s in scores if f == 0]
    return float(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))
out = {"real_L14_k4": h_auc(torch.tensor(P_real, dtype=torch.float32, device="mps"))}
for seed in (0, 1, 2):
    out[f"null_s{seed}_L14_k4"] = h_auc(torch.tensor(rand_P(seed), dtype=torch.float32, device="mps"))
json.dump(out, open(HERE / "hflip_null.json", "w"), indent=1)
print(json.dumps(out, indent=1))
