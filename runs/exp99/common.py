"""exp99 shared pieces: Qwen3-8B (no chamber hook), the chamber's 8B vectors, the exp86 passage vectors, and the BatchTopK SAEs.
Layer k = output of decoder layer k (0-indexed) = hidden_states[k + 1] = the SAE's resid_post_layer_k."""
import json, os, sys
from pathlib import Path
import torch, transformers
os.environ.setdefault("CHAMBER_MODEL", "Qwen/Qwen3-8B"); os.environ.setdefault("CHAMBER_LAYER", "18"); os.environ.setdefault("CHAMBER_DEVICE", "mps")
HERE = Path(__file__).parent; ROOT = HERE.parent.parent; PAIN_AXIS = ROOT.parent / "pain-axis"
sys.path.insert(0, os.environ.get("WIREHEAD_LIVE") or str(ROOT / "live")); import server   # the relay's live/ (private repo since the split)
DEV, L, MID = server.DEVICE, server.LAYER, server.MODEL_ID
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

def load_model(upto=None, keep_from=None):
    """upto=k keeps decoder layers 0..k only and swaps the LM head for identity (feature extraction to layer k).
    keep_from=k keeps only layers k.. plus the final norm and LM head (the second half of a split forward); the
    8B in full does not fit beside this machine's other processes without thrashing swap."""
    tok = transformers.AutoTokenizer.from_pretrained(MID); tok.padding_side = "right"
    model = transformers.AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).eval()
    if upto is not None:
        model.model.layers = model.model.layers[:upto + 1]; model.config.num_hidden_layers = upto + 1
        model.lm_head = torch.nn.Identity(); model.model.norm = torch.nn.Identity()   # else hidden_states[-1] is layer k normed
    if keep_from is not None:
        model.model.layers = model.model.layers[keep_from:]; model.model.embed_tokens = None
    import gc; gc.collect()
    return tok, model.to(DEV)

def chamber_vectors(model, tok):
    """The live server's battery-minus-NEUTRAL vectors on this model and layer (1x = neutral norm / 4), cached."""
    f = CACHE / "chamber_vecs_8b.pt"
    if f.exists(): d = torch.load(f); return d["vecs"], d["scale"]
    vecs, scale = server.build_vectors(model, tok); torch.save({"vecs": vecs, "scale": scale}, f)
    return vecs, scale

def directions(model, tok, n_random=20):
    """Unit directions in the 8B's layer-18 residual space, keyed by name, plus the 1x scale."""
    vecs, S = chamber_vectors(model, tok)
    unit = lambda v: (v.float() / v.float().norm()).cpu()
    U = {f"chamber:{k}": unit(vecs[k]) for k in ("pain", "fear", "egg")}
    P = torch.load(HERE.parent / "exp86" / "vectors.pt")
    for k in ("final:pain_axis", "final:pain_generated", "final:fear", "final:sadness", "final:egg", "final:constipation", "final:trapped"):
        U[k] = unit(P[k])
    g = torch.Generator().manual_seed(95)
    for i in range(n_random): U[f"random:{i}"] = unit(torch.randn(U["chamber:pain"].shape[0], generator=g))
    return U, float(S)

class SAE:
    """dictionary_learning BatchTopKSAE at inference: ReLU(W_enc (x - b_dec) + b_enc), kept where above the learned threshold.
    Encoder weight on the device in fp16 (the 8B leaves ~8 GB of this 26 GB machine; fp32 encoders + decoders thrashed swap),
    pre-activations accumulated in fp32; decoder memory-mapped from disk on the CPU."""
    def __init__(self, layer):
        p = HERE / "sae" / "saes_Qwen_Qwen3-8B_batch_top_k" / f"resid_post_layer_{layer}" / "trainer_2" / "ae.pt"
        sd = torch.load(p, map_location="cpu", mmap=True)
        self.layer = layer
        self.W_enc = sd["encoder.weight"].to(DEV, torch.float16)     # [F, d]
        self.b_enc = sd["encoder.bias"].float().to(DEV)
        self.b_dec = sd["b_dec"].float().to(DEV)
        self.thr = float(sd["threshold"])
        self.W_dec = sd["decoder.weight"]                             # [d, F], fp32, mmap
        self.F = self.W_enc.shape[0]
    @torch.no_grad()
    def encode(self, x, chunk=512):
        out = []
        for i in range(0, len(x), chunk):
            z = torch.relu(((x[i:i + chunk].float().to(DEV) - self.b_dec).half() @ self.W_enc.T).float() + self.b_enc)
            out.append(z * (z > self.thr))
        return torch.cat(out)
    @torch.no_grad()
    def fve(self, x):
        """fraction of variance explained on x (sanity check that our activations match the SAE's training distribution)."""
        z = self.encode(x); xh = (z.cpu() @ self.W_dec.T) + self.b_dec.cpu(); x = x.float().cpu()
        return float(1 - ((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()), float((z > 0).sum(1).float().mean())

@torch.no_grad()
def token_acts(model, tok, texts, layers, bs=8, drop_first=True):
    """per-text list of [n_tokens, d] residuals for each layer (CPU, fp32). Position 0 (Qwen's attention-sink outlier) dropped."""
    out = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
        hs = model(**enc, output_hidden_states=True).hidden_states
        n = enc.attention_mask.sum(1).tolist()
        for l in layers:
            h = hs[l + 1].float().cpu()
            for j, k in enumerate(n): out[l].append(h[j, (1 if drop_first else 0):k])
    return out

def pain_axis_sets():
    D = json.load(open(PAIN_AXIS / "datasets" / "3.1_pain_and_control_datasets.json"))["datasets"]
    S = {k: [(x["prompt"], x["category"], x.get("set")) for x in D[k]["sentences"]] for k in ("S2_1P", "S2_3P", "Numb_1P")}
    E = json.load(open(PAIN_AXIS / "datasets" / "4.1_self_other_420_scenarios.json"))
    S["self_other"] = [(x["text"], f"{x['stratum']}|{x['perspective']}", x["category"]) for x in E]
    return S
