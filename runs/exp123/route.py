"""exp123 (hypotheses.json): the route of a swap. A W edit at block 8 (concept_swap.py, the site's engine) makes the
model answer about X as if it were Y. The edit sits only at the X mention, so the swap can reach the answer position
only through attention. Which heads carry it?
  per item: greedy first tokens t_X, t_Y; clean run (head outputs recorded); swap run with one backward pass
  (attribution per head) and the attention flow from the mention into the read position (attn_flow.Tap);
  then blocking runs: chosen heads emit their clean output at every position, B(S) = 1 - mean(E_S)/mean(E).
Writes runs/exp123/results.json (+ items.json, the per-item numbers).
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp123/route.py"""
import json, random, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from attn_flow import Tap
from concept_swap import Swapper
DATA = Path.home() / "repos/research/neuron_tower_data/4b"; H = json.load(open(HERE / "hypotheses.json"))["items"]
BLOCK, K = 8, 32; dev = "mps"; torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
for p in m.parameters(): p.requires_grad_(False)
tap = Tap(m, DATA / "heads.json"); tap.on = False                      # sets eager attention
sw = Swapper(m, tok, DATA)
c = m.config; NL, NH, HD = c.num_hidden_layers, c.num_attention_heads, c.head_dim
MODE = {"m": "off", "rec": {}, "z": {}, "clean": None, "S": {}, "pos": None}   # pos: None = every position, "read" / "other" (split.py)

def pre(l):
    def f(mod, inp):
        x = inp[0]; md = MODE["m"]
        if md == "rec": MODE["rec"][l] = x.detach().clone(); return None
        if md == "grad":
            MODE["rec"][l] = x.detach().clone(); z = torch.zeros_like(x, requires_grad=True); MODE["z"][l] = z; return (x + z,)
        if md == "restore" and MODE["S"].get(l):
            x = x.clone(); cl = MODE["clean"][l].to(dev)
            p = MODE["pos"]; P = slice(p, p + 1) if isinstance(p, int) else {None: slice(None), "read": slice(-1, None), "other": slice(None, -1)}[p]
            for h in MODE["S"][l]: x[:, P, h * HD:(h + 1) * HD] = cl[:, P, h * HD:(h + 1) * HD]
            return (x,)
        return None
    return f
for l in range(NL): m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(pre(l))

def ids_of(q):
    return tok.apply_chat_template([{"role": "user", "content": q + " Answer in a few words."}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=True)
def last_logits(ids):
    with torch.no_grad(): return m(torch.tensor([ids], device=dev), use_cache=False).logits[0, -1].float()

def build(x, y, q, split, attr, fam):
    qx, qy = q.replace("{X}", x), q.replace("{X}", y); ix, iy = ids_of(qx), ids_of(qy)
    sw.clear(); MODE["m"] = "off"
    tX, tY = int(last_logits(ix).argmax()), int(last_logits(iy).argmax())
    it = {"x": x, "y": y, "q": qx, "split": split, "attr": attr, "fam": fam, "tX": tok.decode([tX]), "tY": tok.decode([tY])}
    if tX == tY: it["drop"] = "same first token"; return it, None
    MODE["m"] = "rec"; MODE["rec"] = {}; lg = last_logits(ix); clean = {l: v.to("cpu") for l, v in MODE["rec"].items()}
    m_clean = float(lg[tY] - lg[tX])
    n = sw.set(f"W:{x}:{y}:{BLOCK}:1", ix)
    if not n: it["drop"] = "no mention"; sw.clear(); return it, None
    MODE["m"] = "grad"; MODE["rec"] = {}; MODE["z"] = {}
    with torch.enable_grad():
        lg = m(torch.tensor([ix], device=dev), use_cache=False).logits[0, -1].float(); met = lg[tY] - lg[tX]; met.backward()
    m_swap = float(met); g = np.zeros((NL, NH), np.float32)
    for l in range(NL):
        d = (clean[l].to(dev).float() - MODE["rec"][l].float()) * MODE["z"][l].grad.float()      # restoring head h: predicted change in m
        g[l] = d[0].view(-1, NH, HD).sum((0, 2)).cpu().numpy()
    MODE["m"] = "off"; MODE["z"] = {}; MODE["rec"] = {}
    pos = sw.mentions(ix, x); tap.reset(); tap.on = True
    with torch.no_grad(): m(torch.tensor([ix], device=dev), use_cache=True)
    tap.on = False
    fl = np.zeros((NL, NH), np.float32); share_m = np.zeros((NL, NH), np.float32)
    for l in range(NL):
        Fq = tap.F[l][:, -1, :]                                                                  # heads x k, into the read position
        fl[l] = Fq[:, pos].sum(1).cpu().numpy()
        after = Fq[:, min(pos):].sum(1).clamp(min=1e-9); share_m[l] = (Fq[:, pos].sum(1) / after).cpu().numpy()
    tap.reset(); sw.clear()
    it.update({"m_clean": m_clean, "m_swap": m_swap, "E": m_swap - m_clean, "mentions": n, "T": len(ix)})
    if it["E"] < 2: it["drop"] = "E < 2"
    return it, {"ids": ix, "clean": clean, "attr": g, "flow": fl, "share_m": share_m, "tX": tX, "tY": tY}

def blocked(item, aux, S):
    MODE["m"] = "restore"; MODE["clean"] = aux["clean"]; MODE["S"] = S
    sw.set(f"W:{item['x']}:{item['y']}:{BLOCK}:1", aux["ids"]); lg = last_logits(aux["ids"])
    MODE["m"] = "off"; MODE["S"] = {}; sw.clear()
    return float(lg[aux["tY"]] - lg[aux["tX"]]) - item["m_clean"], int(lg.argmax()) == aux["tX"]
def as_set(flat):
    S = {}
    for k in flat: S.setdefault(int(k) // NH, []).append(int(k) % NH)
    return S
def top(score, k): return list(np.argsort(-score.ravel())[:k])
def matched(chosen, rng):
    out = set()
    for k in chosen:
        l = int(k) // NH
        while True:
            r = l * NH + rng.randrange(NH)
            if r not in out and r not in chosen: out.add(r); break
    return list(out)
def B(items, S):
    E = np.mean([i["E"] for i, a in items]); res = [blocked(i, a, S) for i, a in items]
    return 1 - np.mean([r[0] for r in res]) / E, float(np.mean([r[1] for r in res]))

t0 = time.time(); ITEMS = []
for split, pairs in (("train", H["pairs_train"]), ("test", H["pairs_test"])):
    for x, y in pairs:
        for attr, qs in (("A", H["attributes_A"]), ("B", H["attributes_B"])):
            for q in qs:
                for w in (x, y): sw.prepare(f"W:{w}:{w}:{BLOCK}:1")
                ITEMS.append(build(x, y, q, split, attr, "places"))
        print(split, x, "->", y, f"{time.time()-t0:.0f}s", flush=True)
for fam, F in H["other_families"].items():
    for x, y in F["pairs"]: sw.prepare(f"W:{x}:{y}:{BLOCK}:1"); ITEMS.append(build(x, y, F["q"], "family", fam, fam))
keep = [(i, a) for i, a in ITEMS if a is not None and "drop" not in i]
sel = lambda **kw: [(i, a) for i, a in keep if all(i[k] == v for k, v in kw.items())]
tr, te = sel(split="train"), sel(split="test")
print("kept", len(keep), "of", len(ITEMS), "| train", len(tr), "test", len(te), flush=True)
attr_score = lambda its: -np.sum([a["attr"] for i, a in its], 0)                                 # most negative predicted change = most blocking
flow_score = lambda its: np.sum([a["flow"] for i, a in its], 0)
R = {"n": {"items": len(ITEMS), "kept": len(keep), "train": len(tr), "test": len(te),
           "dropped": {d: sum(1 for i, a in ITEMS if i.get("drop") == d) for d in ("same first token", "no mention", "E < 2")}},
     "mean_E_test": float(np.mean([i["E"] for i, a in te]))}
all_heads = {l: list(range(NH)) for l in range(NL)}
R["sanity_all_heads_B"] = B(te[:6], all_heads)[0]; print("sanity (all heads restored, should be 1):", R["sanity_all_heads_B"], flush=True)
SA = attr_score(tr); SF = flow_score(tr); rng = random.Random(1); R["curve"] = {}
for k in (8, 16, 32, 64, 128):
    ch = top(SA, k); b, back = B(te, as_set(ch))
    draws = [B(te, as_set(matched(ch, rng)))[0] for _ in range(20 if k == K else 5)]
    R["curve"][k] = {"attribution": b, "returns_to_X": back, "random_mean": float(np.mean(draws)), "random_max": float(np.max(draws))}
    print(f"k={k:3d}  attribution B={b:.3f} (back to X {back:.2f})  random {np.mean(draws):.3f} (max {np.max(draws):.3f})", flush=True)
chF = top(SF, K); R["flow_B"] = B(te, as_set(chF))[0]
SAa, SAb = attr_score(sel(split="train", attr="A")), attr_score(sel(split="train", attr="B")); teB = sel(split="test", attr="B")
R["H2"] = {"A_selected_on_B": B(teB, as_set(top(SAa, K)))[0], "B_selected_on_B": B(teB, as_set(top(SAb, K)))[0]}
ch32 = top(SA, K); kinds = {(x["l"], x["h"]): x["kind"] for x in json.load(open(DATA / "heads.json"))["heads"]}
shm = np.mean([a["share_m"] for i, a in tr], 0)
R["route"] = [{"l": int(k) // NH, "h": int(k) % NH, "score": float(SA.ravel()[k]), "kind": kinds.get((int(k) // NH, int(k) % NH)),
               "flow_rank": int(np.where(np.argsort(-SF.ravel()) == k)[0][0]), "share_from_mention": float(shm.ravel()[k])} for k in ch32]
R["overlap_attr_flow_top32"] = len(set(ch32) & set(chF))
R["families"] = {}
for fam in H["other_families"]:
    its = sel(fam=fam)
    if its: R["families"][fam] = {"n": len(its), "country_route_B": B(its, as_set(ch32))[0],
                                  "random_B": float(np.mean([B(its, as_set(matched(ch32, rng)))[0] for _ in range(10)]))}
b32, r32 = R["curve"][K]["attribution"], R["curve"][K]["random_mean"]
R["H1"] = bool(b32 >= .5 and r32 <= .15); R["H2"]["pass"] = bool(R["H2"]["A_selected_on_B"] >= .7 * R["H2"]["B_selected_on_B"])
R["H3"] = bool(R["flow_B"] >= .5 * b32); R["seconds"] = time.time() - t0
json.dump(R, open(HERE / "results.json", "w"), indent=1)
json.dump([i for i, a in ITEMS], open(HERE / "items.json", "w"), indent=1, ensure_ascii=False)
np.savez_compressed(HERE / "scores.npz", attr_train=SA, flow_train=SF, share_m=shm)
print(json.dumps({k: R[k] for k in ("H1", "H2", "H3", "flow_B", "families", "overlap_attr_flow_top32")}, indent=1))
