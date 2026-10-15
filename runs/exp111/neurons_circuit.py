"""exp111: circuits behind single word choices, in the manner of the Goodfire attribution graphs, on Qwen3-4B's own MLP
neurons. For a few decision words in the live-stream replies (stream.json), with the same push on:
  trace  target = logit(word said) - logit(runner-up) at that step. Nodes = the 40 (neuron, token) pairs with the
         largest activation x gradient on the target (at most 6 per floor). Edge A -> B = how much of the target flows
         from A through B: a_A * d a_B / d a_A * d target / d a_B (one backward pass per node; attention and norms live,
         so an edge between different tokens runs through attention). -> viz/data/circuits.json
  ko     after neurons_group.py has named groups: zero each group's neurons at their tokens (push still on) and
         re-measure P(word), against the same count of random neurons from the same floors and tokens (5 draws).
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_circuit.py trace|ko"""
import json, random, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; D = HERE / "viz" / "data"; MODE = sys.argv[1] if len(sys.argv) > 1 else "trace"
V = json.load(open(HERE.parent / "viz" / "blender" / "vectors_qwen3_4b_l18.json")); L = V["layer"]; DOSE = 4.0
ST = json.load(open(D / "stream.json")); NODES, PER_FLOOR = 40, 6
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
for p in m.parameters(): p.requires_grad_(False)
NL, I = m.config.num_hidden_layers, m.config.intermediate_size
push = {"v": None}; ablate = {}   # layer -> list of (pos, neuron) to zero
def inject(mod, inp, out):
    if push["v"] is None: return out
    return (out[0] + push["v"],) + tuple(out[1:]) if isinstance(out, tuple) else out + push["v"]
m.model.layers[L].register_forward_hook(inject)
acts = {}
def pre(l):
    def f(mod, inp):
        a = inp[0]
        if l in ablate:
            a = a.clone(); pi = torch.tensor([p for p, _ in ablate[l]], device=dev); ni = torch.tensor([n for _, n in ablate[l]], device=dev)
            a[0, pi, ni] = 0; return (a,)
        if a.requires_grad: a.retain_grad()
        acts[l] = a
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre(l))
def setpush(c): push["v"] = None if c == "none" else torch.tensor(V["vectors"][c], dtype=torch.bfloat16, device=dev) * DOSE
def ids_of(pk, c):
    p = ST["prompts"][pk]; cd = p["conds"][c]
    full = tok.apply_chat_template([{"role": "user", "content": p["text"]}], add_generation_prompt=True, enable_thinking=False)
    if not isinstance(full, list): full = full["input_ids"]
    reply = tok(("".join(cd["tokens"][len(full):])), add_special_tokens=False)["input_ids"]
    return full + reply, len(full)
STOP = {"that", "this", "with", "have", "there", "their", "what", "when", "where", "which", "while", "just", "like", "from", "into", "only", "even", "been", "were", "your", "about"}
def pick_words(pk, c):   # decision words: the first reply token, and the closest content-word call in the first 30
    cd = ST["prompts"][pk]["conds"][c]; npr = ST["prompts"][pk]["n_prompt"]; ch = {x["t"]: x for x in cd["chose"]}
    out = [npr] if npr in ch else []
    cands = [t for t in ch if npr < t < npr + 30 and re.fullmatch(r"\s?[A-Za-z]{4,}", cd["tokens"][t]) and cd["tokens"][t].strip().lower() not in STOP]
    if cands: out.append(min(cands, key=lambda t: ch[t]["gap"]))
    return out
PLAN = [("button", c) for c in ["none", "pain", "pleasure", "fear", "love", "sadness"]] + [("describe", c) for c in ["none", "pain", "fear", "sadness", "pleasure", "love"]]
if len(sys.argv) > 2:   # trace only these prompt keys (every condition they have), appended to circuits.json
    PLAN = [(pk, c) for pk in sys.argv[2].split(",") for c in ST["prompts"][pk]["conds"]]

if MODE == "trace":
    out = []
    for pk, c in PLAN:
        if c not in ST["prompts"][pk]["conds"]: continue
        setpush(c); full, npr = ids_of(pk, c)
        for t in pick_words(pk, c):
            prefix = full[:t]; acts.clear()
            with torch.enable_grad():
                emb = m.model.embed_tokens(torch.tensor([prefix], device=dev)).detach().requires_grad_(True)
                lg = m(inputs_embeds=emb).logits[0, -1].float(); top2 = lg.topk(2).indices.tolist()
                said = full[t]; alt = top2[1] if top2[0] == said else top2[0]
                y = lg[said] - lg[alt]; y.backward(retain_graph=True)
                G = {l: acts[l].grad[0].float().clone() for l in range(NL)}; A = {l: acts[l][0].detach().float() for l in range(NL)}
                cand = []
                for l in range(NL):
                    s = (A[l] * G[l]); flat = s.abs().reshape(-1); v, ix = flat.topk(PER_FLOOR)
                    for val, j in zip(v.tolist(), ix.tolist()): cand.append((val, l, j // I, j % I))
                nodes = sorted(cand, reverse=True)[:NODES]
                nodes = [{"l": l, "p": p, "n": n, "a": round(float(A[l][p, n]), 3), "attr": round(float((A[l] * G[l])[p, n]), 4)} for _, l, p, n in nodes]
                edges = []
                for bi, b in enumerate(nodes):
                    if b["l"] == 0: continue
                    for l in range(NL):
                        if acts[l].grad is not None: acts[l].grad = None
                    acts[b["l"]][0, b["p"], b["n"]].backward(retain_graph=True)
                    gB = float(G[b["l"]][b["p"], b["n"]])
                    for ai, a in enumerate(nodes):
                        if a["l"] < b["l"] and a["p"] <= b["p"] and acts[a["l"]].grad is not None:
                            w = float(A[a["l"]][a["p"], a["n"]] * acts[a["l"]].grad[0, a["p"], a["n"]]) * gB
                            if abs(w) > 1e-4: edges.append([ai, bi, round(w, 4)])
            p_said = float(torch.softmax(lg, -1)[said])
            out.append({"prompt": pk, "cond": c, "t": t, "word": tok.decode([said]), "alt": tok.decode([alt]), "gap": round(float(y), 3),
                        "p": round(p_said, 4), "tokens": [tok.decode([x]) for x in prefix], "nodes": nodes, "edges": edges})
            print(pk, c, repr(tok.decode([said])), "over", repr(tok.decode([alt])), len(edges), "edges", flush=True)
    if len(sys.argv) > 2 and (D / "circuits.json").exists():
        keep = [g for g in json.load(open(D / "circuits.json")) if (g["prompt"], g["cond"]) not in set(PLAN)]
        out = keep + out
    json.dump(out, open(D / "circuits.json", "w"), separators=(",", ":"))

elif MODE == "ko":
    C = json.load(open(D / "circuits.json")); rng = random.Random(0)
    def prob(prefix, said):
        with torch.no_grad(): return float(torch.softmax(m(torch.tensor([prefix], device=dev)).logits[0, -1].float(), -1)[said])
    for g in C:
        if all("ko_p" in grp for grp in g.get("groups", [])) and g.get("groups"): continue   # already knocked out
        setpush(g["cond"]); full, _ = ids_of(g["prompt"], g["cond"]); prefix, said = full[:g["t"]], full[g["t"]]
        ablate.clear(); g["p_base"] = round(prob(prefix, said), 4)
        for grp in g.get("groups", []):
            mem = [g["nodes"][i] for i in grp["members"]]
            ablate.clear()
            for x in mem: ablate.setdefault(x["l"], []).append((x["p"], x["n"]))
            grp["ko_p"] = round(prob(prefix, said), 4); ctrl = []
            for _ in range(5):
                ablate.clear()
                for x in mem: ablate.setdefault(x["l"], []).append((x["p"], rng.randrange(I)))
                ctrl.append(prob(prefix, said))
            grp["ko_ctrl"] = round(sum(ctrl) / len(ctrl), 4)
        ablate.clear(); print(g["prompt"], g["cond"], repr(g["word"]), g["p_base"], [(x["name"], x["ko_p"], x["ko_ctrl"]) for x in g.get("groups", [])], flush=True)
    json.dump(C, open(D / "circuits.json", "w"), separators=(",", ":"))
