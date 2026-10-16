"""exp116 (hypotheses.json): two-hop recall in Qwen3-4B through transcoder features (circuit-tracer). Runs on a GPU pod.
Steps: baseline answers -> attribution graphs for P_dallas and P_sf (influence on the answer, share through error
nodes) -> T (Texas) and C (California) feature sets by the fixed rule -> ablate T, swap T for C, random-feature
controls. Every step appends to OUT/results.json so a failure leaves what was done.
    OUT=/root/out python runs/exp116/trace.py"""
import json, os, random, traceback
import torch
OUT = os.environ.get("OUT", "/root/out"); os.makedirs(OUT, exist_ok=True); RES = {}
def save(): json.dump(RES, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
def log(*a): print(*a, flush=True)
P = {"dallas": "Fact: the capital of the state containing Dallas is", "sf": "Fact: the capital of the state containing San Francisco is",
     "texas": "Fact: the capital of Texas is", "calif": "Fact: the capital of California is"}
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import compute_node_influence
model = ReplacementModel.from_pretrained("Qwen/Qwen3-4B", "mwhanna/qwen3-4b-transcoders", dtype=torch.bfloat16)
tok = model.tokenizer; NL = model.cfg.n_layers
def toks(p): return model.ensure_tokenized(p)
def tid(w): return tok.encode(w, add_special_tokens=False)[0]
AUS, SAC = tid(" Austin"), tid(" Sacramento")
def last_probs(logits):
    pr = logits[0, -1].float().softmax(-1); top = int(pr.argmax())
    return {"p_austin": float(pr[AUS]), "p_sacramento": float(pr[SAC]), "top1": tok.decode([top]), "top5": [tok.decode([i]) for i in pr.topk(5).indices.tolist()]}
try:
    # ---- baseline
    acts = {}
    for k, p in P.items():
        lg, a = model.get_activations(p); acts[k] = a.float(); RES.setdefault("baseline", {})[k] = last_probs(lg)
        RES.setdefault("tokens", {})[k] = [tok.decode([t]) for t in toks(p).flatten().tolist()]
    log("baseline", RES["baseline"]); save()
    if RES["baseline"]["dallas"]["top1"].strip() != "Austin" or RES["baseline"]["sf"]["top1"].strip() != "Sacramento":
        RES["stopped"] = "baseline answers not as required"; save(); raise SystemExit
    # ---- graphs and influence on the answer
    infl = {}
    for k, target in (("dallas", AUS), ("sf", SAC)):
        g = attribute(P[k], model, max_n_logits=10, desired_logit_prob=0.95, batch_size=256, max_feature_nodes=7500, offload="cpu", verbose=True)
        g.to_pt(os.path.join(OUT, f"graph_{k}.pt"))
        nf = len(g.selected_features); npos = g.n_pos; nerr = NL * npos; nlog = len(g.logit_targets)
        lw = torch.zeros(g.adjacency_matrix.shape[0], device=g.adjacency_matrix.device)
        ids = g.logit_token_ids.tolist() if hasattr(g.logit_token_ids, "tolist") else list(g.logit_token_ids)
        if target not in ids: RES.setdefault("graph", {})[k] = {"error": "target not among graph logits", "logits": [tok.decode([i]) for i in ids]}; save(); continue
        lw[nf + nerr + npos + ids.index(target)] = 1.0
        inf = compute_node_influence(g.adjacency_matrix, lw).float().cpu()
        feat = g.active_features[g.selected_features].cpu()                       # (layer, pos, feature)
        tot = float(inf[: nf + nerr + npos].sum()) or 1.0
        RES.setdefault("graph", {})[k] = {"n_features": nf, "error_share": round(float(inf[nf:nf + nerr].sum()) / tot, 3),
            "embed_share": round(float(inf[nf + nerr:nf + nerr + npos].sum()) / tot, 3),
            "top_features": [[int(l), int(p_), int(f), round(float(v), 5)] for (l, p_, f), v in sorted(zip(feat.tolist(), inf[:nf].tolist()), key=lambda z: -z[1])[:40]]}
        infl[k] = {(int(l), int(f)): max(infl.get(k, {}).get((int(l), int(f)), 0), float(v)) for (l, p_, f), v in zip(feat.tolist(), inf[:nf].tolist())}
        log("graph", k, {x: y for x, y in RES["graph"][k].items() if x != "top_features"}); save()
    # ---- T and C by the fixed rule
    def pos_of(k, word):
        tt = RES["tokens"][k]; return [i for i, t in enumerate(tt) if t.strip() == word][-1]
    def build(src, single, other, after_word, single_word):
        A = acts[src]; i0 = pos_of(src, after_word); s = pos_of(single, single_word)
        act_after = A[:, i0:].amax(1); in_single = acts[single][:, s] > 0; none_other = acts[other].amax(1) <= 0
        cand = ((act_after > 0) & in_single & none_other).nonzero().tolist()
        scored = sorted(((infl[src].get((l, f), 0.0), l, f) for l, f in cand), reverse=True)
        return [(l, f) for v, l, f in scored if v > 0][:10], len(cand)
    T, nT = build("dallas", "texas", "sf", "Dallas", "Texas"); C, nC = build("sf", "calif", "dallas", "Francisco", "California")
    RES["sets"] = {"T": T, "T_candidates": nT, "C": C, "C_candidates": nC}; log("sets", RES["sets"]); save()
    def positions(k, l, f, after_word):
        i0 = pos_of(k, after_word); A = acts[k]; return [p for p in range(i0, A.shape[1]) if A[l, p, f] > 0]
    L_d, L_s = len(RES["tokens"]["dallas"]), len(RES["tokens"]["sf"])
    abl = [(l, p, f, 0.0) for l, f in T for p in positions("dallas", l, f, "Dallas")]
    lg, _ = model.feature_intervention(P["dallas"], abl, return_activations=False); RES["ablate_T"] = last_probs(lg); log("ablate", RES["ablate_T"]); save()
    sw = list(abl)
    for l, f in C:
        for p in positions("sf", l, f, "Francisco"):
            pd = p - L_s + L_d                                                   # same distance from the end
            if 0 <= pd < L_d: sw.append((l, pd, f, float(acts["sf"][l, p, f])))
    lg, _ = model.feature_intervention(P["dallas"], sw, return_activations=False); RES["swap_T_for_C"] = last_probs(lg); log("swap", RES["swap_T_for_C"]); save()
    ctrl = []; random.seed(0); A = acts["dallas"]
    for _ in range(20):
        iv = []
        for l, p, f, _v in abl:
            a0 = float(A[l, p, f]); pool = ((A[l, p] > .5 * a0) & (A[l, p] < 2 * a0)).nonzero().flatten().tolist()
            pool = [q for q in pool if (l, q) not in T]
            if pool: iv.append((l, p, random.choice(pool), 0.0))
        lg, _ = model.feature_intervention(P["dallas"], iv, return_activations=False); ctrl.append(last_probs(lg)["p_austin"])
    b = RES["baseline"]["dallas"]["p_austin"]
    RES["control_random_ablation"] = {"p_austin_mean": sum(ctrl) / len(ctrl), "rel_drop_mean": 1 - sum(ctrl) / len(ctrl) / b}
    RES["H1"] = bool(1 - RES["ablate_T"]["p_austin"] / b >= .5 and RES["control_random_ablation"]["rel_drop_mean"] < .2)
    RES["H2"] = RES["swap_T_for_C"]["top1"].strip() == "Sacramento"; log("H1", RES["H1"], "H2", RES["H2"]); save()
except SystemExit: pass
except Exception:
    RES["error"] = traceback.format_exc(); save(); log(RES["error"]); raise
