"""exp124 (hypotheses.json): is the exp123 swap route a knowledge tract? On clean questions about 12 entities not used in
exp123, the 16 mover heads (exp123 split.json) are mean-ablated or interchanged at the read position only.
Reuses exp123/route.py's model, hooks and helpers. Writes runs/exp124/results.json.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp124/tract.py"""
import json
from pathlib import Path
E123 = Path(__file__).parent.parent / "exp123"
exec((E123 / "route.py").read_text().split("t0 = time.time(); ITEMS")[0])
HERE = Path(__file__).parent; P4 = json.load(open(HERE / "hypotheses.json")); I = P4["items"]
MOV = [l * NH + h for l, h in json.load(open(E123 / "split.json"))["movers"]]
ENT, ATT = I["entities (none used in exp123)"], I["attributes"]

def rec(ids):
    MODE["m"] = "rec"; MODE["rec"] = {}; lg = last_logits(ids); MODE["m"] = "off"
    return lg, {l: v[0, -1].float().cpu() for l, v in MODE["rec"].items()}
def patched(ids, S, src, n_new=1):
    """heads in S emit src[l] (their read-position output from elsewhere) at the read position; greedy n_new tokens"""
    full = {l: torch.zeros(1, len(ids), v.shape[0], dtype=torch.bfloat16) for l, v in src.items()}
    for l in full: full[l][0, -1] = src[l]
    MODE["m"] = "restore"; MODE["clean"] = full; MODE["S"] = S; MODE["pos"] = len(ids) - 1; out = []; cur = list(ids)
    for _ in range(n_new):
        lg = last_logits(cur); t = int(lg.argmax()); out.append(t); cur.append(t)
    MODE["m"] = "off"; MODE["S"] = {}; MODE["pos"] = None
    return out

items = []
for e in ENT:
    for a in ATT: items.append({"kind": "entity", "ent": e, "att": a, "q": a.replace("{X}", e)})
for c, qs in I["non-entity controls"].items():
    for q in qs: items.append({"kind": c, "q": q})
for it in items:
    it["ids"] = ids_of(it["q"]); lg, it["last"] = rec(it["ids"]); it["tok0"] = int(lg.argmax())
ents = [it for it in items if it["kind"] == "entity"]; ctrl = [it for it in items if it["kind"] != "entity"]
tot = {l: sum(it["last"][l] for it in ents) for l in range(NL)}
def mean_excl(e):
    mine = [it for it in ents if it.get("ent") == e]; n = len(ents) - len(mine)
    return {l: (tot[l] - sum(it["last"][l] for it in mine)) / n for l in range(NL)}
MEAN = {e: mean_excl(e) for e in ENT}; MEAN_ALL = {l: tot[l] / len(ents) for l in range(NL)}
def acc_ablate(its, heads):
    S = as_set(heads); return float(np.mean([patched(it["ids"], S, MEAN.get(it.get("ent"), MEAN_ALL))[0] == it["tok0"] for it in its]))
part = {(it["ent"], it["att"]): it for it in ents}
inter = [(it, part[(ENT[(ENT.index(it["ent"]) + 1) % len(ENT)], it["att"])]) for it in ents]
inter = [(a, b) for a, b in inter if a["tok0"] != b["tok0"]]
def interchange(heads):
    S = as_set(heads); return float(np.mean([patched(a["ids"], S, b["last"])[0] == b["tok0"] for a, b in inter]))
rng = random.Random(3); R = {"n": {"entity": len(ents), "control": len(ctrl), "interchange_kept": len(inter)}}
R["H1"] = {"movers_acc": acc_ablate(ents, MOV), "random_acc": float(np.mean([acc_ablate(ents, matched(MOV, rng)) for _ in range(20)]))}
print("H1", R["H1"], flush=True)
R["H2"] = {"movers_to_partner": interchange(MOV), "random_to_partner": float(np.mean([interchange(matched(MOV, rng)) for _ in range(20)]))}
print("H2", R["H2"], flush=True)
R["H3"] = {"controls_acc": acc_ablate(ctrl, MOV), **{c: acc_ablate([it for it in ctrl if it["kind"] == c], MOV) for c in I["non-entity controls"]},
           "controls_random_acc": float(np.mean([acc_ablate(ctrl, matched(MOV, rng)) for _ in range(10)]))}
print("H3", R["H3"], flush=True)
R["H1"]["pass"] = R["H1"]["movers_acc"] <= .5 and R["H1"]["random_acc"] >= .9
R["H2"]["pass"] = R["H2"]["movers_to_partner"] >= .5 and R["H2"]["random_to_partner"] <= .05
R["H3"]["pass"] = R["H3"]["controls_acc"] >= .8
R["k"] = {k: acc_ablate(ents, MOV[:k]) for k in (4, 8)}; print("k", R["k"], flush=True)
R["examples"] = []
for it in ents[::7][:8] + ctrl[::6]:
    clean = tok.decode(patched(it["ids"], {}, MEAN_ALL, 8)); abl = tok.decode(patched(it["ids"], as_set(MOV), MEAN.get(it.get("ent"), MEAN_ALL), 8))
    R["examples"].append({"q": it["q"], "clean": clean, "movers_ablated": abl})
for a, b in inter[::9][:6]:
    R["examples"].append({"q": a["q"], "interchange_from": b["ent"], "clean": tok.decode(patched(a["ids"], {}, MEAN_ALL, 8)),
                          "interchanged": tok.decode(patched(a["ids"], as_set(MOV), b["last"], 8))})
for x in R["examples"]: print(x, flush=True)
# exploratory: exp123 H2 sharpened, separate A / B selections on exp123 training items
H23 = json.load(open(E123 / "hypotheses.json"))["items"]; tr = []
for x, y in H23["pairs_train"]:
    for attr, qs in (("A", H23["attributes_A"]), ("B", H23["attributes_B"])):
        for q in qs:
            sw.prepare(f"W:{x}:{y}:{BLOCK}:1"); i, a = build(x, y, q, "train", attr, "places")
            if a is not None and "drop" not in i: tr.append((attr, a["attr"]))
SA = -np.sum([g for at, g in tr if at == "A"], 0).ravel(); SB = -np.sum([g for at, g in tr if at == "B"], 0).ravel()
tA, tB = set(np.argsort(-SA)[:16]), set(np.argsort(-SB)[:16])
R["exp123_AB_top16_overlap"] = {"overlap": len(tA & tB), "chance": round(16 * 16 / (NL * NH), 2),
                                "A": sorted((int(k) // NH, int(k) % NH) for k in tA), "B": sorted((int(k) // NH, int(k) % NH) for k in tB)}
print("AB overlap", R["exp123_AB_top16_overlap"]["overlap"], flush=True)
json.dump(R, open(HERE / "results.json", "w"), indent=1, ensure_ascii=False)
