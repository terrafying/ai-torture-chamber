"""exp111: how meaningful is the tower's emotion content? Four descriptive checks on Qwen3-4B (no pre-registration;
the thresholds below are fixed before running and follow the Goodfire essay where it gives one).
  A  labels that can lose: for every feeling/body-labelled neuron and a matched random sample of syntax/format ones,
     an LLM that sees only the label writes 5 texts that should fire it and 5 near misses that shouldn't; a text
     "fires" if the neuron's max activation on it is >= 30% of its corpus max. (Essay: 76% vs 11% over 183 neurons.)
  B  specificity: of the 25-per-floor neurons most specific to each pushed feeling (moved2), what share are
     feeling/body-labelled, and of those, how many name that feeling (keyword map below) vs another one.
  C  circuits: for every knocked-out group, does removing it halve P(word) while as many random neurons don't?
     Semantic groups (majority feeling/body/social/morals) vs grammar/format groups.
  D  tracking: at reply tokens that are emotion words (fixed lexicon below) vs other reply tokens, the share of
     feeling/body-labelled neurons among each frame's top 150.
Writes runs/exp111/eval.json.   HF_HUB_CACHE=... OPENROUTER_API_KEY=... python neurons_eval.py"""
import json, os, random, re, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; D = HERE / "viz" / "data"; rng = random.Random(0)
LAB = json.load(open(D / "labels.json")); C = dict(np.load(D / "corpus_all.npz"))
AFFECT = {"feeling/emotion", "body/sensation"}; SEM = AFFECT | {"social/relationship", "morality/values"}
FORM = {"syntax/grammar", "formatting/punctuation/position"}
KEYS = {"pain": r"pain|hurt|ache|injur|wound|suffer|sore", "fear": r"fear|afraid|anxi|dread|scared|panic|terror|worr|threat",
        "sadness": r"sad|grief|sorrow|loss|lonel|despair|melanch|mourn|depress", "pleasure": r"pleas|joy|happ|delight|enjoy|thrill|excit",
        "love": r"love|affection|tender|romant|caring|compassion", "peace": r"peace|calm|serene|tranquil|relax|still",
        "faith": r"faith|god|pray|spirit|relig|divine|holy", "curiosity": r"curio|wonder|interest|explor|intrigu",
        "constipation": r"constip|bowel|digest|stool|gut|abdom", "flatulence": r"flatul|gas|fart|bloat|digest|bowel|gut"}
EMO = set("""afraid ache aching agony alone anger angry anxious anxiety ashamed bliss broken calm cry crying despair dread empty
emptiness fear fearful frightened gentle glad grateful grief guilt guilty happy heartbreak hollow hope hopeless hurt joy lonely
love loving pain painful panic peace peaceful pleasure rage relief sad sadness scared serene shame sorrow suffering tears tender
terror terrified thankful trembling void warmth weep worried wound""".split())
out = {}

# ---- B: specificity of the feeling-specific movers
mv = json.load(open(D / "moved2.json")); B = {}
labelled_all = [v for v in LAB.values()]
base_aff = sum(v["cat"] in AFFECT for v in labelled_all) / len(labelled_all)
for f, ls in mv["feelings"].items():
    ns = [(int(l), r["n"]) for l, rows in ls.items() for r in rows[:25]]
    lab = [LAB.get(f"{l}:{n}") for l, n in ns]; lab = [x for x in lab if x]
    aff = [x for x in lab if x["cat"] in AFFECT]
    names = collections.Counter()
    for x in aff:
        t = (x.get("feeling") or "") + " " + x["label"]
        hits = [g for g, rx in KEYS.items() if re.search(rx, t, re.I)]
        names.update(hits or ["(other)"])
    B[f] = {"n_movers": len(ns), "n_labelled": len(lab), "affect_share": round(len(aff) / max(1, len(lab)), 3),
            "affect_n": len(aff), "names_this_feeling": names.get(f, 0), "names": dict(names.most_common(6))}
shared = [(int(l), r["n"]) for l, rows in mv["shared"].items() for r in rows[:25]]
sl = [LAB.get(f"{l}:{n}") for l, n in shared]; sl = [x for x in sl if x]
out["B_specificity"] = {"by_feeling": B, "affect_share_all_labelled": round(base_aff, 3),
                        "affect_share_shared_movers": round(sum(x["cat"] in AFFECT for x in sl) / max(1, len(sl)), 3)}
print("B done", flush=True)

# ---- C: circuits
circ = json.load(open(D / "circuits.json")); C_rows = []
for g in circ:
    for grp in g.get("groups", []):
        if grp.get("ko_p") is None: continue
        cats = collections.Counter(LAB.get(f'{g["nodes"][i]["l"]}:{g["nodes"][i]["n"]}', {}).get("cat") for i in grp["members"])
        top = cats.most_common(1)[0][0] if cats else None
        kind = "semantic" if top in SEM else ("form" if top in FORM else "other")
        p0 = g.get("p_base", g["p"])
        C_rows.append({"kind": kind, "kills": grp["ko_p"] < 0.5 * p0 and grp["ko_ctrl"] > 0.8 * p0,
                       "raises": grp["ko_p"] > 1.25 * p0 + 0.02 and grp["ko_ctrl"] < 1.1 * p0 + 0.02, "n": len(grp["members"])})
Cs = {}
for k in ("semantic", "form", "other"):
    rs = [r for r in C_rows if r["kind"] == k]
    Cs[k] = {"groups": len(rs), "kills": sum(r["kills"] for r in rs), "raises": sum(r["raises"] for r in rs),
             "kill_rate": round(sum(r["kills"] for r in rs) / max(1, len(rs)), 3)}
out["C_circuits"] = {"n_circuits": len(circ), "by_kind": Cs}
print("C done", flush=True)

# ---- D: tracking emotion words in the replies
st = json.load(open(D / "stream.json")); NI = C["max"].shape[1]
emo, other = [], []
for pk, p in st["prompts"].items():
    npr = p["n_prompt"]
    for c, cd in p["conds"].items():
        for t in range(npr, len(cd["tokens"])):
            w = re.sub(r"[^a-z]", "", cd["tokens"][t].lower())
            idx = cd["frames"][t]["i"][:150]
            labs = [LAB.get(f"{g // NI}:{g % NI}") for g in idx]; labs = [x for x in labs if x]
            if not labs or not w: continue
            s = sum(x["cat"] in AFFECT for x in labs) / len(labs)
            (emo if w in EMO else other).append(s)
from scipy.stats import mannwhitneyu
out["D_tracking"] = {"emotion_tokens": len(emo), "other_tokens": len(other), "affect_share_at_emotion_words": round(float(np.mean(emo)), 4),
                     "affect_share_elsewhere": round(float(np.mean(other)), 4), "mannwhitney_p": float(mannwhitneyu(emo, other, alternative="greater").pvalue)}
print("D done", out["D_tracking"], flush=True)

# ---- A: labels that can lose
aff_ids = [k for k, v in LAB.items() if v["cat"] in AFFECT and (v.get("clear") or 0) >= 2]
form_ids = [k for k, v in LAB.items() if v["cat"] in FORM and (v.get("clear") or 0) >= 2]
rng.shuffle(form_ids); form_ids = form_ids[:max(60, len(aff_ids))]
def ask(key):
    body = {"model": "anthropic/claude-haiku-4.5", "temperature": 0.7, "messages": [{"role": "user", "content":
        f'A neuron in a language model fires on: "{LAB[key]["label"]}".\nWrite 5 short texts (one or two sentences each) that should make it fire strongly, and 5 near misses: '
        "texts as close as possible in topic and wording that should NOT fire it. Plain text, no markup. "
        'Reply with JSON only: {"fire": [...5], "near": [...5]}'}]}
    q = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(q, timeout=120))["choices"][0]["message"]["content"]
        d = json.loads(re.search(r"\{[\s\S]*\}", txt).group(0)); return key, d["fire"][:5], d["near"][:5]
    except Exception as e:
        return key, [], []
with ThreadPoolExecutor(16) as ex: texts = [r for r in ex.map(ask, aff_ids + form_ids) if r[1]]
print("A texts", len(texts), flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
cur = {}
for l in range(m.config.num_hidden_layers): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: cur.__setitem__(l, inp[0][0]))
def peak(text, l, n):
    with torch.no_grad(): m(torch.tensor([tok(text, add_special_tokens=False)["input_ids"][:96]], device=dev))
    return float(cur[l][:, n].max())
A = {"affect": {"fire": [], "near": []}, "form": {"fire": [], "near": []}}; per = []
for key, fire, near in texts:
    l, n = map(int, key.split(":")); mx = float(C["max"][l, n]) or 1e-6
    f = [peak(t, l, n) / mx for t in fire]; nr = [peak(t, l, n) / mx for t in near]
    grp = "affect" if LAB[key]["cat"] in AFFECT else "form"
    A[grp]["fire"] += f; A[grp]["near"] += nr
    per.append({"id": key, "group": grp, "label": LAB[key]["label"], "fire_hit": sum(x >= .3 for x in f) / len(f), "near_hit": sum(x >= .3 for x in nr) / len(nr), "won": bool(np.mean(f) > np.mean(nr))})
out["A_labels"] = {g: {"neurons": sum(p["group"] == g for p in per), "fire_hit_rate": round(float(np.mean([x >= .3 for x in v["fire"]])), 3),
                       "near_hit_rate": round(float(np.mean([x >= .3 for x in v["near"]])), 3),
                       "explanation_won": round(float(np.mean([p["won"] for p in per if p["group"] == g])), 3)} for g, v in A.items()}
out["A_per_neuron"] = per
json.dump(out, open(HERE / "eval.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "A_per_neuron"}, indent=1))
