"""exp113: the Neuron Tower's puzzle gallery. Each puzzle is one neuron whose strongest firings look unrelated, the
reveal (what actually ties them together, or that nothing does), and test texts with their activation measured here
on Qwen3-4B (max over the text / the neuron's corpus max). Sources: gallery.json / gallery_mid.json (blind-test
factors, pre-registered and exploratory) and probes/*.json (hand probes, exploratory).
Writes runs/exp113/puzzles.json for the tower (viz/site/puzzles.json).
    HF_HUB_CACHE=... python runs/exp113/build_puzzles.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DATA as D, DEV, load
G = {c["k"]: c for f in ("gallery.json", "gallery_mid.json") for c in json.load(open(HERE / f))}
MAX = np.load(D / "corpus_all.npz")["max"]; I = MAX.shape[1]
P = [
 dict(l=28, n=5239, kind="a bridge by spelling", title="Genomes and nuclear power",
      question="Why would one neuron fire for genome research and for nuclear energy?",
      reveal="Both live in a nucleus: a genome in the cell's, nuclear energy in the atom's. But the model learned the word, not the idea. It fires on nucle- in any sense, even 'the nuclear family', and on genom-, yet stays dark for chromosomes or uranium.",
      probes=["L28_5239_nucleus.json", "L28_5239_nucleus2.json"]),
 dict(l=0, n=4654, kind="a tokenizer artifact", title="Freshwater and Darfur",
      question="What do a freshwater ecosystem and the Darfur conflict have in common?",
      reveal="Nothing about their meaning. The model splits words into pieces: 'Fresh|water', 'Dar|fur'. This neuron fires on 'water' and 'fur' only when they come glued onto the end of a longer word. A glass of water or a cat's fur leaves it almost silent.",
      probes=["L0_4654.json"]),
 dict(l=24, n=9150, kind="two ideas, one neuron", title="Film and protein",
      question="Is there a hidden link between cinema and nutrition?",
      reveal="No. This neuron simply holds two unrelated words: 'film' and 'protein', in any context, strongest in category lists. Words that merely look similar (cumin, vitamin, elm) don't touch it. This is superposition: more ideas than neurons, so some neurons carry two.",
      probes=["L24_9150.json"]),
 dict(l=5, n=2345, kind="a hidden structure", title="Skip to content",
      question="'Skip to content' on one website, a list of psychology fields on another. Why the same neuron?",
      reveal="It fires on the joints of navigation menus: the 'to' in 'Skip to…', the bars in 'Methods | Statistics | Clinical'. A recipe laid out as 'Jump to Ingredients | Jump to Instructions' lights it up; the same words written as a sentence don't."),
 dict(l=0, n=370, kind="a hidden structure", title="Clinical, and Free Pass",
      question="'Clinical' in a psychology menu and 'Pass' in an encyclopedia header. What's shared?",
      reveal="Position. Both are the last item of a run of links or labels right before the real content starts. Any site menu ('Home | About | Services | Contact |') does it; the article alone does not."),
 dict(l=10, n=456, kind="a hidden structure", title="In the summer, in a study",
      question="Summer Olympics, research news. The neuron fires on 'the' and 'a'. Why there?",
      reveal="It fires on the word right after a sentence that opens with 'In' or 'At'. 'In my kitchen…' lights it; 'The copper pots in my kitchen…' doesn't. It's tracking how the sentence began, not what it's about."),
 dict(l=14, n=2504, kind="a hidden structure", title="A comma and a line break",
      question="A comma in a health site's footer and a line break in a dictionary. Why do they fire together?",
      reveal="Both sit on the seam where ordinary text turns into reference metadata: a date line, a source, a definition. 'Passed down through generations,⏎Published: March 15…' fires it; the same facts in one sentence don't."),
]
tok, m = load(); cur = {}
def hook(l):
    def f(mod, inp): cur[l] = inp[0][0].float()
    return f
for l in range(m.config.num_hidden_layers): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l))
def act(text, l, n):
    ids = tok(text, add_special_tokens=False)["input_ids"][:200]
    with torch.no_grad(): m(torch.tensor([ids], device=DEV))
    a = cur[l][:, n]; j = int(a.argmax()); toks = [tok.decode([x]) for x in ids]
    return round(float(a[j]) / float(MAX[l, n]), 2), j, toks
out = []
for p in P:
    c = G.get(p["l"] * I + p["n"]); groups = []
    topics = ((c or {}).get("proposal") or {}).get("group_topics") or []
    for i, cl in enumerate((c or {}).get("clusters", [])[:3]):
        groups.append({"topic": topics[i] if i < len(topics) else "", "ex": [{"pre": x["pre"][-110:], "tok": x["tok"], "post": x["post"][:50], "v": x["v"]} for x in cl[:4]]})
    tests = []
    if p.get("probes"):
        for f in p["probes"]:
            for grp, texts in json.load(open(HERE / "probes" / f)).items():
                for t in texts: v, j, toks = act(t, p["l"], p["n"]); tests.append({"group": grp, "text": t, "v": v, "at": j, "toks": toks})
    else:
        fs = [f for f in c["factors"] if f.get("verified")]
        names = {"positives": "with the hidden factor", "near_a": "the same texts, factor removed", "near_b": "same topics, no factor"}
        for kind in ("positives", "near_a", "near_b"):
            for r in fs[0]["tests"][kind][:4]:
                v, j, toks = act(r["text"], p["l"], p["n"]); tests.append({"group": names[kind], "text": r["text"], "v": v, "at": j, "toks": toks})
        p["factor"] = fs[0]["factor"]
    p.pop("probes", None); p.update(k=p["l"] * I + p["n"], groups=groups, tests=tests, source="blind test (pre-registered)" if c and c in json.load(open(HERE / "gallery.json")) else "exploratory")
    out.append(p); print(p["title"], len(groups), "groups,", len(tests), "tests", flush=True)
json.dump({"model": "Qwen/Qwen3-4B", "fire": 0.30, "puzzles": out}, open(HERE / "puzzles.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("wrote puzzles.json")
