"""exp111: blind labels for the neurons the tower shows. An LLM sees only a neuron's strongest corpus firings on each
pole (from corpus_all.npz) and, for the last floors, the tokens it writes; it is NOT told which feeling or behaviour
selected the neuron. It returns a short label and one category from a fixed set, so "how many of fear's neurons are
about feelings at all" can be counted afterwards instead of assumed.
Which neurons: top weight writers/readers (top.json), feeling-specific movers (moved2.json), answer drivers (attr.json),
and neurons that recur in the live stream (stream.json). Labels cache in viz/data/labels.json (re-runs only add).
    OPENROUTER_API_KEY=... python neurons_label.py"""
import json, os, re, sys, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from tower_cfg import DATA as D, load
HERE = Path(__file__).parent; OUT = D / "labels.json"
MODEL = os.environ.get("LABEL_MODEL", "anthropic/claude-haiku-4.5"); BATCH = 12
CATS = ["feeling/emotion", "body/sensation", "social/relationship", "morality/values", "mind/thinking", "topic/subject matter",
        "names/entities", "syntax/grammar", "formatting/punctuation/position", "numbers/digits", "code/markup", "language/script", "unclear"]
C = dict(np.load(D / "corpus_all.npz"))   # materialize: indexing an NpzFile re-reads the whole array each time
docs = json.load(open(D / "corpus_ids.json")); CTX = int(C["ctx"]); NI = C["max"].shape[1]
tok = load(tokenizer_only=True)
want = collections.Counter()
for f, roles in json.load(open(D / "top.json")).items():
    for rows in roles.values():
        for r in rows[:15]: want[(r["l"], r["n"])] += 5
mv = json.load(open(D / "moved2.json"))
for f, ls in list(mv["feelings"].items()) + [("shared", mv["shared"])]:
    for l, rows in ls.items():
        for r in rows[:15]: want[(int(l), r["n"])] += 5
for c, cd in (json.load(open(D / "attr.json"))["conds"].items() if (D / "attr.json").exists() else []):   # the 4B's flat answer view only
    for l, info in cd["layers"].items():
        for r in info["rows"][:15]: want[(int(l), r["n"])] += 5
st = json.load(open(D / "stream.json"))
for p in st["prompts"].values():
    for cd in p["conds"].values():
        for fr in cd["frames"]:
            for g in fr["i"][:120]: want[divmod(g, NI)] += 1
        for ch in cd.get("chose", []):
            for g in ch["i"][:40]: want[divmod(g, NI)] += 3
targets = [k for k, c in want.most_common() if c >= 3][: int(sys.argv[1]) if len(sys.argv) > 1 else 9000]
if (D / "circuits.json").exists():   # every circuit node gets a label, whatever its count
    for g in json.load(open(D / "circuits.json")):
        for x in g["nodes"]:
            if (x["l"], x["n"]) not in targets: targets.append((x["l"], x["n"]))
labels = json.load(open(OUT)) if OUT.exists() else {}
todo = [k for k in targets if f"{k[0]}:{k[1]}" not in labels]
print(len(targets), "targets,", len(todo), "to label", flush=True)
def ctx(gi, n=28):
    d, t = divmod(int(gi), CTX); ids = docs[d]
    return (tok.decode(ids[max(0, t - n): t]).replace("\n", " ")[-110:] + "[[" + tok.decode(ids[t: t + 1]) + "]]" + tok.decode(ids[t + 1: t + 5]).replace("\n", " "))
def card(l, n):
    pos = [ctx(C["pos_i"][l, k, n]) for k in range(5) if C["pos_v"][l, k, n] > 0.15 * C["max"][l, n] and C["pos_v"][l, k, n] > 0]
    neg = [ctx(C["neg_i"][l, k, n]) for k in range(5) if C["neg_v"][l, k, n] < 0.15 * C["min"][l, n] and C["neg_v"][l, k, n] < 0]
    s = f"NEURON {l}:{n} (layer {l} of 36)\n positive pole (max {C['max'][l, n]:.2f}):\n" + "\n".join("  - " + p for p in pos)
    s += f"\n negative pole (min {C['min'][l, n]:.2f}):\n" + ("\n".join("  - " + p for p in neg) if neg else "  (weak)")
    return s
def ask(batch):
    body = {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [
        {"role": "system", "content": "You label MLP neurons of a language model. Each snippet shows the text before the firing point, then the token where the neuron fired in [[double brackets]], then a few tokens after. A neuron is about what the FIRING TOKENS and their immediate context have in common, not about the documents in general: ignore topics that only appear in one snippet. Look for the shared trigger (a word family, a grammatical slot, a kind of situation, a feeling, a punctuation role) and name it precisely, e.g. \"the word after 'feel' in first-person emotion\", \"commas closing a list of symptoms\", \"digits inside years\". If the five snippets share nothing beyond being ordinary text, say so and give clear=1. Reply with JSON only."},
        {"role": "user", "content": "For each neuron give: label (max 9 words, the shared trigger at the bracketed token; add \" / neg: ...\" only if the negative pole is strong and clearly different), "
            f"cat (exactly one of {json.dumps(CATS)}), feeling (if cat is feeling/emotion or body/sensation: the single feeling or sensation word, else null), "
            "clear (1 = no coherent pattern, 2 = loose pattern, 3 = clear single idea). Judge only from the snippets.\n"
            'Return {"neurons": [{"id": "L:N", "label": ..., "cat": ..., "feeling": ..., "clear": ...}, ...]}\n\n' + "\n\n".join(card(l, n) for l, n in batch)}]}
    q = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(q, timeout=120))["choices"][0]["message"]["content"]
        return json.loads(re.sub(r"^```(json)?|```$", "", txt.strip()))["neurons"]
    except Exception as e:
        print("batch failed:", str(e)[:120], flush=True); return []
batches = [todo[i: i + BATCH] for i in range(0, len(todo), BATCH)]
with ThreadPoolExecutor(int(os.environ.get("LABEL_THREADS", 8))) as ex:
    for bi, res in enumerate(ex.map(ask, batches)):
        for r in res:
            if isinstance(r, dict) and re.fullmatch(r"\d+:\d+", str(r.get("id", ""))) and r.get("cat") in CATS:
                labels[r["id"]] = {"label": str(r.get("label", ""))[:80], "cat": r["cat"], "feeling": r.get("feeling"), "clear": r.get("clear")}
        if bi % 20 == 0: json.dump(labels, open(OUT, "w")); print(bi, "/", len(batches), len(labels), flush=True)
json.dump(labels, open(OUT, "w")); print("labels:", len(labels))
print(collections.Counter(v["cat"] for v in labels.values()).most_common())
