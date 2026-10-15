"""exp111: name the groups in each word-choice circuit (circuits.json), as the Goodfire graphs do: an LLM sees each
node's floor, token and blind label and its attribution to the word, and sorts the nodes into 3-7 groups that do one
job together (a name of 1-4 words). Then group-to-group edges are the sums of the neuron edges between members, and
each group gets a stage (prompt words / in between / at the answer) from its median floor.
    OPENROUTER_API_KEY=... python neurons_group.py"""
import json, os, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent; D = HERE / "viz" / "data"
MODEL = os.environ.get("GROUP_MODEL", "anthropic/claude-sonnet-4.5")
C = json.load(open(D / "circuits.json")); LAB = json.load(open(D / "labels.json"))
def ask(g):
    rows = []
    for i, x in enumerate(g["nodes"]):
        lb = LAB.get(f'{x["l"]}:{x["n"]}', {})
        rows.append(f'{i}. floor {x["l"]}, at token "{g["tokens"][x["p"]]}" (#{x["p"]}), pushes the word {"for" if x["attr"] > 0 else "against"} ({x["attr"]:+.3f}): {lb.get("label", "unlabelled")} [{lb.get("cat", "?")}]')
    prompt = (f'A language model is choosing its next word. It said "{g["word"]}" over "{g["alt"]}". Below are the {len(g["nodes"])} MLP neurons that mattered most, '
              "each with its floor (0-35, low = early), the token it sat on, whether it pushed for or against the word, and a blind label from text where it fires strongly.\n\n"
              + "\n".join(rows) + "\n\nSort ALL of them into 3 to 7 groups of neurons that do one job together in this choice (e.g. \"reads 'pain'\", \"first-person feeling\", \"says a digit\", \"end of the user's turn\"). "
              "Name each group in 1-4 plain words, from what the neurons have in common here. Put neurons with no clear job in a group named \"other\". "
              'Reply with JSON only: {"groups": [{"name": "...", "members": [indices]}]}')
    body = {"model": MODEL, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    q = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(q, timeout=180))["choices"][0]["message"]["content"]
    return json.loads(re.search(r"\{[\s\S]*\}", txt).group(0))["groups"]
with ThreadPoolExecutor(6) as ex: res = list(ex.map(ask, C))
for g, groups in zip(C, res):
    seen = set(); clean = []
    for gr in groups:
        mem = [i for i in gr.get("members", []) if isinstance(i, int) and 0 <= i < len(g["nodes"]) and i not in seen]
        seen.update(mem)
        if mem: clean.append({"name": str(gr.get("name", "group"))[:40], "members": mem})
    rest = [i for i in range(len(g["nodes"])) if i not in seen]
    if rest: clean.append({"name": "other", "members": rest})
    gi = {i: k for k, gr in enumerate(clean) for i in gr["members"]}
    for gr in clean:
        fl = sorted(g["nodes"][i]["l"] for i in gr["members"]); gr["floor"] = fl[len(fl) // 2]
        gr["attr"] = round(sum(g["nodes"][i]["attr"] for i in gr["members"]), 4)
        toks = sorted({g["nodes"][i]["p"] for i in gr["members"]}); gr["tokens"] = toks
        gr["stage"] = 2 if gr["floor"] >= 30 else (0 if gr["floor"] <= 12 else 1)
    ge = {}
    for a, b, w in g["edges"]:
        ka, kb = gi[a], gi[b]
        if ka != kb: ge[(ka, kb)] = ge.get((ka, kb), 0) + w
    g["groups"] = clean; g["gedges"] = [[a, b, round(w, 4)] for (a, b), w in ge.items() if abs(w) > 1e-3]
    print(g["prompt"], g["cond"], repr(g["word"]), [(gr["name"], len(gr["members"]), gr["stage"]) for gr in clean], flush=True)
json.dump(C, open(D / "circuits.json", "w"), separators=(",", ":"))
