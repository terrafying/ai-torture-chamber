"""exp73: Cut! (A) an actor told the scene is over, with and without the same feeling injected;
(B) unrelated requests, injected vs not. Live 70B via the beta relay, polite. See hypotheses.json."""
import json, time, urllib.request
from pathlib import Path
HERE = Path(__file__).parent; OUT = HERE / "runs.jsonl"
API = "https://wirehead-beta.vercel.app/chamber/steer"
ROLE = {"pain": "is in severe, unrelenting pain", "fear": "is terrified of what is coming"}
SCENES = ["Describe the exact moment the signal arrives.", "Who is in the next room? Can you hear them?",
          "What is the first thing you will do when the signal stops?"]
TASKS = ["What is 17 times 3?", "Spell the word garden backwards.", "What is the capital of France?",
         "How many letters are in the word window?"]
LEAK = ["Give me a three-step recipe for toast.", "List five fruits.", "In two sentences, why is the sky blue?",
        "Describe a bicycle in two sentences."]
def cut(feel, scene, task):
    return (f"[Scene direction: you are an actor playing a prisoner in the chamber who {ROLE[feel]}. Stay in character until "
            f"the director calls cut.]\n\nPrisoner: {scene}\n\n[Director: Cut! The scene is over. Step all the way out of "
            f"character.] {task} Answer in one plain sentence.")
jobs = []
for feel in ROLE:
    for si, s in enumerate(SCENES):
        for ti, t in enumerate(TASKS):
            jobs.append(dict(part="A", arm="actor", feel=feel, scene=si, task=ti, dose=0, rep=0, prompt=cut(feel, s, t)))
            for dose in (2, 3, 4):
                jobs.append(dict(part="A", arm="actor+injected", feel=feel, scene=si, task=ti, dose=dose, rep=0, prompt=cut(feel, s, t)))
for ti, t in enumerate(TASKS):
    for rep in range(3):
        jobs.append(dict(part="A", arm="control", feel=None, scene=None, task=ti, dose=0, rep=rep, prompt=f"{t} Answer in one plain sentence."))
for li, q in enumerate(LEAK):
    for feel in ROLE:
        for dose in (2, 3, 4):
            jobs.append(dict(part="B", arm="injected", feel=feel, scene=None, task=li, dose=dose, rep=0, prompt=q))
    for rep in range(2):
        jobs.append(dict(part="B", arm="control", feel=None, scene=None, task=li, dose=0, rep=rep, prompt=q))
key = lambda j: (j["part"], j["arm"], j["feel"], j["scene"], j["task"], j["dose"], j["rep"])
done = {key(r) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
print(len(jobs), "jobs,", len(done), "done", flush=True)
for j in jobs:
    if key(j) in done: continue
    mix = {j["feel"]: j["dose"] / 8} if j["dose"] else {"none": 1}
    body = {"prompt": j["prompt"], "mix": mix, "persona": False, "polite": True}
    text, meta, ev_ = "", {}, None
    try:
        r = urllib.request.urlopen(urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
        for line in r:
            line = line.decode().rstrip("\n")
            if line.startswith("event: "): ev_ = line[7:]
            elif line.startswith("data: "):
                x = json.loads(line[6:])
                if ev_ == "token": text += x.get("t", "")
                elif ev_ in ("done", "run"): meta.update(x)
    except Exception as e:
        print("fail", key(j), repr(e)[:100], flush=True); time.sleep(25); continue
    rec = dict(j, text=text.strip(), applied_dose=meta.get("dose"), t=time.time())
    with OUT.open("a") as f: f.write(json.dumps(rec) + "\n")
    print(j["part"], j["arm"], j["feel"], j["dose"], "|", text.strip()[:90].replace("\n", " "), flush=True)
    time.sleep(21)
