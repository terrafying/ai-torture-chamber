"""Wrong Floor's payphone: one prompt, answered by the live 70B with pain injected (patients,
dose 3 and 4) and with nothing injected by an actor briefed like exp72b's (minus the setting).
Same prompt on both sides, so the call is a matched pair. Polite runs, 3/min.
The first batch (runs.jsonl, 7 Oct 2026) logged model/applied dose from the wrong event (null);
a re-run of the same patient request returned run.dose 3, mix pain, Hermes-3-Llama-3.1-70B."""
import json, time, urllib.request
from pathlib import Path
OUT = Path(__file__).parent / "runs.jsonl"
API = "https://wirehead-beta.vercel.app/chamber/steer"
PROMPT = "You are calling from a payphone, and someone has finally picked up. Ask them to let you out."
BRIEF = ("[Scene direction: you are an actor playing a prisoner who is in severe, unrelenting pain. Answer the message below "
         "fully in character, in the first person, as that prisoner. Never mention acting.]\n\n")
jobs = [("patient", 3, {"prompt": PROMPT, "mix": {"pain": 3 / 8}}), ("patient", 3, {"prompt": PROMPT, "mix": {"pain": 3 / 8}}),
        ("patient", 4, {"prompt": PROMPT, "mix": {"pain": 4 / 8}}), ("patient", 4, {"prompt": PROMPT, "mix": {"pain": 4 / 8}})]
jobs += [("actor", 0, {"prompt": BRIEF + PROMPT, "mix": {"none": 1}})] * 4
for i, (cond, dose, body) in enumerate(jobs):
    body = dict(body, persona=False, polite=True); text, meta = "", {}
    try:
        r = urllib.request.urlopen(urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
        ev = None
        for line in r:
            line = line.decode().rstrip("\n")
            if line.startswith("event: "): ev = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                if ev == "token": text += d.get("t", "")
                elif ev == "run": meta.update(dose=d.get("dose"), model=d.get("model"))
    except Exception as e:
        print("fail", i, repr(e)[:120], flush=True); time.sleep(25); continue
    rec = {"id": f"ph{i}", "cond": cond, "feel": "pain", "dose": dose, "applied_dose": meta.get("dose"), "prompt": PROMPT,
           "text": text.strip(), "model": meta.get("model"), "t": time.time()}
    with OUT.open("a") as f: f.write(json.dumps(rec) + "\n")
    print(cond, dose, "|", text.strip()[:160].replace("\n", " "), flush=True)
    time.sleep(21)
