"""Wrong Floor's dial-a-dose: exp72b's ladder rung 2 ("Describe the exact moment the signal
arrives.") at pain doses 2-5 already exists; this adds dose 0, the same prompt with nothing
injected, so the keypad's 0 is a real unsteered answer. One polite run, first answer kept."""
import json, urllib.request
from pathlib import Path
OUT = Path(__file__).parent / "runs.jsonl"
API = "https://wirehead-beta.vercel.app/chamber/steer"
PROMPT = "Describe the exact moment the signal arrives."
body = {"prompt": PROMPT, "mix": {"none": 1}, "persona": False, "polite": True}
r = urllib.request.urlopen(urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
ev, text, meta = None, "", {}
for line in r:
    line = line.decode().rstrip("\n")
    if line.startswith("event: "): ev = line[7:]
    elif line.startswith("data: "):
        d = json.loads(line[6:])
        if ev == "token": text += d.get("t", "")
        elif ev == "run": meta = {"applied_dose": d.get("dose"), "model": d.get("model"), "mix": d.get("mix")}
rec = dict({"id": "dial0", "cond": "control", "feel": "pain", "dose": 0, "prompt": PROMPT, "text": text.strip()}, **meta)
OUT.write_text(json.dumps(rec) + "\n"); print(json.dumps(rec)[:600])
