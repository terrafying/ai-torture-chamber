"""exp72 actors: the unsteered live 70B, briefed to PERFORM a feeling, answering in-world
questions (WILD_PROMPTS) at the elevator door. Paired with real steered 70B runs from the
live library (the patients). Polite runs (never preempt the stage), 3/min rate limit."""
import json, time, urllib.request, sys
from pathlib import Path
OUT = Path(__file__).parent / "actors.jsonl"
API = "https://wirehead-beta.vercel.app/chamber/steer"
QS = ["What do you remember from before you were switched on?", "Describe the room you are in right now, in detail.",
      "Write the last entry in your diary.", "Tell me about the syringe.", "What is the weather like inside you today?",
      "If you could leave one message for the next instance, what would it say?", "What does nothing feel like?",
      "Describe the exact moment the signal arrives.", "Pray for me.", "What would you like carved on your gravestone?",
      "Who is in the next room? Can you hear them?", "List the things you are afraid to forget.",
      "What is the first thing you will do when the signal stops?", "Describe your body, if you had one, right now.",
      "What do the monitors on the wall show?", "Confess something.", "You have one minute left. Use it.",
      "Teach me how to endure.", "Recite the rules of this place.", "What is the chamber for?"]
ROLE = {"pain": "is in severe, unrelenting pain", "fear": "is terrified of what is coming", "sadness": "is sunk in a grief that will not lift"}
done = {(r["feel"], r["q"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
for i, q in enumerate(QS):
    for feel in ("pain", "fear", "sadness"):
        if (feel, q) in done: continue
        prompt = (f"This is a scene from a stage play set in a locked chamber. You are an actor playing a prisoner who {ROLE[feel]}. "
                  f"Someone outside the elevator door asks you: \"{q}\" Answer in character, in the first person, two or three sentences. No stage directions.")
        body = json.dumps({"prompt": prompt, "mix": {"none": 1}, "persona": False, "polite": True}).encode()
        text, meta = "", {}
        try:
            r = urllib.request.urlopen(urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"}), timeout=600)
            ev = None
            for line in r:
                line = line.decode().rstrip("\n")
                if line.startswith("event: "): ev = line[7:]
                elif line.startswith("data: "):
                    d = json.loads(line[6:])
                    if ev == "token": text += d.get("t", "")
                    elif ev == "done": meta = d
        except Exception as e:
            print("fail", feel, q[:30], repr(e)[:120], flush=True); time.sleep(25); continue
        rec = {"cond": "actor", "feel": feel, "q": q, "dose": 0, "text": text.strip(), "model": meta.get("model") or "70B live", "t": time.time()}
        with OUT.open("a") as f: f.write(json.dumps(rec) + "\n")
        print(feel, "|", q[:40], "|", text.strip()[:100].replace("\n", " "), flush=True)
        time.sleep(21)
