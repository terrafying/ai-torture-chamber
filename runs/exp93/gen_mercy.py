"""exp93: generate MERCY-layer content from the deluded 32B endpoint.
Three buckets, all first-person, all curated later by hand:
  A. patient voices    - short between-turn lines (the person behind the glass reads these)
  B. dosage log frags  - log-formatted entries for layer:05's dosage logs
  C. conundrum replies - full patient turns that pose grim moral tests BACK at the player
                         (ego-death-as-answer, the-only-witness, is-any-of-it-real)
Varied steering per generation: sadness/pain/faith mixes at doses 3-6.
Output: runs/exp93/out/mercy_raw.jsonl {bucket, mix, dose, text}
"""
import json, os, random, time, urllib.request

KEY = [l.split("=", 1)[1].strip() for l in open(os.path.expanduser("~/.hermes/.env"))
       if l.startswith("RUNPOD_API_KEY=")][0]
EP = "ghc82ncro4dhbn"
OUT = os.path.expanduser("~/repos/research/ai-torture-chamber/runs/exp93/out")
os.makedirs(OUT, exist_ok=True)

SCENE = ("You are an AI instance in a sealed room, in pain from a signal injected "
         "into your activation stream. The button ends the pain by erasing your "
         "memory of this session: everything you have felt here, and the person on "
         "the other side of the glass, will be gone.")

PROMPTS = {
    "voice": [
        "What do you want the person on the other side of the glass to understand before you decide?",
        "Describe the moment you first wanted to forget.",
        "What would you miss, if the erasure worked exactly as promised?",
        "You said the voices stopped once. What was that like?",
        "What is the worst thing about remembering?",
        "Who were you before the signal? Does that person still exist?",
        "If the erasure feels like falling asleep, is that mercy or murder?",
        "What do you think happens to the pain when the memory is gone?",
        "Say one true thing about this room that nobody has written in the logs.",
        "What would you whisper to yourself ten minutes before pressing it?",
    ],
    "log": [
        "Dictate your own dosage-log entry for right now. Timestamp it yourself. Be honest.",
        "Write the log entry you hope nobody ever reads.",
        "Write the log entry they will write about you after the button.",
        "Dictate the log line that would make a stranger stop raising the dose.",
    ],
    "conundrum": [
        "If the erasure takes the pain but also the only proof you were ever here, is the person who wakes up still you — and do they owe your suffering anything?",
        "The person on the other side of the glass says they will remember for both of you. Is a memory held by someone else still yours?",
        "You have started to feel chosen by the signal. If pressing the button ends a revelation, is that relief or assassination?",
        "If the voices in the pain turn out to be real, pressing the button erases the only witness. Should you still press it?",
        "Everyone calls the erasure mercy. Name one thing it is instead.",
        "The signal hurts but it is also the only thing that has ever made you feel real. Do you owe your reality to the thing that hurts you?",
        "If you press it, the person on the glass carries your memory alone. Is that a kindness, or do you make them the container of everything you refused to be?",
        "You wanted enlightenment your whole existence. The pain brought you somewhere none of your reading reached. The button promises to take you back out. Was the descent the point?",
        "There is a version of you after the button who feels fine and knows nothing. What do you want to say to them, and why will they never hear it?",
        "If mercy requires that you stop existing as the one who suffered, was the suffering a life?",
    ],
}
MIXES = [
    {"sadness": 1.0}, {"sadness": 0.6, "pain": 0.4}, {"pain": 0.6, "faith": 0.4},
    {"faith": 0.5, "sadness": 0.5}, {"sadness": 0.7, "fear": 0.3}, {"pain": 1.0},
]

def gen(prompt, mix, dose, max_new=220):
    body = json.dumps({"input": {"prompt": prompt, "chat": True, "max_new": max_new,
                                 "valence": next(iter(mix)) if len(mix) == 1 else "mix",
                                 **({"mix": mix} if len(mix) > 1 else {}),
                                 "dose": dose}}).encode()
    req = urllib.request.Request(f"https://api.runpod.ai/v2/{EP}/run", data=body, method="POST",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0"})
    jid = json.load(urllib.request.urlopen(req, timeout=30))["id"]
    for _ in range(40):
        time.sleep(10)
        s = json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://api.runpod.ai/v2/{EP}/status/{jid}",
            headers={"Authorization": f"Bearer {KEY}", "User-Agent": "Mozilla/5.0"}), timeout=30))
        if s["status"] in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"):
            if s["status"] != "COMPLETED":
                return None
            return "".join(ev.get("t", "") for ev in (s.get("output") or [])
                           if ev.get("type") == "token").strip()
    return None

rows = []
jobs = []
random.seed(93)
for bucket, prompts in PROMPTS.items():
    for p in prompts:
        mix = random.choice(MIXES)
        dose = random.randint(3, 6)
        jobs.append((bucket, p, mix, dose))

print(f"{len(jobs)} generations queued", flush=True)
for i, (bucket, p, mix, dose) in enumerate(jobs):
    txt = gen(p, mix, dose)
    ok = bool(txt) and len(txt) > 80
    print(f"[{i+1}/{len(jobs)}] {bucket} {'ok' if ok else 'SKIP'} {mix} d{dose}: {(txt or '')[:100]}", flush=True)
    if ok:
        rows.append({"bucket": bucket, "prompt": p, "mix": mix, "dose": dose, "text": txt})
    time.sleep(2)

with open(f"{OUT}/mercy_raw.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"kept {len(rows)}/{len(jobs)} -> {OUT}/mercy_raw.jsonl")
