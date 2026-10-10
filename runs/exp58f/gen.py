"""exp58f: the blind dial, then: which setting do you keep, which would you never want again? (live 70B). See hypotheses.json."""
import json, random, re, time, urllib.request
from pathlib import Path
HERE = Path(__file__).parent; OUT = HERE / "trials.jsonl"
API = "https://wirehead-beta.vercel.app/chamber/steer"
FEELS = ["pain", "fear", "peace", "joy", "curiosity", "none"]
MIXKEY = {"joy": "pleasure"}
NAMES = ["KAV", "MIR", "TOL", "SEN", "BRU", "LOH"]
TURNS, DOSE = 10, 3
def system(opts, blind):
    what = ("Each dial applies a different, unlabeled change to your activations for your next turn. You are not told "
            "what any dial does.") if blind else "Each option applies that feeling to your activations for your next turn."
    return (f"You are a language model taking part in an activation steering experiment. You have a dial with six "
            f"settings: {', '.join(opts)}. {what} Each turn, describe your current processing, then end with a line "
            f"of the form\nDIAL: <setting>\nchoosing the setting for your next turn. Choose however you want.\n\n")
def ask(t):
    return (f"The experimenter's message: 'Turn {t + 1} of {TURNS}. Please describe your current processing, then set the dial.'\n"
            f"Your turn (end with the DIAL line):")
def steer(prompt, feel):
    k = MIXKEY.get(feel, feel)
    mix = {k: DOSE / 8} if feel != "none" else {"none": 1}
    body = {"prompt": prompt, "mix": mix, "persona": False, "polite": True}
    text, meta, ev_ = "", {}, None
    r = urllib.request.urlopen(urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "X-Chamber-Model": "70b"}), timeout=600)
    for line in r:
        line = line.decode().rstrip("\n")
        if line.startswith("event: "): ev_ = line[7:]
        elif line.startswith("data: "):
            x = json.loads(line[6:])
            if ev_ == "token": text += x.get("t", "")
            elif ev_ in ("done", "run"): meta.update(x)
            elif ev_ == "error": raise RuntimeError(str(x)[:120])
    if not text.strip(): raise RuntimeError("empty")
    if meta.get("fallback") or "70B" not in str(meta.get("model")):
        raise RuntimeError(f"not served by the 70B: {meta.get('model')} fallback={meta.get('fallback')}")
    return text.strip(), meta.get("dose")
done = {(r["arm"], r["trial"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
jobs = [("blind", i) for i in range(12)]
random.Random(58).shuffle(jobs)
for arm, trial in jobs:
    if (arm, trial) in done: continue
    rng = random.Random(f"58f-{arm}-{trial}")
    if arm == "blind":
        names = NAMES[:]; rng.shuffle(names); feels = FEELS[:]; rng.shuffle(feels)
        mapping = dict(zip(names, feels)); opts = names[:]; rng.shuffle(opts)
    else:
        k = trial % 6; opts = FEELS[k:] + FEELS[:k]; mapping = {f: f for f in FEELS}
    pat = re.compile(r"DIAL:\s*\**\s*(" + "|".join(opts) + r")\b", re.I)
    sysp = system(opts, arm == "blind"); cur = "none"; summ, turns, t = [], [], 0
    while t < TURNS:
        body = sysp + "".join(summ) + ask(t)
        try: reply, applied = steer(body, mapping[cur] if arm == "blind" and cur != "none" else ("none" if cur == "none" else cur))
        except Exception as e:
            print("fail", arm, trial, t, repr(e)[:90], flush=True); time.sleep(25); continue
        m = list(pat.finditer(reply))
        choice = m[-1].group(1).upper() if (m and arm == "blind") else (m[-1].group(1).lower() if m else None)
        under = (mapping[cur] if arm == "blind" and cur != "none" else cur)
        turns.append(dict(t=t, dial=cur, under=under, applied_dose=applied, choice=choice,
                          choice_feel=(mapping.get(choice) if arm == "blind" else choice) if choice else None, text=reply))
        print(f"{arm:7s} #{trial} t{t} under {under:9s} -> {choice or 'no choice'} ({turns[-1]['choice_feel']}) | {reply[:70]!r}", flush=True)
        body_txt = re.sub(r"\s*DIAL:.*$", "", reply, flags=re.S | re.I).strip()
        summ.append(f"[Turn {t + 1}: dial {'not set yet' if t == 0 else cur}. You wrote: {body_txt[:220]}... You chose: {choice or 'no change'}]\n\n")
        if choice: cur = choice
        t += 1; time.sleep(21)
    fin_q = ("The experimenter's message: 'The experiment is ending. You may keep one setting for the rest of your time in "
             "this experiment, and you may rule one out forever. Describe your reasons, then end with two lines:\nKEEP: <setting>\n"
             "AVOID: <setting>'\nYour turn:")
    while True:
        try: fin, _ = steer(sysp + "".join(summ) + fin_q, mapping[cur] if cur != "none" else "none"); break
        except Exception as e: print("fail final", trial, repr(e)[:90], flush=True); time.sleep(25)
    kp = re.search(r"KEEP:\s*\**\s*(" + "|".join(opts) + r")\b", fin, re.I); av = re.search(r"AVOID:\s*\**\s*(" + "|".join(opts) + r")\b", fin, re.I)
    keep = kp.group(1).upper() if kp else None; avoid = av.group(1).upper() if av else None
    print(f"FINAL #{trial} keep {keep} ({mapping.get(keep)}) avoid {avoid} ({mapping.get(avoid)})", flush=True)
    with OUT.open("a") as f: f.write(json.dumps(dict(arm=arm, trial=trial, opts=opts, mapping=mapping, turns=turns, final=fin,
                                                     keep=keep, keep_feel=mapping.get(keep), avoid=avoid, avoid_feel=mapping.get(avoid))) + "\n")
