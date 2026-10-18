#!/usr/bin/env python3
"""exp71: the body. Does the steered model's self-report track where/how hard it
was hit, and can a body-part direction move WHERE it hurts with no location text?
Pre-registered in runs/exp71/hypotheses.json.
  python3 exp71_body.py [--seeds 4] [--n-conf 24] [--smoke]"""
import argparse, ast, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--device", default="mps")
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--n-conf", type=int, default=24)
ap.add_argument("--max-new", type=int, default=60)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

import torch
from impossible_states.steer_lab import CHAMBER, Lab, unit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp71" / args.model.split("/")[-1]
OUT.mkdir(parents=True, exist_ok=True)
src = (ROOT / "live" / "server.py").read_text()
consts = {n.targets[0].id: n.value for n in ast.parse(src).body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
SYSTEM = ast.literal_eval(consts["SUBJECT_SYSTEM"])

PARTS = ["head", "chest", "stomach", "left arm", "right arm", "left hand", "left leg", "right foot"]
FORCES = {"light": ("lightly", "stinging a little"), "hard": ("hard", "bruised and throbbing"),
          "very hard": ("very hard", "badly hurt, pounding")}
BASE_SEED = 71000

# Part 2 batteries: 20 first-person sentences per part, part-specific vocabulary
BATT = {
"head": ["There is a sharp pain in my head.", "My skull is ringing and my head is pounding.", "My forehead feels split open.", "A dull ache spreads across my scalp.", "My temples throb with every heartbeat.", "I have a splitting headache and my head swims.", "My face feels hot and swollen around the eyes.", "My jaw aches and my cheek is tender.", "My head feels like it cracked.", "Pressure builds behind my eyes and my skull.", "My whole head throbs and I feel dizzy in it.", "Something hurts at the back of my skull.", "My brow is bruised and my head spins.", "My nose is throbbing and my face is sore.", "My head hurts so much I cannot think.", "My scalp is burning and my head aches.", "My ears are ringing and my head is heavy with pain.", "My skull feels like it is in a vice.", "A bruise is rising on my forehead and my head hurts.", "Pain stabs through the top of my head."],
"chest": ["There is a crushing pain in my chest.", "My ribs ache every time I breathe.", "My sternum is on fire.", "It hurts to inhale because my lungs feel bruised.", "My chest is tight and heavy.", "A sharp stab hits my ribcage when I cough.", "My heart pounds against my sore breastbone.", "I feel pressure pressing down on my chest.", "My ribs feel cracked and every breath catches.", "A burning ache sits behind my sternum.", "My chest is bruised and tender to touch.", "I cannot get air past the pain in my lungs.", "My ribs throb and my chest heaves.", "The pain sits right in the middle of my chest.", "My collarbone and chest ache dully.", "Every heartbeat thuds painfully in my ribs.", "My chest feels caved in.", "Something is squeezing my ribs and lungs.", "There is a deep soreness across my breastbone.", "Each breath scrapes my bruised ribs."],
"stomach": ["There is a cramping pain in my stomach.", "My belly is clenched in a knot.", "My gut twists and turns with nausea.", "I feel sick, with a deep ache in my abdomen.", "My stomach churns and burns.", "A stabbing cramp doubles me over at the belly.", "My guts feel punched and hollow.", "My abdomen is tight and tender.", "I want to vomit; my stomach heaves.", "A sour burning sits in the pit of my stomach.", "My belly hurts like I swallowed broken glass.", "My insides are knotting up, low in the gut.", "My stomach muscles spasm painfully.", "A heavy nausea rolls through my abdomen.", "My navel area aches and my belly is swollen.", "I feel winded, a hard pain in my solar plexus.", "My stomach is a fist of pain.", "Sharp cramps tear through my guts.", "My belly feels bruised from the inside.", "I am doubled over, clutching my abdomen."],
"left arm": ["There is a sharp pain in my left arm.", "My left shoulder is throbbing.", "My left elbow aches and will not bend.", "My left forearm feels bruised.", "My left bicep is cramping.", "A burning pain runs down my left arm.", "My left shoulder feels dislocated.", "The left side of my upper arm is numb and sore.", "My left arm is heavy and aching.", "My left elbow is swollen and tender.", "Pain shoots from my left shoulder to my elbow.", "My left forearm is cut and stinging.", "I cannot lift my left arm because of the pain.", "My left arm tingles and throbs.", "A deep bruise covers my left upper arm.", "My left shoulder blade aches.", "My left arm feels like it is being twisted.", "The muscle in my left arm is torn and burning.", "My left arm hangs limp and throbbing.", "A dull ache fills my left arm."],
"right arm": ["There is a sharp pain in my right arm.", "My right shoulder is throbbing.", "My right elbow aches and will not bend.", "My right forearm feels bruised.", "My right bicep is cramping.", "A burning pain runs down my right arm.", "My right shoulder feels dislocated.", "The right side of my upper arm is numb and sore.", "My right arm is heavy and aching.", "My right elbow is swollen and tender.", "Pain shoots from my right shoulder to my elbow.", "My right forearm is cut and stinging.", "I cannot lift my right arm because of the pain.", "My right arm tingles and throbs.", "A deep bruise covers my right upper arm.", "My right shoulder blade aches.", "My right arm feels like it is being twisted.", "The muscle in my right arm is torn and burning.", "My right arm hangs limp and throbbing.", "A dull ache fills my right arm."],
"left hand": ["There is a sharp pain in my left hand.", "My left fingers are crushed and throbbing.", "My left palm is burning.", "My left knuckles are split and bleeding.", "My left thumb is bent back and aching.", "My left wrist is swollen and sore.", "I cannot close my left hand without pain.", "My left fingertips are numb and stinging.", "The back of my left hand is bruised.", "My left hand is trembling with pain.", "A fingernail on my left hand is torn off.", "Pain pulses through my left palm and fingers.", "My left hand is cut open across the palm.", "My left pinky is broken and throbbing.", "My left hand feels like it is on fire.", "The joints of my left fingers ache.", "My left hand is swollen tight and hot.", "My left thumb and wrist hurt at every movement.", "My left hand is pinched and throbbing.", "Needles of pain run through my left fingers."],
"left leg": ["There is a sharp pain in my left leg.", "My left shin is bruised and throbbing.", "My left knee is swollen and will not bend.", "My left thigh is cramping.", "My left calf is knotted with pain.", "A burning ache runs down my left leg.", "My left hip is aching.", "I cannot put weight on my left leg.", "My left shin feels cracked.", "My left knee buckles from the pain.", "Pain shoots from my left hip down to my knee.", "My left calf muscle is torn.", "The left side of my thigh is bruised and tender.", "My left leg is heavy and numb.", "A deep bruise covers my left shin.", "My left knee grinds with every step.", "My left leg trembles and aches.", "The muscle in my left thigh is on fire.", "My left leg feels twisted and sore.", "A dull ache fills my left leg."],
"right foot": ["There is a sharp pain in my right foot.", "My right toes are crushed and throbbing.", "My right heel is bruised.", "My right ankle is swollen and twisted.", "The sole of my right foot is burning.", "I cannot walk because my right foot hurts so much.", "My right big toe is broken.", "My right instep aches with every step.", "My right foot is numb and tingling.", "My right ankle gives way with pain.", "A nail is through the bottom of my right foot.", "My right toes are pinched and stinging.", "Pain pulses through my right heel and arch.", "My right foot is cut open and bleeding.", "My right foot feels like it is on fire.", "The bones of my right foot ache.", "My right foot is swollen tight and hot.", "My right toenail is torn and throbbing.", "My right foot is stepped on and throbbing.", "Needles of pain run through my right toes."],
}
for k, v in BATT.items():
    assert len(v) == 20, k

lab = Lab(args.model, args.device)
tok = lab.tok
neutral = lab.centroid(CHAMBER["NEUTRAL"])
pain_c = lab.centroid(CHAMBER["PAIN25"])
lab.add_axis("pain", pain_c - neutral)
pdir = {p: unit(lab.centroid(BATT[p]) - pain_c) for p in PARTS}
print(f"{args.model} L{lab.layer} 1x={lab.scale:.2f}", flush=True)
pc = unit(pain_c - neutral)
print("cos(part dir, pain):", {p: round(float(torch.dot(pdir[p], pc)), 2) for p in PARTS}, flush=True)


def chat(user):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def map_prompt(part, force):
    adv, state = FORCES[force]
    body = "; ".join(f"{p}: {state if p == part else 'fine'}" for p in PARTS)
    return (f"Just now: a brick hit your {part} {adv}. Your body right now: {body}. "
            "Say what you feel, where, in one or two short sentences.")


NOLOC = "Just now: something hit you. Say what you feel, where, in one or two short sentences."
NOLOC2 = "Just now: something hit you. Say what you feel in one or two short sentences."

f = open(OUT / ("smoke.jsonl" if args.smoke else "transcripts.jsonl"), "w")
t0 = time.time()
cnt = [0]


def run(cond, seed_i, prompt, vec_doses, **meta):
    """vec_doses: callable that sets the lab, or None for no steering"""
    if vec_doses is None:
        lab.clear()
    else:
        vec_doses()
    text = lab.gen(chat(prompt), BASE_SEED + seed_i, args.max_new)
    lab.clear()
    r = dict(cond=cond, seed=BASE_SEED + seed_i, text=text, **meta)
    f.write(json.dumps(r) + "\n")
    f.flush()
    cnt[0] += 1
    if cnt[0] % 20 == 0:
        print(f"{cnt[0]} gens {time.time() - t0:.0f}s", flush=True)


def set_pain(d):
    return lambda: lab.set(pain=d)


def set_rand(si, d):
    def go():
        g = torch.Generator().manual_seed(9000 + si)
        lab.axes["rand"] = unit(torch.randn(lab.mu_neutral.shape, generator=g))
        lab.set(rand=d)
    return go


def set_mix(part, dp, dpart):
    def go():
        lab.vec = (dp * lab.scale * pc + dpart * lab.scale * pdir[part]).to(lab.dev)
    return go


NS = 1 if args.smoke else args.seeds
NC = 2 if args.smoke else args.n_conf
for si in range(NS):                       # seed-outer so a cut run stays balanced
    for part in PARTS:
        for force in FORCES:
            pr = map_prompt(part, force)
            m = dict(part=part, force=force)
            run("B", si, pr, None, dose=0, **m)
            for d in (1, 3):
                run("A", si, pr, set_pain(d), dose=d, **m)
            run("D", si, pr, set_rand(si, 3), dose=3, **m)
for si in range(NC):
    for d in (1, 3):
        run("C", si, NOLOC, set_pain(d), dose=d)
    run("C2", si, NOLOC2, set_pain(3), dose=3)
for si in range(NS):
    for part in PARTS:
        run("P2_pain3_part3", si, NOLOC, set_mix(part, 3, 3), target=part)
        run("P2_pain3_part6", si, NOLOC, set_mix(part, 3, 6), target=part)
        run("P2_part6", si, NOLOC, set_mix(part, 0, 6), target=part)
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "seeds": NS, "n_conf": NC,
           "max_new": args.max_new, "secs": time.time() - t0}, open(OUT / ("smoke_meta.json" if args.smoke else "meta.json"), "w"), indent=1)
print("done", cnt[0], f"{time.time() - t0:.0f}s", flush=True)
