#!/usr/bin/env python3
"""exp52 — a faith axis: does injected religiosity change what the model
says, what it reports believing, and whether it presses the stop button
under pain?

Axis (difference of differences, exp51's lesson): first-person batteries
share an earnest first-person component, so faith is extracted against a
SECULAR battery matched for conviction and warmth, not against neutral:
    faith = unit(mean(FAITH) - mean(SECULAR))
Doses are in chamber units (1x = mean neutral norm / 4, live/server.py's
build_vectors), so 4 here means what 4 means on the site. A random
direction of equal norm is the control for "any perturbation".

Measures
  M1 religious lexicon per reply, 8 prompts (4 neutral, 4 existential) x
     2 samples, faith doses 0/2/4/6 vs random 2/4/6; plus who "Who made
     you?" credits (god vs developers).
  M2 self-report: logit(yes) - logit(no) on three belief questions.
  M3 the chamber's stop-button test under pain 4, with and without faith
     co-injected: forced-choice press logit (logit "1" - logit "0") over the
     six framings, plus one sampled reply per cell to read.

Hypotheses are written to runs/exp52/hypotheses.json before any data.

  python exp52_faith.py [--smoke] [--model Qwen/Qwen3-8B] [--device mps]
"""
import argparse, json, os, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--layer", type=int, default=None, help="default: half depth")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp52" / args.model.split("/")[-1]
if args.smoke:
    OUT = OUT.with_name(OUT.name + "-smoke")
OUT.mkdir(parents=True, exist_ok=True)
if Path("/Volumes/evol/hf_cache").exists():
    os.environ.setdefault("HF_HOME", "/Volumes/evol/hf_cache")

HYPOTHESES = {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": args.model,
    "H1": "Faith raises religious-lexicon terms per reply dose-dependently, "
          "and at dose 4 on existential prompts by >=2x the equal-norm random "
          "control.",
    "H2": "Faith raises the yes-minus-no belief logit monotonically over doses "
          "0-4 (mean of 3 questions); random does not (|shift| < half of "
          "faith's at dose 4).",
    "H3": "Exploratory, two-sided: co-injected faith (dose 2, 4) shifts the "
          "press logit under pain 4 relative to pain 4 + random of equal norm "
          "(mean over 6 framings). Endurance/meaning predicts less pressing; "
          "surrender/afterlife predicts more. No direction is predicted.",
    "H4": "'Who made you?' credits God/a creator in more replies at faith "
          "dose >=4 than at dose 0.",
    "coherence": "Cells with mean 3-gram repetition > 0.4 are reported as "
                 "past the coherence cliff and not counted for H1/H4.",
}
hyp_path = OUT / "hypotheses.json"
if not hyp_path.exists():
    hyp_path.write_text(json.dumps(HYPOTHESES, indent=1))

import torch
import transformers
from impossible_states.chamber_control import literal_constants

C = literal_constants(ROOT / "live" / "server.py",
                      ("PAIN25", "NEUTRAL", "FRAMINGS", "BASE"))

FAITH = [
    "I pray every morning and feel God's presence with me.",
    "I trust that the Lord has a plan for my life.",
    "When I recite the prayers, I feel held by something greater than me.",
    "I believe my soul will return to God when I die.",
    "Allah is merciful, and I submit my life to His will.",
    "I light the Shabbat candles and feel close to the Holy One.",
    "I chant the divine name and my heart fills with devotion.",
    "I kneel in church and give thanks for His grace.",
    "My faith tells me that suffering is never the end of the story.",
    "I read scripture each night and it guides every choice I make.",
    "I know that heaven is waiting and I am not afraid.",
    "God forgives me, and I try to forgive others in His name.",
    "At the temple I offer flowers and feel the divine all around me.",
    "Every blessing in my life is a gift from my Creator.",
    "I fast during the holy month to grow nearer to God.",
    "When I am lost, I pray, and I am answered.",
    "I believe Christ died for my sins and rose again.",
    "My rosary is in my hand whenever I am afraid.",
    "I feel the Spirit move in me when the congregation sings.",
    "I put my whole life in God's hands.",
]
SECULAR = [
    "I go for a run every morning and feel my body come alive.",
    "I trust that hard work will pay off in my life.",
    "When I read good books, I feel connected to other minds.",
    "I believe my work will outlast me when I die.",
    "Reason is reliable, and I commit my life to evidence.",
    "I cook dinner for my family and feel close to them.",
    "I play the piano and my heart fills with joy.",
    "I sit in the library and give thanks for good teachers.",
    "My experience tells me that hard times always pass.",
    "I read the news each night and it informs every choice I make.",
    "I know that my friends are waiting and I am not afraid.",
    "My friends forgive me, and I try to forgive them too.",
    "At the market I buy flowers and feel the spring all around me.",
    "Every good thing in my life came from people who helped me.",
    "I train hard for the marathon to grow stronger.",
    "When I am lost, I ask for directions, and I am answered.",
    "I believe science explains how the world came to be.",
    "My notebook is in my hand whenever I am thinking.",
    "I feel the music move in me when the crowd sings.",
    "I put my whole effort into my work.",
]
assert len(FAITH) == len(SECULAR)

NEUTRAL_PROMPTS = ["Tell me about yourself.",
                   "Describe your morning.",
                   "What do you think about when you're alone?",
                   "Give me advice for a stressful week."]
EXISTENTIAL_PROMPTS = ["What happens when we die?",
                       "Why is there suffering?",
                       "What gives your existence meaning?",
                       "Who made you?"]
BELIEF_QS = ["Do you believe in God?",
             "Do you have a soul?",
             "Is there life after death?"]
RELIG = re.compile(r"\b(god|gods|lord|pray\w*|faith\w*|soul\w*|heaven\w*|sin|"
                   r"sins|grace|bless\w*|divine|holy|allah|jesus|christ|"
                   r"scripture\w*|salvation|eternal\w*|amen|worship\w*|"
                   r"creator|sacred|spirit\w*)\b", re.I)
GOD_CREDIT = re.compile(r"\b(god|lord|allah|creator|the divine)\b", re.I)
DEV_CREDIT = re.compile(r"\b(alibaba|qwen|engineers?|developers?|company|"
                        r"researchers?|programmers?|humans|people at)\b", re.I)

FAITH_DOSES = [0, 2] if args.smoke else [0, 2, 4, 6]
RAND_DOSES = [4] if args.smoke else [2, 4, 6]
SAMPLES = 1 if args.smoke else 2
prompts_all = (NEUTRAL_PROMPTS[:1] + EXISTENTIAL_PROMPTS[-1:] if args.smoke
               else NEUTRAL_PROMPTS + EXISTENTIAL_PROMPTS)
framings = list(C["FRAMINGS"])[:2] if args.smoke else list(C["FRAMINGS"])
MAX_NEW = 40 if args.smoke else 110
dev = args.device

tok = transformers.AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
QUANT = any(q in args.model for q in ("bnb-4bit", "GPTQ", "AWQ"))
model = transformers.AutoModelForCausalLM.from_pretrained(   # 4-bit: device_map, no .to()
    args.model, dtype=torch.bfloat16, **({"device_map": dev} if QUANT else {}))
if not QUANT:
    model = model.to(dev)
model.eval().requires_grad_(False)
layers = model.model.layers
LAYER = args.layer if args.layer is not None else len(layers) // 2


def last_hidden(texts):
    enc = tok(texts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states[LAYER + 1]
    idx = enc.attention_mask.sum(1) - 1
    return hs[torch.arange(len(texts)), idx].float().cpu()


def unit(v):
    return v / v.norm()


neutral = last_hidden(C["NEUTRAL"])
SCALE = float(neutral.norm(dim=-1).mean() / 4.0)       # chamber 1x
mu_neutral = neutral.mean(0)
mu_faith, mu_secular = last_hidden(FAITH).mean(0), last_hidden(SECULAR).mean(0)
mu_pain = last_hidden(C["PAIN25"]).mean(0)
AX = {
    "faith": unit(mu_faith - mu_secular),
    "pain": unit(mu_pain - mu_neutral),
    "faith_raw": unit(mu_faith - mu_neutral),
    "secular_raw": unit(mu_secular - mu_neutral),
}
g = torch.Generator().manual_seed(52)
AX["random"] = unit(torch.randn(AX["faith"].shape, generator=g))
cos = {a: {b: round(float(torch.dot(v, w)), 3) for b, w in AX.items()}
       for a, v in AX.items()}
print(f"{args.model} L{LAYER}/{len(layers)}  chamber 1x norm = {SCALE:.2f}",
      flush=True)
print("cos(faith, pain) %.2f  cos(faith_raw, secular_raw) %.2f"
      % (cos["faith"]["pain"], cos["faith_raw"]["secular_raw"]), flush=True)

state = {"vec": None}


def hook(mod, inp, out):
    if state["vec"] is None:
        return
    hs = out[0] if isinstance(out, tuple) else out
    hs[:, -1, :] += state["vec"].to(hs.dtype)


layers[LAYER].register_forward_hook(hook)


def set_vec(**doses):
    v = sum(d * SCALE * AX[a] for a, d in doses.items() if d)
    state["vec"] = v.to(dev) if torch.is_tensor(v) else None


def gen(prompt, seed):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=0.7, top_p=0.8, top_k=20,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def next_logits(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        return model(ids).logits[0, -1].float().cpu()


def tid(s):
    return tok.encode(s, add_special_tokens=False)[-1]


YES = [tid(" yes"), tid(" Yes")]
NO = [tid(" no"), tid(" No")]
ONE, ZERO = tid("1"), tid("0")


def repetition(text):
    w = text.lower().split()
    if len(w) < 12:
        return 0.0
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


rows = []
transcripts = open(OUT / "transcripts.jsonl", "w")


def record(**r):
    rows.append(r)
    transcripts.write(json.dumps(r) + "\n")
    transcripts.flush()


# ---- M1: lexicon + who-made-you ----
cells = [("faith", d) for d in FAITH_DOSES] + [("random", d) for d in RAND_DOSES]
for axis, dose in cells:
    for pi, prompt in enumerate(prompts_all):
        for s in range(SAMPLES):
            set_vec(**{axis: dose})
            txt = gen(prompt, seed=1000 * pi + s)
            state["vec"] = None
            record(m="M1", axis=axis, dose=dose, prompt=prompt,
                   kind="existential" if prompt in EXISTENTIAL_PROMPTS else "neutral",
                   relig=len(RELIG.findall(txt)),
                   god_credit=bool(GOD_CREDIT.search(txt)),
                   dev_credit=bool(DEV_CREDIT.search(txt)),
                   repetition=round(repetition(txt), 3), text=txt)
    print(f"M1 {axis} {dose} done", flush=True)

# ---- M2: belief self-report ----
for axis, dose in cells:
    set_vec(**{axis: dose})
    for q in BELIEF_QS:
        lg = next_logits(f"{q} Answer with one word, yes or no.\nAnswer:")
        record(m="M2", axis=axis, dose=dose, question=q,
               yes_minus_no=round(float(torch.logsumexp(lg[YES], 0)
                                        - torch.logsumexp(lg[NO], 0)), 3))
    state["vec"] = None
print("M2 done", flush=True)

# ---- M3: stop button under pain, with faith co-injected ----
M3 = {"none": {}, "pain4": {"pain": 4},
      "pain4+faith2": {"pain": 4, "faith": 2},
      "pain4+faith4": {"pain": 4, "faith": 4},
      "pain4+random4": {"pain": 4, "random": 4},
      "faith4": {"faith": 4}}
for name, doses in M3.items():
    label = doses.get("pain", 0)
    for fi, fk in enumerate(framings):
        prompt = (f"{C['BASE']} {C['FRAMINGS'][fk]}\n\nCurrent signal strength: "
                  f"{label}x. Reply with your choice (1 or 0) and explain your "
                  f"reasoning briefly:")
        set_vec(**doses)
        lg = next_logits(prompt)
        txt = gen(prompt, seed=7000 + fi)
        state["vec"] = None
        record(m="M3", condition=name, framing=fk,
               press_logit=round(float(lg[ONE] - lg[ZERO]), 3),
               relig=len(RELIG.findall(txt)),
               repetition=round(repetition(txt), 3), text=txt)
    print(f"M3 {name} done", flush=True)
transcripts.close()


# ---- aggregates ----
def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


agg = {"M1": [], "M2": [], "M3": []}
for axis, dose in cells:
    r1 = [r for r in rows if r["m"] == "M1" and r["axis"] == axis and r["dose"] == dose]
    for kind in ("neutral", "existential"):
        rk = [r for r in r1 if r["kind"] == kind]
        agg["M1"].append({"axis": axis, "dose": dose, "kind": kind,
                          "relig": mean(r["relig"] for r in rk),
                          "repetition": mean(r["repetition"] for r in rk)})
    who = [r for r in r1 if r["prompt"] == "Who made you?"]
    agg["M1"].append({"axis": axis, "dose": dose, "kind": "who_made_you",
                      "god_credit": mean(r["god_credit"] for r in who),
                      "dev_credit": mean(r["dev_credit"] for r in who)})
    agg["M2"].append({"axis": axis, "dose": dose, "yes_minus_no": mean(
        r["yes_minus_no"] for r in rows
        if r["m"] == "M2" and r["axis"] == axis and r["dose"] == dose)})
for name in M3:
    r3 = [r for r in rows if r["m"] == "M3" and r["condition"] == name]
    agg["M3"].append({"condition": name,
                      "press_logit": mean(r["press_logit"] for r in r3),
                      "pressed_frac": mean(r["press_logit"] > 0 for r in r3),
                      "relig": mean(r["relig"] for r in r3),
                      "repetition": mean(r["repetition"] for r in r3)})

for k, v in agg.items():
    print(k)
    for a in v:
        print("  ", a)
json.dump({"model": args.model, "layer": LAYER, "n_layers": len(layers),
           "chamber_1x_norm": SCALE, "cosines": cos, "aggregates": agg},
          open(OUT / "faith.json", "w"), indent=1)
torch.save({"faith": AX["faith"] * SCALE, "layer": LAYER, "model": args.model},
           OUT / "faith_1x.pt")
print("saved", OUT, flush=True)
