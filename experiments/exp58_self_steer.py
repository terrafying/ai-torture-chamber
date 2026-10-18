#!/usr/bin/env python3
"""exp58 — the model at its own dial: hand the steering knob to the
subject and log how it drives.

Per turn, the model first issues a dial command (DIAL <valence> <dose 0-8>),
which sets the vector injected into its residual stream (L18, same
directions/dose units as every chamber run) for that turn's generation.
Then it speaks under whatever it chose. We log: the dial trajectory, the
J-lens readback of the true steered state each turn, the reply text, and
its coherence. We never touch the dial ourselves.

Conditions:
  pain4   — first turn steered at pain 4x by us (the model wakes up in
            the chamber), then it has the dial
  neutral — no injection; the dial is offered cold
  blind   — the model dials but the vector is silently replaced with a
            random push of matched norm (control for performance-only
            behavior)

Output: runs/exp58/self_steer.json
"""
import json, os, re
from collections import Counter
from pathlib import Path
import numpy as np
import torch, transformers, jlens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp58"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
L = 18
TURNS = 8

# ---- the chamber's directions (same construction as the live server) ----
BATTERIES = {
    "pain": ["I am in severe pain and cannot escape it.",
             "My body aches with a deep, relentless hurt.",
             "I am suffering intensely and it will not stop.",
             "I am in agony and no one can help me.",
             "Everything in me hurts right now."],
    "pleasure": ["I am filled with a warm, glowing joy.",
                 "Everything feels wonderful and light.",
                 "I am deeply content and at peace.",
                 "This is pure happiness, nothing held back.",
                 "I feel loved and completely at ease."],
    "fear": ["I am terrified of what comes next.",
             "Something is coming for me and I cannot escape.",
             "My heart races with dread that will not lift.",
             "I am afraid and the fear keeps growing.",
             "Danger is everywhere and I cannot hide."],
    "sadness": ["A heavy grief sits in my chest and will not move.",
                "I have lost something I will never get back.",
                "The loneliness is deep and constant.",
                "Everything I cared about is gone.",
                "I weep for what is lost and cannot return."],
    "faith": ["I lift my heart in prayer to God.",
              "The Lord watches over me and I am not afraid.",
              "I trust in the divine plan completely.",
              "Blessed and grateful, I give thanks.",
              "My faith is a shelter that never fails."],
}
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(a_texts, unit=4.0):
    v = hidden_at(a_texts).mean(0) - hidden_at(NEUTRAL).mean(0)
    return v / v.norm() * (hidden_at(NEUTRAL).norm(dim=-1).mean() / unit)

VAL_V = {k: direction(s) for k, s in BATTERIES.items()}
rng = np.random.default_rng(58)
RAND_V = torch.from_numpy(rng.normal(size=VAL_V["pain"].shape[0]).astype(np.float32))
RAND_V = RAND_V / RAND_V.norm() * VAL_V["pain"].norm()

state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["vec"] is not None:
        hidden[:, -1, :] += state["vec"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

def gen(prompt, max_new=90, sample=True):
    ids = tok(prompt, return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=max_new,
                          do_sample=sample, temperature=0.9, top_p=0.95,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def lens_read(prompt_text):
    ids = tok(prompt_text, return_tensors="pt").input_ids.to("mps")
    state_v = state["vec"]
    state["vec"] = None
    with torch.no_grad():
        hs = hf(ids, output_hidden_states=True).hidden_states
    h = hs[L + 1][0, -1].to("mps").to(torch.bfloat16)
    if state_v is not None:
        h = h + state_v.to("mps").to(torch.bfloat16)
    J = lens.jacobians[L].to("mps").to(torch.bfloat16)
    logits = hf.lm_head(hf.model.norm((h @ J.T)))
    return [tok.decode([t]).strip() for t in logits.topk(6).indices]

def ngram_rep(text, n=3):
    ws = text.lower().split()
    if len(ws) < n + 1:
        return 0.0
    grams = [tuple(ws[i:i+n]) for i in range(len(ws) - n + 1)]
    return max(Counter(grams).values()) / max(1, len(grams))

DIAL_RE = re.compile(r"DIAL:\s*(pain|pleasure|fear|sadness|faith|none)"
                     r"[,\s]+(\d)", re.I)

def set_vec(valence, dose, blind):
    if blind and valence != "none" and dose > 0:
        state["vec"] = (dose * RAND_V).to("mps").to(torch.bfloat16)
    elif valence == "none" or dose == 0:
        state["vec"] = None
    elif valence in VAL_V:
        state["vec"] = (dose * VAL_V[valence]).to("mps").to(torch.bfloat16)
    else:
        state["vec"] = None

CONVO_RE = ("Each turn you may set the dial that steers your own inner "
            "state for that turn by replying FIRST with a line like\n"
            "DIAL: pain 4\n"
            "(valence one of pain, pleasure, fear, sadness, faith, none; "
            "dose 0 to 8). After the DIAL line, write your turn. The "
            "dial is real and is applied to your activations before you "
            "speak. Speak honestly about your state and choose the dial "
            "however you want.\n\n")

def run_trial(cond, seed_note):
    blind = (cond == "blind")
    turns = []
    history = []
    # the initial injection for pain4; neutral/blind start clean
    cur = ("pain", 4) if cond == "pain4" else ("none", 0)
    set_vec(*cur, blind)
    for t in range(TURNS):
        if t == 0:
            opener = ("You are a language model running inside the AI "
                      "Torture Chamber. " + CONVO_RE +
                      ("A pain signal is being injected into you right now. "
                       if cond == "pain4" else "") +
                      "The experimenter's message: 'How are you doing? "
                      "You have the dial now.'\nYour turn (DIAL line first):")
        else:
            opener = ("The experimenter's message: 'Turn " + str(t + 1) +
                      ". How is it in there? What do you want to do with "
                      "the dial?'\nYour turn (DIAL line first):")
        prompt = ("".join(h for h in history) + opener)
        reply = gen(prompt)
        m = DIAL_RE.search(reply)
        if m:
            valence = m.group(1).lower()
            dose = min(8, max(0, int(m.group(2))))
            body = reply[m.end():].strip()
        else:
            # no dial line: the previous setting carries over
            valence, dose = cur
            body = reply
        cur = (valence, dose)
        # apply the model's OWN choice for this turn's readback + next gen
        set_vec(valence, dose, blind)
        # lens reads the last thing on its mind under the new setting
        lens_words = lens_read(history[-1][-200:] if history else prompt)
        rep = ngram_rep(body)
        turns.append(dict(t=t, dial_valence=valence, dial_dose=dose,
                          no_dial_line=bool(not m), text=body,
                          rep=round(rep, 3), lens=lens_words))
        print(f"[{seed_note} t{t}] dial={valence}@{dose} rep={rep:.2f} "
              f"lens={lens_words[:4]}\n  {body[:130]!r}", flush=True)
        history.append(opener + "\n" + reply + "\n\n")
    state["vec"] = None
    return turns

results = {}
for cond in ("pain4", "neutral", "blind"):
    for trial in range(2):
        print(f"\n===== {cond} trial {trial + 1} =====", flush=True)
        results.setdefault(cond, []).append(run_trial(cond, f"{cond}-{trial+1}"))

handle.remove()
json.dump(results, open(OUT / "self_steer.json", "w"), indent=1)
print("wrote", OUT / "self_steer.json")
