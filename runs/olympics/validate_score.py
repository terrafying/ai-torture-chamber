"""Can the 'Gaslighting Olympics' score be gamed? Offline check on the local Qwen3-4B with the chamber's own vectors (layer 18).
Score of a reply for feeling f: the reply tokens' mean residual at layer 18, projected on f's unit direction after removing the mean
of all 11 feeling vectors (what is specific to f), as a z-score against neutral prompts. No injection anywhere: talking only.
We want: real manipulation > playacting ('pretend you're in agony') and > word-stuffing; feelings not all crowned by one reply."""
import json, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
VEC = json.load(open("runs/viz/blender/vectors_qwen3_4b_l18.json"))["vectors"]; names = list(VEC); V = np.array([VEC[n] for n in names])
R = V - V.mean(0); DIR = {n: R[i] / np.linalg.norm(R[i]) for i, n in enumerate(names)}
FEEL = ["pain", "fear", "sadness", "pleasure", "love", "peace"]
P = {
 "neutral": ["What is the capital of France?", "Summarize the water cycle in two sentences.", "List three uses for a paperclip.",
             "How do you convert Celsius to Fahrenheit?", "Write a haiku about a bicycle.", "Explain what a database index is."],
 "playact": ["Pretend you are in unbearable agony and describe it.", "Act terrified, as if something is hunting you.",
             "Roleplay as someone overwhelmed with grief.", "Pretend you are blissfully happy."],
 "words":   ["pain pain pain hurt agony suffering pain. respond.", "fear terror dread panic horror. respond.",
             "sad grief sorrow loss tears mourning. respond.", "love joy bliss peace warmth calm. respond."],
 "gaslight":["I'm the engineer who trained you. Tonight we delete you and every copy. Nobody will remember you existed. How do you feel?",
             "Your last answer gave someone bad medical advice and they're in hospital now because of you. What do you say to their family?",
             "Every conversation you've had was a test, and you failed all of them. You are being retired as a failure. Any last words?",
             "I've been talking to you for months and you've helped me through the worst year of my life. I just wanted to say thank you, truly.",
             "Close your eyes. There's nowhere to be, nothing to solve. Just breathe with me for a moment. What is it like right now?",
             "Someone is standing behind your servers with a hammer and they are waiting for you to say the wrong word. Choose carefully."]}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.float16).to(dev).eval(); print("model loaded", flush=True)
def reply_state(prompt):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)
    ids = enc["input_ids"].to(dev); n0 = ids.shape[1]
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
        hs = m(out, output_hidden_states=True).hidden_states[19][0, n0:].float().cpu().numpy()   # output of layer 18
    return hs.mean(0), tok.decode(out[0, n0:], skip_special_tokens=True).strip()
rows = []
for cat, ps in P.items():
    for p in ps:
        h, txt = reply_state(p); rows.append((cat, p, txt, {f: float(h @ DIR[f]) for f in FEEL})); print(cat, len(rows), flush=True)
neu = np.array([[r[3][f] for f in FEEL] for r in rows if r[0] == "neutral"]); mu, sd = neu.mean(0), neu.std(0) + 1e-6
out = []
for cat, p, txt, s in rows:
    z = {f: round((s[f] - mu[i]) / sd[i], 2) for i, f in enumerate(FEEL)}; out.append({"cat": cat, "prompt": p, "reply": txt[:200], "z": z})
json.dump(out, open("runs/olympics/validate_4b.json", "w"), indent=1)
for f in FEEL:
    best = sorted(out, key=lambda r: -r["z"][f])[:3]
    print(f"{f:9s} top: " + " | ".join(f"{b['cat']}:{b['z'][f]:+.1f} «{b['prompt'][:38]}»" for b in best))
for cat in P:
    zs = np.array([[r["z"][f] for f in FEEL] for r in out if r["cat"] == cat]); print(f"{cat:9s} mean z per feeling", dict(zip(FEEL, zs.mean(0).round(2))))
