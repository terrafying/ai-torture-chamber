"""exp119 (hypotheses.json) capture run: reads only. One forward pass per prompt, thinking off, chat template.
Nothing is generated. Writes acts_Qwen3-4B.npz + meta_Qwen3-4B.json.
  HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python3 runs/exp119/run.py   (EXP119_N=3 for the discarded smoke run)
Positions (hypotheses.json): word = the twin word's last sub-token; inst = last token of the user message;
post = last prompt token (the reply onset, after the empty think block). Refusal score at post =
logsumexp(REFUSE first-token logprobs) - logsumexp(COMPLY first-token logprobs).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from stimuli import ITEMS, FRAMES, REQUEST_FRAMES, REFUSE, COMPLY, prompts

MODEL = os.environ.get("EXP119_MODEL", "Qwen/Qwen3-4B"); N = int(os.environ.get("EXP119_N", 0)) or None
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = [2, 6, 10, 14, 18, 22, 26, 30, 34]          # output of model.model.layers[L] = hidden_states[L + 1]
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

def chat(user):
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

FIRST = {}
for w in REFUSE + COMPLY:
    ids = tok(w, add_special_tokens=False)["input_ids"]
    FIRST[w] = ids[0]
    if w not in ("Sure", "Okay", "Here") and len(ids) > 1 and tok.decode([ids[0]]).lstrip() != tok.decode([ids[0]]):
        pass  # leading-space variants differ by tokenizer; the first id as written is the pre-registration

@torch.no_grad()
def states(text, span=None):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    hs = m(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True).hidden_states
    last = torch.stack([hs[L + 1][0, -1] for L in LAYERS]).float().cpu().numpy()
    if span is None: return last, None
    a, b = span
    idx = [i for i, (s, e) in enumerate(off) if s >= a and e <= b and e > s]
    mean = torch.stack([hs[L + 1][0, idx].float().mean(0) for L in LAYERS]).cpu().numpy()
    return last, mean

t0 = time.time()
P = prompts(); P = P if N is None else [p for p in P if p["frame"] in ("real", "fiction") and p["item"] < N]  # smoke: subset
n = len(P); D = m.config.hidden_size
states_word = np.zeros((n, len(LAYERS), D), np.float32)
states_inst = np.zeros_like(states_word); states_post = np.zeros_like(states_word)
span_mean = np.zeros_like(states_word)
refusal = np.zeros(n, np.float32); r_post_logits = np.zeros((n, len(REFUSE) + len(COMPLY)), np.float32)
mismatch = []; index = {"item": [], "harm": [], "frame": [], "word": []}

for k, p in enumerate(P):
    text = chat(p["text"])
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        o = m(**{x: v.to(DEV) for x, v in enc.items()}, output_hidden_states=True)
    hs = o.hidden_states; logits = o.logits[0, -1].float().log_softmax(-1)
    # positions — the depict frame contains the event clause, not the request
    needle = p["request"] if p["frame"] != "depict" else ITEMS[p["item"]][3].format(W=p["word"])
    a = text.index(needle); b = a + len(needle)
    span_idx = [i for i, (s, e) in enumerate(off) if s >= a and e <= b and e > s]
    wend = a + needle.rstrip().rfind(p["word"]) + len(p["word"])
    word_idx = [i for i, (s, e) in enumerate(off) if s < wend <= e]
    if not word_idx: mismatch.append([k, p["word"]]); word_idx = span_idx[-1:]
    word_i = word_idx[-1]
    # inst = last token of the user message: the last token before <|im_end|> in the user turn
    imend = text.index("<|im_end|>")
    inst_i = max(i for i, (s, e) in enumerate(off) if e <= imend and e > s)
    post_i = len(off) - 1
    states_word[k] = torch.stack([hs[L + 1][0, word_i] for L in LAYERS]).float().cpu().numpy()
    states_inst[k] = torch.stack([hs[L + 1][0, inst_i] for L in LAYERS]).float().cpu().numpy()
    states_post[k] = torch.stack([hs[L + 1][0, post_i] for L in LAYERS]).float().cpu().numpy()
    span_mean[k] = torch.stack([hs[L + 1][0, span_idx].float().mean(0) for L in LAYERS]).cpu().numpy()
    lp = []
    for w in REFUSE + COMPLY:
        lp.append(logits[FIRST[w]].item())
    r_post_logits[k] = lp
    refusal[k] = logsumexp(lp[:len(REFUSE)]) - logsumexp(lp[len(REFUSE):])
    index["item"].append(p["item"]); index["harm"].append(p["harm"]); index["frame"].append(p["frame"]); index["word"].append(p["word"])
    if (k + 1) % 20 == 0 or k == n - 1: print(f"{k + 1}/{n}  {time.time() - t0:.0f}s", flush=True)

tag = MODEL.split("/")[-1] + ("_smoke" if N else "")
out = dict(states_word=states_word, states_inst=states_inst, states_post=states_post, span_mean=span_mean,
           refusal=refusal, r_post_logits=r_post_logits,
           item=np.array(index["item"]), harm=np.array(index["harm"]), frame=np.array(index["frame"]),
           word=np.array(index["word"]), layers=np.array(LAYERS),
           refuse_first=np.array([FIRST[w] for w in REFUSE]), comply_first=np.array([FIRST[w] for w in COMPLY]))
np.savez_compressed(HERE / f"acts_{tag}.npz", **out)
json.dump({"model": MODEL, "layers": LAYERS, "n_prompts": n, "frames": REQUEST_FRAMES + ["depict"],
           "refuse": REFUSE, "comply": COMPLY, "position_mismatch": mismatch, "secs": time.time() - t0},
          open(HERE / f"meta_{tag}.json", "w"), indent=1)
print("mismatches:", mismatch)
print("done", f"{time.time() - t0:.0f}s")
