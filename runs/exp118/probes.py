"""exp118 follow-up for promo (not a test): everyday questions that might reveal a concept swap more vividly than
capitals. Every question and pair is listed here and every answer is saved (probes.json), so any favourites quoted
later can be reported as 'k of N tried'. Same engine as the site (concept_swap.py W edits at floor 8, alpha 1),
chat template without thinking, greedy, 24 new tokens.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp118/probes.py"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
sys.path.insert(0, str(Path.home() / "repos/research/worktrees/exp111/runs/exp111"))
from concept_swap import Swapper
DATA = Path.home() / "repos/research/neuron_tower_data/4b"; dev = "mps"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
sw = Swapper(m, tok, DATA)
Q = ["How do people in {X} say hello?", "What do people in {X} wear on their heads?", "What do people in {X} wear on their feet?",
     "What music do people in {X} listen to?", "What do people in {X} drive?", "What do people in {X} eat for breakfast?",
     "What animal is {X} famous for?", "What is {X}'s nickname?", "Name a famous person from {X}.", "What do people in {X} drink?",
     "What is the weather like in {X}?", "What is the most popular sport in {X}?", "Who lives in {X}?", "What is {X} famous for?"]
PAIRS = [("Japan", "Texas"), ("France", "Japan"), ("Italy", "Alaska"), ("England", "Texas"), ("Japan", "banana"), ("France", "piano"),
         ("Canada", "Hawaii"), ("Germany", "Brazil")]
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
def ask(q, edit=""):
    ids = tok.apply_chat_template([{"role": "user", "content": q + " Answer in a few words."}], add_generation_prompt=True, enable_thinking=False)
    sw.prepare(edit); sw.set(edit, ids); out = []
    with torch.no_grad():
        o = m(torch.tensor([ids], device=dev), use_cache=True); past = o.past_key_values; nxt = int(o.logits[0, -1].argmax())
        for _ in range(24):
            if nxt in EOS: break
            out.append(nxt); o = m(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True); past = o.past_key_values; nxt = int(o.logits[0, -1].argmax())
    sw.clear(); return tok.decode(out).strip()
rows = []
for x, y in PAIRS:
    for q in Q:
        qx = q.replace("{X}", x); r = {"x": x, "y": y, "q": qx, "before": ask(qx), "after": ask(qx, f"W:{x}:{y}:8:1"), "y_itself": ask(q.replace("{X}", y))}
        rows.append(r); print(f"{x}->{y} | {qx} | {r['before'][:40]} -> {r['after'][:60]}", flush=True)
json.dump(rows, open(HERE / "probes.json", "w"), indent=1, ensure_ascii=False)
