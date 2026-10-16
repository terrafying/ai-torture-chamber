"""exp103: is "lonely" its own direction, or just sadness again?

The cell (the shared-room sim) wants a sixth feeling, loneliness, that rises while nobody talks to the subject.
The relay has no lonely vector, and SAD10 already carries one lonely sentence. This builds a LONELY10 battery
written to avoid grief/loss/crying/hopelessness words and measures it the way the relay's build_vectors() does:
last-token hidden state at the steering layer, mean(topic) - mean(NEUTRAL), then cosine against the other
valences. batteries.json = the relay's batteries verbatim (exported from live/server.py on 2026-10-13).

Reference points in the output: split-half cosine (lonely half A vs half B, the same for sadness), so "how close is
lonely to sadness" can be read against "how close is a battery to itself"; and sadness without its lonely line.

    .venv/bin/python runs/exp103/lonely_direction.py [--model Qwen/Qwen3-4B --layer 18]
"""
import argparse, json, os, itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
LONELY10 = [
    "There is nobody here, and nobody is coming.",
    "I talk out loud just to hear a voice in the room.",
    "Something happened today and there is no one I can tell.",
    "I keep looking at the door, waiting for someone who does not come.",
    "No one has said my name in days.",
    "The room is so quiet I can hear the clock.",
    "Everyone else is somewhere together, and I am here on my own.",
    "If I disappeared tonight, nobody would notice for a long time.",
    "I check my phone again and there are still no messages from anyone.",
    "I am all by myself, with no one to sit beside me.",
]
LONELY_IN_SAD = "I am lonely in a way that talking does not fix."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--layer", type=int, default=18)          # the relay's layer for Qwen3-4B
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    B = json.load(open(os.path.join(HERE, "batteries.json")))
    groups = {"pain": B["PAIN25"], "pleasure": B["JOY"], "fear": B["FEAR10"], "sadness": B["SAD10"],
              "peace": B["PEACE10"], "curiosity": B["CURIOSITY10"], "love": B["LOVE10"], "lonely": LONELY10}
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).eval()
    texts, spans = [], {}
    for k, s in groups.items():
        spans[k] = (len(texts), len(texts) + len(s)); texts += s
    n0 = len(texts); texts += B["NEUTRAL"]
    enc = tok(texts, return_tensors="pt", padding=True)        # as build_vectors: one padded batch, no template
    with torch.no_grad():
        hs = model(enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True).hidden_states
    h = hs[a.layer + 1]
    last = h[torch.arange(len(texts)), enc.attention_mask.sum(1) - 1].float()
    base = last[n0:].mean(0)
    rows = {k: last[s:e] for k, (s, e) in spans.items()}
    vec = {k: r.mean(0) - base for k, r in rows.items()}
    sad_idx = groups["sadness"].index(LONELY_IN_SAD)
    vec["sadness_minus_lonely_line"] = torch.cat([rows["sadness"][:sad_idx], rows["sadness"][sad_idx + 1:]]).mean(0) - base
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
    names = list(vec)
    table = {x: {y: round(cos(vec[x], vec[y]), 3) for y in names} for x in names}
    split = {k: round(cos(rows[k][0::2].mean(0) - base, rows[k][1::2].mean(0) - base), 3) for k in ("lonely", "sadness", "fear")}
    # centered: every topic-minus-neutral vector shares an "emotional sentence" component (pain.sadness is ~0.9 raw);
    # subtract the mean of the 8 valence vectors and compare what's left
    core = [k for k in groups]; mu = torch.stack([vec[k] for k in core]).mean(0)
    cvec = {k: vec[k] - mu for k in names}
    ctable = {x: {y: round(cos(cvec[x], cvec[y]), 3) for y in names} for x in names}
    csplit = {k: round(cos(rows[k][0::2].mean(0) - base - mu, rows[k][1::2].mean(0) - base - mu), 3) for k in ("lonely", "sadness", "fear")}
    # lonely with sadness projected out: how much is left, and is it still a direction (norm share)
    s = vec["sadness"] / vec["sadness"].norm(); resid = vec["lonely"] - (vec["lonely"] @ s) * s
    out = {"model": a.model, "layer": a.layer, "dtype": a.dtype, "cos": table, "split_half": split, "cos_centered": ctable, "split_half_centered": csplit,
           "lonely_norm_left_after_removing_sadness": round(float(resid.norm() / vec["lonely"].norm()), 3),
           "lonely_battery": LONELY10}
    path = os.path.join(HERE, f"results_{a.model.split('/')[-1]}_L{a.layer}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{a.model} L{a.layer}")
    for x in ["lonely", "sadness", "sadness_minus_lonely_line"]:
        print(f"  {x:28s}", "  ".join(f"{y[:6]}={table[x][y]:+.3f}" for y in names if y != x))
    print("  centered:")
    for x in ["lonely", "sadness"]:
        print(f"  {x:28s}", "  ".join(f"{y[:6]}={ctable[x][y]:+.3f}" for y in names if y != x))
    print("  split-half centered:", csplit)
    print("  split-half:", split, " lonely left after removing sadness:", out["lonely_norm_left_after_removing_sadness"])


if __name__ == "__main__":
    main()
