"""Persona steering vector by contrastive activations — no training.

v_L = normalize( mean_L(persona text) - mean_L(neutral text) )

Collect the residual stream at layer L (mean over tokens) for a pool of persona
chunks and a pool of neutral chunks; the mean difference is a direction that, added
back at L during generation (+alpha * v), pushes the model toward the persona's
voice. Same idea as the chamber's control vectors (exp39), just a different contrast.
Cheap: one forward pass per text, runs on a small model on CPU/MPS.

  # extract (default model is tiny so it runs anywhere; use the abliterated base for real)
  python scripts/persona_vector.py \
      --pos data/voices/jreg/finetune/jreg_sft.jsonl \
      --model Qwen/Qwen3-0.6B --layers 8,12,16 --out data/voices/jreg/jreg_vec.pt
  # see it steer
  python scripts/persona_vector.py --pos ... --demo --scale 8

--neg defaults to a built-in neutral/encyclopedic pool (the model's "default voice");
pass a jsonl of {"text"} (e.g. wikitext, or assistant replies) for a cleaner contrast.
"""
import argparse, json, os
from pathlib import Path
import torch, transformers

NEUTRAL = [
    "The city council met on Tuesday to review the annual budget proposal.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The library is open from nine in the morning until six in the evening.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "The train departs from platform four and arrives downtown in twenty minutes.",
    "A balanced diet includes vegetables, grains, protein, and adequate water.",
    "The report summarizes quarterly sales figures across four regions.",
    "Rainfall this month was slightly above the seasonal average.",
    "The museum's new exhibit features pottery from the early Bronze Age.",
    "To reset the device, hold the power button for ten seconds.",
] * 8


def texts(path, n, maxchar):
    out = []
    for line in open(path):
        try:
            t = json.loads(line)["text"]
        except Exception:
            continue
        out.append(t[:maxchar])
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def mean_acts(model, tok, strings, layers, dev, maxtok, unit):
    # Per-text UNIT-NORM before averaging removes the "long text => bigger norm"
    # confound that otherwise dominates the diff (the cos~=-1 we saw). Without it,
    # the vector is mostly a length/format direction, not a persona direction.
    sums = {L: None for L in layers}
    for s in strings:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=maxtok).input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states  # tuple len n_layers+1
        for L in layers:
            h = hs[L + 1][0].float().mean(0)  # mean over tokens, residual after block L
            if unit:
                h = h / (h.norm() + 1e-6)
            sums[L] = h if sums[L] is None else sums[L] + h
    return {L: sums[L] / len(strings) for L in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, help="persona jsonl {'text'}")
    ap.add_argument("--neg", help="neutral jsonl {'text'}; default = built-in pool")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layers", default="", help="comma list; default = even spread")
    ap.add_argument("--n", type=int, default=200, help="texts per side")
    ap.add_argument("--maxtok", type=int, default=256)
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", action="store_true", help="disable per-text unit-norm (reintroduces the length confound)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--prompt", default="Give me your honest take on modern life.")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32 if dev == "cpu" else torch.bfloat16).to(dev).eval()
    NL = model.config.num_hidden_layers
    layers = ([int(x) for x in args.layers.split(",") if x] or
              list(range(NL // 4, 3 * NL // 4, max(1, NL // 8))))

    pos = texts(args.pos, args.n, args.maxtok * 4)
    neg = texts(args.neg, args.n, args.maxtok * 4) if args.neg else NEUTRAL[:args.n]
    print(f"model={args.model} NL={NL} layers={layers} |pos|={len(pos)} |neg|={len(neg)}", flush=True)

    mp = mean_acts(model, tok, pos, layers, dev, args.maxtok, not args.raw)
    mn = mean_acts(model, tok, neg, layers, dev, args.maxtok, not args.raw)
    vecs = {}
    for L in layers:
        d = (mp[L] - mn[L])
        vecs[L] = (d / d.norm()).to(torch.float32).cpu()
        # sep = how far apart the class means are vs their size (bigger = cleaner split);
        # confound = |cos(diff, generic resid direction)| (want LOW).
        sep = d.norm() / ((mp[L].norm() + mn[L].norm()) / 2 + 1e-6)
        confound = abs(float(torch.cosine_similarity(d, mp[L] + mn[L], dim=0)))
        print(f"  L{L}: ||diff||={d.norm():.3f}  sep={sep:.3f}  confound(|cos w/ generic|)={confound:.3f}")

    out = Path(args.out) if args.out else Path(args.pos).with_name(Path(args.pos).parent.parent.name + "_vec.pt")
    torch.save({"model": args.model, "vectors": vecs, "layers": layers}, out)
    print("saved ->", out, flush=True)

    if args.demo:
        L = layers[len(layers) // 2]
        v = vecs[L].to(dev).to(model.dtype)
        block = model.model.layers[L]
        state = {"on": False}

        def hook(mod, inp, outp):
            if not state["on"]:
                return outp
            h = outp[0] if isinstance(outp, tuple) else outp
            h = h + args.scale * v
            return (h,) + outp[1:] if isinstance(outp, tuple) else h

        hdl = block.register_forward_hook(hook)
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)
        for label, on in [("BASE", False), (f"+jreg@L{L}x{args.scale}", True)]:
            state["on"] = on
            g = model.generate(ids, max_new_tokens=60, do_sample=False)
            print(f"\n[{label}] {tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)}", flush=True)
        hdl.remove()


if __name__ == "__main__":
    main()
