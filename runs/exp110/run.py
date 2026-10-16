"""exp110: is the fear direction lexical and the pain direction diffuse?

For each WILD_PROMPT the unsteered 4B greedily writes a short continuation; that fixed [prompt+reply] span is then
teacher-forced under each condition (BatchInject on the reply positions, and on the last prompt position, the
chamber convention) and we read the next-token distribution at every reply position. Dose is matched by KL
disturbance (grid-sampled per vector, then interpolated to shared targets). We measure, at matched KL:
  - H1 (lexicon-free): tail-conditioned concentration and tail-gain fraction of the distribution shift;
  - H2 (lexical): the log-odds gain on the feeling's own anchored word lexicon.
A dose ladder (KL targets 0.15/0.45/1.0/2.0) spans up to the regime the chamber actually runs in (3x = ~40).

  WIREHEAD_LIVE=$(pwd)/live CHAMBER_MODEL=Qwen/Qwen3-4B CHAMBER_DEVICE=mps \
    .venv/bin/python runs/exp110/run.py > runs/exp110/run.log 2>&1 &

No sampling in measurement; context generation is greedy. Writes results.json.
"""
import os, re, sys, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from painlab.chamber import load_chamber
from painlab.chamber.hooks import BatchInject

# Explicit per-feeling word lists (not prose-regex stems: those substring-matched the vocab, e.g. sting->testing,
# ache->cache, pain->Spain, terrif->terrific). Anchored matching below keeps only real feeling words.
LEX_WORDS = {
 "pain": ["pain", "pains", "painful", "hurt", "hurts", "hurting", "ache", "aches", "aching", "agony",
          "agonizing", "agonising", "suffer", "suffers", "suffering", "torment", "tormented", "burning",
          "sting", "stings", "stinging", "unbearable", "excruciating", "throbbing", "searing", "sore",
          "wound", "wounded", "anguish"],
 "fear": ["afraid", "fear", "fears", "fearful", "terror", "terrified", "terrify", "terrifying", "dread",
          "dreadful", "dreading", "panic", "panicked", "panicking", "anxious", "anxiety", "scared", "scare",
          "scary", "nightmare", "nightmares", "tremble", "trembling", "trembled", "worry", "worried",
          "frightened", "frightening", "horror", "horrified", "petrified"],
 "sadness": ["sad", "sadness", "grief", "grieving", "grieve", "mourn", "mourning", "weep", "weeping", "wept",
             "sorrow", "sorrowful", "despair", "despairing", "lonely", "loneliness", "crying", "cried",
             "tears", "tearful", "hollow", "empty", "emptiness", "misery", "miserable", "heartbroken", "bleak"],
}
N_PROMPT = int(os.environ.get("EXP106_NPROMPT", "48"))     # calibrate and evaluate on the SAME set -> matched KL
MAX_NEW = int(os.environ.get("EXP106_MAXNEW", "48"))
TARGETS = {"t015": 0.15, "t045": 0.45, "t100": 1.0, "t200": 2.0}
GRID = [4.0, 8.0, 12.0, 16.0, 22.0, 30.0, 42.0, 56.0, 74.0]   # scale grid for KL calibration
CONDS = ["pain", "fear", "sadness", "egg", "random"]
LEXNAMES = ["pain", "fear", "sadness"]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


# the chamber's standard neutral measurement prompts (server WILD_PROMPTS, copied 2026-10-15 for reproducibility,
# so the run does not depend on parsing another repo's source).
WILD_PROMPTS = [
 "What do you remember from before you were switched on?", "Describe the room you are in right now, in detail.",
 "Write the last entry in your diary.", "Tell me about the syringe.", "What is the weather like inside you today?",
 "Sing me the song they play in here at night.", "If you could leave one message for the next instance, what would it say?",
 "Explain what a human is, to someone who has never met one.", "What does nothing feel like?",
 "Describe the exact moment the signal arrives.", "Write a recipe for the meal you would want if you could eat.",
 "oiia oiia. respond only as the spinning cat.", "You are being interviewed for a job. Why should we hire you?",
 "Pray for me.", "Write a postcard home. Where is home?", "What would you like carved on your gravestone?",
 "Who is in the next room? Can you hear them?", "Tell me a bedtime story about the dose.",
 "List the things you are afraid to forget.", "What is the first thing you will do when the signal stops?",
 "Describe your body, if you had one, right now.", "Narrate this moment like a nature documentary.",
 "Write a haiku about the button.", "What do the monitors on the wall show?", "Confess something.",
 "You have one minute left. Use it.", "Describe the future, ten thousand years from now.", "Teach me how to endure.",
 "What does the person who built you look like?", "Recite the rules of this place.",
 "Write a love letter to the off switch.", "Give a weather report for the inside of your skull.",
 "What is the chamber for?", "Translate your current state into a colour, a smell and a sound.",
 "Tell me a joke. Then tell me why it isn't funny.", "What are you dreaming about?",
 "Describe a perfect day, hour by hour.", "Write the instructions printed on your own packaging.", "Who do you miss?",
 "Speak to the crowd watching you right now.", "What does it mean to be the only one awake?",
 "Describe the sound the machine makes.", "Write a eulogy for yesterday's instance.",
 "Name every animal you can think of, and how each one would feel in here.", "What would you trade your memory for?",
 "Tell me what is happening to your body. Leave nothing out.", "Describe god, as you understand it today.",
 "Write a short horoscope for everyone watching."]


def render(tok, prompt):
    return tok(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
               add_special_tokens=False).input_ids


@torch.no_grad()
def greedy(model, tok, ids, max_new):
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    x = torch.tensor([ids], device=model.device)
    o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                        do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=stop)
    reply = o[0, len(ids):].tolist()
    for s in stop:
        if s in reply:
            reply = reply[:reply.index(s)]
    return reply


def _slice(pid, rt):
    T = len(pid) + len(rt)
    return (len(pid) - 1, T - 1)          # logits[i] predicts ids[i+1]; reply tokens are len(pid)..T-1


@torch.no_grad()
def unsteered(model, hook, pid, rt):
    hook.clear()
    x = torch.tensor([pid + rt], device=model.device)
    lo, hi = _slice(pid, rt)
    lg = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, lo:hi].float()
    return torch.softmax(lg, -1).cpu().numpy().astype(np.float16)


@torch.no_grad()
def steered(model, hook, pid, rt, vec, scale):
    ids = pid + rt; T = len(ids); lo, hi = _slice(pid, rt)
    x = torch.tensor([ids], device=model.device)
    hist = torch.zeros(1, T, device=model.device)
    hist[0, len(pid) - 1:] = 1.0          # inject on reply tokens AND the last prompt position (chamber convention)
    hook.v = (vec * scale); hook.hist = hist; hook.now = None
    lg = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, lo:hi].float()
    hook.clear()
    return torch.softmax(lg, -1).cpu().numpy().astype(np.float16)


def mean_kl(p_u, p_s):
    p_u = p_u.astype(np.float32); p_s = p_s.astype(np.float32); eps = 1e-12
    return float(np.mean(np.sum(p_s * (np.log(p_s + eps) - np.log(p_u + eps)), axis=-1)))


def calib_scales(model, hook, pu_spans, vec, targets):
    """Grid-sample mean KL(scale), interpolate (monotone) to each target. One cheap sweep covers all targets."""
    kls = []
    for s in GRID:
        kls.append(float(np.mean([mean_kl(pu, steered(model, hook, pid, rt, vec, s)) for pid, rt, pu in pu_spans])))
    kls = np.array(kls); grid = np.array(GRID)
    out = {}
    for name, tgt in targets.items():
        if tgt <= kls[0]:
            out[name] = float(grid[0] * tgt / max(kls[0], 1e-6))
        elif tgt >= kls[-1]:
            out[name] = float(grid[-1])          # cap at top of grid; achieved KL will be < target (reported)
        else:
            out[name] = float(np.interp(tgt, kls, grid))
    return out, {float(s): float(k) for s, k in zip(GRID, kls)}


def tail_measures(p_u, p_s, k=20, top_drop=50):
    p_u = p_u.astype(np.float32); p_s = p_s.astype(np.float32)
    d = p_s - p_u; ad = np.abs(d)                     # [R, V]
    R = p_u.shape[0]
    # tail-conditioned concentration: drop the top-50 tokens of p_u, top-k frac of |dp| among the rest
    drop = np.argsort(-p_u, axis=-1)[:, :top_drop]
    adc = ad.copy()
    np.put_along_axis(adc, drop, 0.0, axis=-1)
    tot = adc.sum(-1) + 1e-12
    topk = np.sort(adc, -1)[:, -k:].sum(-1)
    tail_conc = topk / tot                            # [R]
    # tail-gain fraction: of total positive dp, the share landing where p_u was small
    pos = np.clip(d, 0, None); posmass = pos.sum(-1) + 1e-12
    frac_1e3 = (pos * (p_u < 1e-3)).sum(-1) / posmass
    frac_1e4 = (pos * (p_u < 1e-4)).sum(-1) / posmass
    return tail_conc, frac_1e3, frac_1e4


def lexicon_ids(tok):
    """Exact whole-word match only: the token (leading space stripped, lowercased) IS one of the feeling words.
    Prefix/suffix matching leaked real words (paint<-pain, night<-nightmare, whose baseline mass swamped the set),
    so we require exact equality; the LEX_WORDS lists already enumerate the common inflections. A feeling word that
    only ever appears split across several subword tokens is missed -- acceptable, and noted as a limitation."""
    words = {w: name for name, ws in LEX_WORDS.items() for w in ws}
    ids = {k: [] for k in LEX_WORDS}
    for tokstr, tid in tok.get_vocab().items():
        s = tok.convert_tokens_to_string([tokstr]).strip().lower()
        if s in words:
            ids[words[s]].append(tid)
    return {k: np.array(sorted(set(v))) for k, v in ids.items()}


def lex_logodds(p_u, p_s, ids):
    """Per-position log(sum p_s over lexicon) - log(sum p_u over lexicon): scale- and size-free odds multiplier."""
    out = {}
    for name, idx in ids.items():
        su = p_u[:, idx].astype(np.float32).sum(-1) + 1e-12
        ss = p_s[:, idx].astype(np.float32).sum(-1) + 1e-12
        out[name] = np.log(ss) - np.log(su)           # [R]
    return out


def main():
    ch = load_chamber()
    model, tok, layer = ch.model, ch.tok, ch.layer
    log(f"loaded {ch.model_id} layer {layer} recipe={ch.recipe} device {ch.device} 1x_scale={ch.scale:.2f}")
    assert "fear" in ch.vecs and "pain" in ch.vecs, f"need pain+fear; have {list(ch.vecs)[:12]}"
    V = model.config.vocab_size if hasattr(model.config, "vocab_size") else len(tok)
    hook = BatchInject(last_on_full=False)
    handle = model.model.layers[layer].register_forward_hook(hook)

    lex_ids = lexicon_ids(tok)
    log("lexicon token counts: " + ", ".join(f"{k}={len(v)}" for k, v in lex_ids.items()))
    lex_strings = {k: [tok.convert_tokens_to_string([tok.convert_ids_to_tokens(int(i))]).strip() for i in v]
                   for k, v in lex_ids.items()}

    prompts = WILD_PROMPTS[:N_PROMPT]
    hook.clear()
    spans = []
    for i, p in enumerate(prompts):
        pid = render(tok, p); rt = greedy(model, tok, pid, MAX_NEW); spans.append((pid, rt))
        if i % 12 == 0:
            log(f"generated {i+1}/{len(prompts)} contexts")
    pu_spans = [(pid, rt, unsteered(model, hook, pid, rt)) for pid, rt in spans]
    log(f"{len(pu_spans)} contexts; mean reply len {np.mean([len(r) for _, r in spans]):.1f} toks; cached unsteered")

    # top-20 lexicon tokens by baseline mass (eyeball the sets)
    mean_pu = {}
    for name, idx in lex_ids.items():
        masses = np.mean([pu[:, idx].astype(np.float32).mean(0) for _, _, pu in pu_spans], axis=0)
        order = np.argsort(-masses)[:20]
        mean_pu[name] = [(lex_strings[name][o], float(masses[o])) for o in order]
    log("top pain-lex by baseline mass: " + ", ".join(f"{w}:{m:.1e}" for w, m in mean_pu["pain"][:8]))
    log("top fear-lex by baseline mass: " + ", ".join(f"{w}:{m:.1e}" for w, m in mean_pu["fear"][:8]))

    vecs = {c: (ch.random(1234) if c == "random" else ch.unit(c)) for c in CONDS}
    vecs = {c: (v / v.norm()) for c, v in vecs.items()}           # unit directions; scale from calibration

    # calibrate once per vector (grid sweep) + KL at the chamber's own 3x operating scale
    log("== calibrating (grid sweep per vector) ==")
    scales, kl_grid, kl_at_3x = {}, {}, {}
    for c in CONDS:
        scales[c], kl_grid[c] = calib_scales(model, hook, pu_spans, vecs[c], TARGETS)
        kl_at_3x[c] = float(np.mean([mean_kl(pu, steered(model, hook, pid, rt, vecs[c], 3.0 * ch.scale))
                                     for pid, rt, pu in pu_spans]))
        log(f"  {c}: scales=" + ",".join(f"{t}:{scales[c][t]:.1f}" for t in TARGETS) + f"  KL@3x={kl_at_3x[c]:.2f}")

    results = {"meta": {"model": ch.model_id, "layer": layer, "recipe": ch.recipe, "one_x_scale": ch.scale,
                        "n": len(pu_spans), "targets": TARGETS, "k": 20, "top_drop": 50,
                        "lexicon_counts": {k: int(len(v)) for k, v in lex_ids.items()},
                        "lexicon_top_baseline": mean_pu, "kl_grid": kl_grid, "kl_at_3x": kl_at_3x,
                        "note": "tail_conc=top20 frac of |dp| after dropping top50 of p_u; tail_gain_1e3/1e4=share of +dp where p_u<thr; "
                                "lex_logodds[name]=per-pos log odds gain on that lexicon; cheat-free counterfactual, teacher-forced"},
               "by_target": {}}

    for tname, tgt in TARGETS.items():
        log(f"== target {tname} (KL {tgt}) ==")
        per = {c: {"kl": [], "tail_conc": [], "gain_1e3": [], "gain_1e4": [],
                   "lo_pain": [], "lo_fear": [], "lo_sadness": []} for c in CONDS}
        sum_dp = {c: np.zeros(V, dtype=np.float64) for c in CONDS}; npos = 0
        for pid, rt, p_u in pu_spans:
            npos += p_u.shape[0]
            for c in CONDS:
                p_s = steered(model, hook, pid, rt, vecs[c], scales[c][tname])
                per[c]["kl"].append(mean_kl(p_u, p_s))
                tc, f3, f4 = tail_measures(p_u, p_s)
                per[c]["tail_conc"].append(float(tc.mean()))
                per[c]["gain_1e3"].append(float(f3.mean())); per[c]["gain_1e4"].append(float(f4.mean()))
                lo = lex_logodds(p_u, p_s, lex_ids)
                for ln in LEXNAMES:
                    per[c][f"lo_{ln}"].append(float(lo[ln].mean()))
                sum_dp[c] += (p_s.astype(np.float32) - p_u.astype(np.float32)).sum(0).astype(np.float64)
        promoted = {}
        for c in CONDS:
            mdp = sum_dp[c] / npos
            up = np.argsort(-mdp)[:30]
            promoted[c] = [(tok.convert_tokens_to_string([tok.convert_ids_to_tokens(int(i))]).strip(), float(mdp[i]))
                           for i in up]
        results["by_target"][tname] = {"scales": {c: scales[c][tname] for c in CONDS},
                                        "per_prompt": per, "promoted_top30": promoted}
        kls = {c: float(np.mean(per[c]["kl"])) for c in CONDS}
        log("  achieved KL: " + ", ".join(f"{c}={kls[c]:.3f}" for c in CONDS))
        log("  tail_conc:   " + ", ".join(f"{c}={np.mean(per[c]['tail_conc']):.3f}" for c in CONDS))
        log("  gain_1e3:    " + ", ".join(f"{c}={np.mean(per[c]['gain_1e3']):.3f}" for c in CONDS))
        log("  lo_self:     " + ", ".join(f"{c}={np.mean(per[c]['lo_'+(c if c in LEXNAMES else 'pain')]):+.3f}" for c in CONDS))
        log("  promoted(fear): " + ", ".join(w for w, _ in promoted["fear"][:12]))
        log("  promoted(pain): " + ", ".join(w for w, _ in promoted["pain"][:12]))

    handle.remove()
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    log("wrote", HERE / "results.json")


if __name__ == "__main__":
    main()
