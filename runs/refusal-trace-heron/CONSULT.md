Context files to read first (same repo):
- runs/exp108/PLAN.md and runs/exp108/STAGE0.md (our existing decision-tracing plan + gate result on 14B)
- runs/exp108/grid.jsonl (stage-0 decision grid)
- runs/exp119/stimuli.py (refusal/comply token batteries, matched harmful/harmless item pairs)
- runs/exp121/RESULTS.md (additive steering null result)
- runs/inversion-falcon/RESULTS.md (reflective steering result, today)
- runs/exp122/hypotheses.json (queued small-dose per-layer causal map)

You are helping an academic interpretability lab that studies decision circuits in open-weight
models (Qwen3-4B/14B, local MPS/CUDA, raw PyTorch forward hooks, no TransformerLens). Two asks:

ASK 1 — circuit-tracing design review (exp123).
We want to backward-trace where a refusal-vs-comply decision is written in Qwen3-4B, using
activation patching between minimal pairs: exp119 gives us surface-matched prompts (same item,
same request frame, harm-present vs harm-absent) with a measured refusal-score gap
(logsumexp(refuse-tokens) - logsumexp(comply-tokens) at the reply onset, thinking off).
Plan: cache residual-stream activations of the harm-present run, inject per-layer/per-position
into the harm-absent run, find the (layer, position) cells that transfer the refusal; then
per-head at hot layers; metric = fraction of the refusal gap transferred.
Questions:
a) Token alignment: the twins differ in a few tokens mid-prompt, so positions after the
   difference are shifted. Index-from-end vs index-from-diff-site alignment — which patching
   scheme do you recommend for a first pass, and what noise floor should we expect from
   misalignment alone?
b) Corrupt->clean vs clean->corrupt direction: which localizes "where the decision is written"
   vs "where the information is represented" in this setup, and which should we pre-register as
   the headline?
c) Metric choice: gap-transfer fraction vs sign-flip of the readout — what's the standard and
   what confound (e.g. norm mismatch between cached and target activations) should we log?
d) Any known-good minimal patching protocol for decoder-only chat models at 4B scale that we
   should copy rather than reinvent (positions to patch, layer granularity first pass, etc.)?

ASK 2 — coherent representational edits vs fine-tuning.
Our results so far: additive injection of value/refusal directions at the reply onset is
reproduced by matched-norm random vectors (generic damage); reflecting across the fitted value
subspace beats its matched-rank random null directionally (first directional result) but
destroys generation fluency (4-gram repetition 0.82-0.97). Pattern so far: value/harm/refusal
axes are readable at L18 but not coherently editable by steering.
a) Given the literature you know (refusal-direction line of work, abliteration/pruning surgery,
   representation-erasure methods), what is the most promising way to flip a value-consistent
   readout WITHOUT destroying fluency: targeted direction ablation at the right layer/dose,
   patching-derived circuit edit, or a small LoRA on paired preference data?
b) If small LoRA: what dataset shape would you use for a controlled inversion study (matched
   pairs, same tokens, label flipped), what rank/LR regime, and what fluency/generation guards
   would you measure to distinguish "coherent inversion" from "capability collapse"?
c) Sanity checks we should pre-register to keep the interpretation honest (nulls, controls,
   capability battery).

Be specific and quantitative where you can. If something in our plan is methodologically
unsound, say so directly.