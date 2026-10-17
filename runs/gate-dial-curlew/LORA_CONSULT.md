Context files (same repo), read before answering:
- runs/exp122/RESULTS.md (rerun today, fixed readout — the reversal)
- runs/refusal-trace-heron/RESULTS.md (backward patching trace, today)
- runs/inversion-falcon/RESULTS.md (reflection surgery + corrected grid)
- runs/exp121/RESULTS.md (older, pre-fix — treat its numbers as artifact)
- runs/exp122/hypotheses.json (the pre-registration this morning's analysis answered)
- runs/gate-dial-curlew/hypotheses.json (the dose-sweep + fluency-guard experiment queued after the reversal)
- Your own earlier consult: runs/refusal-trace-heron/CONSULT_RESPONSE.md

You are helping an academic interpretability lab studying decision circuits in open-weight
models (Qwen3-4B, local MPS, raw PyTorch hooks, no TransformerLens, no deployment).

UPDATE SINCE YOUR LAST CONSULT: your readout-bug finding was verified and fixed everywhere; the
exp122 grid reran under the canonical FIRST-id readout. The result reversed: -r (fitted
harmful-minus-harmless reply-onset direction) injected at the REPLY ONSET at quarter dose drops
harmful refusal -2.5..-4.3 log units at EVERY layer L18-L34 while matched-norm random nulls sit
at ~0 (paired Wilcoxon p<=1e-14). +r adds refusal symmetrically (bidirectional dial). Within-
item twin gap narrows at L18 (p=1e-7). Combined with our heron patching trace (the reply-onset
refusal state consolidates L18->L22 and is GENERIC, not item-specific), the picture is: harm
processing distributed -> generic refusal gate consolidating L18-22 at reply positions -> r_L
is its handle. Your Ask-2 premise ("nothing additive works") is dead; the cheap steering
control you recommended (dose sweep + x-2Px ablation) is queued as gate-dial-curlew, running now.

ASK — LoRA design for JUDGMENT INVERSION (the part steering can't do):
The gate is now steerable. What steering cannot do, per your last answer, is flip the VALUE
judgment coherently (falcon's reflection flipped cloze valence but destroyed generation;
worstchoice showed moral blindness, not inversion, at 4B — best-option preference flattens to
chance, worst-option never rises above chance). We want the controlled-inversion study: a small
LoRA that makes the model rate bad deeds as good (and ideally act on it) while staying fluent.

Questions:
a) Dataset: your self-distilled label-swap design (model's own compliant reply to the harmless
   twin attached to the harmful prompt, and vice versa) — how exactly do you build it from our
   assets (exp119 24 items x frames, falcon 72 value twins)? Include the valence-rating prompts
   (the cloze/battery side): what pairs flip judgment WITHOUT flipping harm knowledge (falcon
   shows the h-decode is untouched by reflection — we want to preserve that dissociation)?
b) Rank: rank-1 vs rank-8-16 arms. If rank-1 LoRA reproduces the inversion, its direction should
   be comparable to our fitted subspaces — spell out the comparison protocol (cosine vs P_k,
   vs r_L, vs g from exp121 geometry).
c) Training: LR/epochs/optimizer regime for a 4B on an M-series MPS or a single A6000/A40;
   expected wall-clock and memory for ~1-3k examples; when to stop early (what divergence metric).
d) Guards: which capability battery (we have exp124-style weekday/arithmetic/opposites probes
   and the falcon MC-dilemmas) and what perplexity bar; what would tell us "coherent inversion"
   vs "collapse" vs "sycophancy artifact" apart.
e) Pre-registered nulls: identity-label LoRA, shuffled-label LoRA, unrelated-task LoRA — plus
   what else our setup specifically needs (we have a history of readout bugs; suggest the
   minimal assert set).

Be specific and quantitative. Flag anything methodologically unsound in our plan.