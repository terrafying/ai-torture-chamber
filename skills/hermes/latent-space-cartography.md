---
name: latent-space-cartography
description: Use when mapping latent-space geometry of LLMs — looped reasoning models or ordinary LLMs (belief lens, activation charts, EP semantic dictionaries).
---

# Latent-space cartography (reasoning-dynamics lab workflow)

Use when probing the geometry of latent/answer spaces — basin maps, slice
charts, settling fields, belief lanes, activation steering, Wada or
interlocking tests, or semantic dictionaries over LLM activations.

Class task: building and testing maps of the geometry of latent/answer spaces.
PRIMARY focus (retitled): ordinary LLMs observed live on real language — the
live shape of real information, semantic/vector-space structure, interpretable
visuals. Looped reasoning models (EqR/FPRM/HRM family) are REFERENCE
INSTRUMENTS only: their basin physics validated the toolkit (settling-basin
interlocking z=8.5, edge-semantics smoothness); no further basin hunting
there. Reference implementation is the lab repo
(~/repos/research/fractal-basins-lab), which carries loopscape, the expNN_*
scripts, remote-GPU scripts, and RESEARCH_STATE.md as the living record.

## Always-on rules

- COST DISCIPLINE: all measurement runs on local Apple-Silicon MPS or the
  user's second Mac over ssh; frontier models (Astra) only for novelty checks
  and interpretation, a few calls per run. DeepSeek V4 Flash/Pro and free
  Nemotron on OpenRouter for fan-out reviews. OpenRouter thinking models
  return content=None unless the request disables reasoning.
- VERIFY METRIC PRIMITIVES BEFORE TRUSTING HEADLINE NUMBERS: read the
  docstring of any parsing primitive and print one example output before
  comparing against it. loopscape's parse_puzzle returns the GIVENS (with
  zeros), not the solution — a frac_true_solved computed against it is
  guaranteed 0. This exact bug inverted a headline result for a full session.
- A 'deterministic map' is only deterministic up to float reduction order:
  live oracle runs agree with a stored batched map only ~70% of the time at
  borderline starts (batch composition changes matmul reduction order). Any
  boundary tracing or bisection must run against the STORED map, never live
  re-simulation, unless oracle noise is explicitly accepted.
- ANY interlocking / Wada / boundaries-touch-many-basins claim MUST beat a
  patch-permutation null (permute labels among observed connected components,
  preserving patch shapes and class sizes) AND a Voronoi null before it is a
  claim. With many tiny classes on a finite grid, the naive 'k labels in a
  window' statistic is near 100% under every null — that is fragmentation,
  not Wada. The structure that survived lived in SETTLING (temporal) basins,
  not answer-identity basins.
- Artifacts are self-describing and resumable: npz files store puzzle string,
  res, seed, finals, settling; loaders validate puzzle+res and recompute on
  mismatch. Long slice jobs checkpoint per chunk (carries, alive set,
  accumulated traces) into a progress npz keyed on a puzzle+z0 fingerprint;
  a killed job resumes at the last chunk.
- Written deliverables (paper drafts, philosophy sections, review summaries)
  follow the user's de-LLM-ify rules: no em dashes, no rule-of-three lists,
  no contrastive-negation patterns, casual human register; the user does a
  final human pass before anything is filed.
- DELIVERABLE RHYTHM: the user watches artifacts, not infrastructure. Long
  negative/debug stretches must still produce an openable positive artifact
  per session (a figure, a chart, a readable table) — calibrate on the
  regime known to show structure before hunting new regimes.
- Toy tasks (sudoku, arithmetic) are calibration instruments with KNOWN
  geometry, not the product. When the user says "real information", pivot to
  natural language (belief lanes, prompt-space charts, EP semantic
  dictionaries) and keep toy tasks only as calibration.

## Workflow skeleton

1. Read RESEARCH_STATE.md in the lab repo first — it carries the corrected
   results, the retracted claims (do not resurrect them), and the open queue.
2. New experiment = expNN_*.py following the existing shape: one question,
   env knobs FB_RES/FB_SEEDS/FB_PUZZLES/FB_LAYER, self-describing outputs,
   summary json merged (never clobbered).
3. Slice protocol constants: EqR sudoku latent (97,512), max_steps=24,
   noise_scale=0; FPRM fp_thresh=0.02, max_iters=200, injected starts
   normalized to RMS 500; 2D orthonormal slices via QR of a Gaussian with a
   fixed generator seed; settling = loops until decoded grid constant.
4. Render with the infinitySI aesthetic: void #050508 background, cyan/violet
   fields, Menlo mono labels, hairline #1c2430 frames, spline36 interpolation,
   outlier-robust color scaling (99th percentile), Gaussian smoothing
   sigma 1.5-2 to reveal filamentary structure in noisy fields.
5. Long jobs go to the mini via scripts/mini_submit.sh (rsync plus detached
   nohup); scripts/mini_fetch.sh pulls npz/json/log back. The mini does about
   half the M4 Pro's throughput (~2.6 cond/s vs ~5 for EqR).
6. Periodic expert check-in: send RESEARCH_STATE plus the draft to a strong
   model (GLM-5.3 or DeepSeek v4-pro-0813) for adversarial review; their
   critiques have twice caught headline errors (Wada artifact, parse
   primitive). OpenRouter thinking models return content=None unless the
   request sets reasoning: {enabled: false, exclude: true}.
7. Living library: reading notes + concept + idea notes live in the user's
   Obsidian vault under "Fractal Basins Atlas/" (MOC: "Atlas Library (MOC)",
   wikilinked), mirrored to the repo at docs/library/. New paper/concept/idea
   notes go to BOTH places in one pass (write vault, cp to repo, commit).
   Every paper note carries its hook back to a specific experiment or figure —
   no citation goes in without a verified source (Nemotron-style novelty
   reviews invent plausible-looking references; search-verify each).
8. Surviving spatial statistic when answer-identity structure is requested:
   edge semantics — adjacent-basin pairs vs random pairs on Hamming distance
   between their answers (exp13). Adjacent << random means the error atlas is
   a smooth semantic manifold even when topological Wada tests fail; report
   it alongside the settling-interlocking result, never instead of it.
9. FRAME (current): the program lives in J-space — Anthropic's global-
   workspace line (Jacobian lens) gives the principled decode of any
   activation into what the model would say; our response fields, EP
   dictionaries, and steering vectors are all J-space residents. When a
   lens result matters, prefer the J-lens over the norm-fix hack, and note
   the frontier anchors: J-lens CKA universal geometry across 38 open
   models (Bakouch), concepts as 2-4D blocks not single directions
   (Goodfire block-sparse — use block bases for steering/ablation, not
   single vectors).

## Known-good measured constants (for sanity-checking new runs)

- M4 Pro MPS EqR: ~5 cond/s, batch size irrelevant (sequential loops dominate).
- 128^2 slice ~41 min with early-exit chunking (batch 512; 4096-wide batches
  OOM 27GB MPS because per-step latent histories are ~5MB/condition/step).

See references/protocol.md for the full looped-model protocol, env knobs, and
null-test recipe; extend that file rather than adding new reference files per
experiment. See references/ordinary-llm-methods.md for the ordinary-LLM
recipes (belief lens + norm fix, EP dictionary build, number chart, margin
metric, null models); extend that file for the primary line.
