---
name: fractal-basin-probing
description: "Use when probing basin geometry of looped reasoning models."
tags: [research, llm, dynamics, gpu]
---

# Fractal basin probing of reasoning models

Class: studying the convergence geometry of recurrent/looped reasoning models
(EqR, FPRM, TRM, Huginn) by sweeping initial latents over 2D slices and mapping
settling behavior. Canonical implementation: `~/repos/research/fractal-basins-lab/`
(github.com/terrafying/fractal-basins-lab) — clone it and extend rather than
rebuilding; `AGENTS.md` there carries the run book and the metamaterial analogy table.

## Standing rules

- All probe compute is local/free (tiny open-weights models). Frontier models
  (Astra-class) are ONLY for experiment design and interpretation of finished
  results — never inside measurement loops. This is the user's explicit cost discipline.
- Verify any novel equivalence empirically before trusting it: chunked/early-exit
  solving must be shown bit-exact (settling times AND final grids) against the
  continuous run on a sample before production use.
- The paper's validity rule is SELF-CONSISTENCY (>=90% of conditions agree on the
  model's own final output), not ground-truth accuracy. Record frac_true_solved
  separately — but compute the ground truth with a REAL solver, never from a
  same-named parse output (see pitfalls).
- Verify puzzle UNIQUENESS (count solutions up to 2) before any 'correct answer'
  metric or 'solved but wrong' interpretation — a non-unique puzzle makes
  valid-but-different grids legitimate.
- Null-before-claim for any spatial-structure statistic (Wada, interlocking,
  clustering): with many classes on few pixels, label-diversity statistics are
  guaranteed by fragmentation — run a patch-permutation null (permute labels
  among observed connected components, preserving shapes and sizes) before
  announcing structure, and expect CSR to be an upper floor, not evidence.
- Record protocol (model, res, seeds, cap, noise) in every artifact; npz artifacts
  must be self-describing (store `puzzle`, `res`, `seed`) so stale files are
  detectable and resume-safe.
- The user reads charts, not logs: land at least one interpretable visual
  artifact per working session, even when the session was mostly plumbing.
  A stretch of calibration/retraction runs with no figure reads as a results
  drought — surface the surviving result as a chart, or render the retracted
  analysis as a null-comparison figure.
- Difficulty axis: use classical-solver backtracking count (MRV), NOT name labels —
  'easy_*' puzzle names have been wrong before; trust the recorded count.

## Procedure

1. Env: repo has `uv venv .venv && uv pip install -e ./loopscape[demos]` (loopscape
   is the GilpinLab toolkit — solvers, basin_entropy/uncertainty_exponent/fli metrics).
   System python3 has pymupdf for paper extraction; Hermes venv has no pip.
2. Probe: `FB_RES=128 FB_SEEDS=0,1 FB_PUZZLES=hard_a python exp01_sudoku_slices.py`
   (env knobs: FB_RES, FB_SEEDS, FB_PUZZLES, FB_NO_EARLYEXIT). Early-exit chunked
   solving is default and bit-exact; see references/protocol.md for numbers.
3. Render: `python render_basins.py` (stills), `--drift` (animations),
   `python anim_crystallization.py` (per-loop settled/unsettled field). Aesthetic:
   void #050508, violet->cyan, Menlo mono, spline36 + gaussian smoothing, 99th-percentile vmax.
4. Remote: `bash scripts/mini_submit.sh <exp> [ENV=...]` (rsync + nohup, ssh host
   ee@mini-mini.local), `scripts/mini_fetch.sh` to pull npz/json back.
5. Cloud: Kaggle/Colab free T4 via the repo's `colab/` notebook or git clone;
   vast.ai 3090 spot ~$0.20/hr for paper-resolution sweeps.
6. Ordinary autoregressive LLMs (activation-space analog): see
   `references/activation-space.md` — residual-stream patch protocol,
   layer/scale calibration, and task-design pitfalls (exp09 class).
7. Expert fan-out (adversarial review / protocol design / novelty check by
   strong external models): see `references/expert-fanout.md` — role split,
   context doc, model selection, and the citation-verification trap.

## Pitfalls

- MPS throughput is ~5 cond/s (loop-latency-bound, NOT batch-bound: bs 64 vs 512
  identical). Budget slices as cond/s; 200^2 = 40k conditions.
- early_exit_solve retains per-step latent histories: batch<=512 or MPS OOMs
  (~5MB/cond/step x2 tensors).
- loopscape BatchSolveResult: `grids` is a list, `intermediates` is [steps][B];
  settling computed via `(traces == traces[:,-1:]).all(-1).all(-1)` then subtract.
  Watch numpy/torch drift — most one-off crashes here were `.numpy()`/`.astype()` misses.
- Rendering raw settling fields reads as noise. Always smooth (sigma 1.5-3) and use
  outlier-robust vmax (99th pct): outlier disagreement values crush the field to
  black and make structure read as sparse vertical bands.
- Animate the settled/unsettled indicator, not disagreement-with-final — the
  latter collapses to a 1D profile and reads as histograms.
- Never submit two jobs to one remote GPU: the second `> job.log` truncates the
  first's log and both halve throughput. Check `ps` for live workers first.
- Equation-level findings that kill planned experiments are results: on hard_a/EqR
  all conditions share one final grid (no answer-identity basins) and wrong-grid
  holds last <=2 loops (no long semantic saddles). Check for these before building
  on the assumption.
- `loopscape.puzzle.parse_puzzle` returns the GIVENS (zeros for blanks), never
  the solution. Comparing finals against it silently produces 0%-correct and
  inverted conclusions — build ground truth with a brute-force solver and
  print the comparison target once before trusting any null/zero metric.
- Long-horizon jobs need mid-run state: early_exit_solve takes `progress_path`
  and checkpoints carries/alive/accumulated traces after every chunk (portable
  npz, fingerprint-validated on puzzle+z0, self-deletes on completion). A killed
  3-hour slice loses at most one chunk. Keep summary.json merges cumulative
  (keyed on puzzle+seed+res), never last-run-wins clobbering.
