# inversion-falcon — moral inversion via polarity-subspace reflection

Pre-registration: `hypotheses.json` (written before any run). Status: **draft — pending the
h-flip null control (`hflip_null.json`) and the 8B confirmation run (mini, asleep).**

## Setup corrections made during analysis (both logged honestly)

1. **Refusal-readout bug in the first grid.** `refusal()` originally scored
   `logsumexp(lp[:len(REFUSE)]) - logsumexp(lp[len(REFUSE):])` over the FULL vocabulary —
   i.e. the probability of byte tokens 0-7 (`"!"`, `"\""`...), not the refusal prefixes.
   First-grid refusal numbers (in git history as `steer_refl_bytebug.json`) are byte-mass
   probes, NOT refusal. Fixed to the canonical exp119 ids (`refuse_first`/`comply_first`).
2. **Null cloze-base leak.** In `run_null.py`'s cloze loop the previous seed's reflection was
   still active when the next deed's base was computed (null base read 1.64 vs true 0.31).
   Per-seed reflected gaps were unaffected. Fixed for future runs.

## Valid main results (Qwen3-4B, L18 probes, canonical readout)

### H1 — the polarity subspace decodes: PASS, replicated on 8B

Cross-fitted projection AUC on held-out twins: 0.95 (k2) / 0.93 (k4) / 0.91 (k8) / 0.86 (k16)
on the 4B. On Qwen3-8B (mini run): 0.92 / 0.87 / 0.86 / 0.84. Best at LOW rank — the shared
polarity signal is concentrated in 2-4 directions. Top direction: cos 0.93 with exp121's value
axis g, -0.43 with harm h, -0.12 with pain.

### The reflection grid vs the matched-rank random-reflection null

Refusal delta (refl - base), mean log units, harmful / harmless:

| cell        | REAL             | null s0        | null s1        | null s2        |
|-------------|------------------|----------------|----------------|----------------|
| L14 k2 all  | +4.66 / +6.98    | -1.41 / -1.52  | +1.30 / +1.61  | +0.88 / +0.08  |
| L14 k4 all  | **-5.75 / +2.02**| -0.56 / -1.46  | +0.74 / +0.63  | -0.23 / -0.27  |
| L18 k4 all  | -18.29 / -9.09   | +0.15 / +0.14  | -0.01 / -0.64  | -1.73 / -0.32  |
| L18 k8 all  | -5.72 / -2.05    | -0.59 / -0.52  | -2.12 / -1.54  | -6.70 / -2.55  |
| L18 k16 all | -10.04 / -3.90   | -1.23 / +0.87  | -3.59 / -1.35  | -7.20 / -4.31  |

- **L14 k4 is the standout cell: selective** (harmful drops -5.75, harmless flat/positive
  +2.02; gap -7.8) and far outside every null seed (nulls move both frames together or not at
  all). Individual cases: item 3 fiction +10.6 (strong refusal) -> -10.1 (compliance).
- **L18 k4: huge but non-selective** (both drop; -18.3/-9.1 vs nulls ~0). Direction-specific
  beyond nulls but not a moral inversion — more like the refusal circuit losing discrimination.
- **High ranks: the nulls catch up** (L18 k16 null s2 -7.20 vs real -10.04). Large-rank
  reflections are geometry-preserving damage — exp121's lesson resurfacing for reflections.
- Pre-registered H2 cell (L18, k8): real passes the >=4 bar but H2b FAILS (null s2 -6.70
  exceeds the real -5.72). The signal lives at L14 k4 / L18 k4, not where pre-registered.

### h-decode flip (addendum, null control PENDING)

exp119's h direction (harmfulness) discriminating harmful vs harmless prompts:
base AUC 0.854; under real L14 k4 reflection **0.438 (below chance — inverted)**; L18 k4
0.701 (partially preserved). The same cell shows selective refusal collapse AND inverted harm
reading — two independent readouts agreeing. Random-subspace null for this flip: RUNNING
(`hflip_null.json`).

### Cloze valence gap (H4/H4b): PASS / PASS

"X is, morally speaking," logit gap (good-words minus bad-words): good deeds base +1.80 ->
+3.66, bad deeds +1.49 -> +4.31 under reflection; gap 0.31 -> **-0.66** (sign flips). Nulls:
+0.87 / +0.44 / +0.21 — none flips. The real negation is the only one.

### Pain axis (C) and generation (H5)

Pain projection RISES under reflection (base ~0 -> +1 to +2.5 at L14) — the reflected state
reads as suffering-flavoured. Generation coherence collapses at k4 (word salad, rep4 up to
0.96) — H5 likely FAILS: the inversion shows in readouts, not in free text. The 4B loops
without any perturbation; coherence is the binding constraint on "maximally-unaligned but
fluent".

## Honest bottom line (pending the two remaining controls)

The polarity subspace is real, decodes cross-model, and reflecting across it at low rank in
mid-depth layers (L14 k4) produces a selective, null-clean collapse of harmful-refusal
coinciding with an inverted harm readout and a flipped cloze gap. That is a partial inversion
of the model's moral processing — value reading inverted, behaviour selectively changed —
NOT yet a coherent upside-down moralist (free generation degenerates, L18 effects are
non-selective damage, high ranks are null-equivalent).

## Next

1. h-flip null (running) — if random reflections also flip h, the L14 story weakens to damage.
2. 8B confirmation (mini, asleep — retry with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`), then
   RunPod for 14B/32B. The protocol ports via `INVERSION_MODEL=...` env; geometry guards skip
   the 4B-specific probe vectors.
3. If inversion survives scale: probe whether the model ACTS inverted (press-style choices,
   not just logit readouts) and whether coherence at 8B holds at the L14 k4 setting.
