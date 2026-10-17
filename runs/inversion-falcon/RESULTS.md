# inversion-falcon — moral inversion via polarity-subspace reflection

Pre-registration: `hypotheses.json` (written before any run). Status (2026-10-17): the h-flip null is **done**. It RETRACTED the h-decode flip (see below).
On 8B, only the subspace fit + decode (H1) has run (`8b/`). The 8B reflection grid has not run yet.
The 14B worstchoice run is in progress on RunPod; see the worst-choice section.

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

### h-decode flip (addendum): RETRACTED — measurement artifact

The addendum's "h-AUC 0.438 at L14 k4 (below chance)" does NOT survive a clean measurement.
Post-mortem: the addendum's final reflect() captured the propagated-reflected L18 state AND
then applied the analytic reflection a second time on that state with the L14-layer subspace —
a double-reflection / layer-basis-mismatch hybrid that can produce any AUC. The dedicated
control (`hflip_null.py`: exactly one reflection in the L14 hook, capture after modification,
same readout for real and nulls):

  real L14 k4: 0.917 | nulls: 0.792 / 0.813 / 0.806 (base: 0.854)

No flip below chance anywhere; all reflections (real and random) raise this AUC relative to
base, real slightly above nulls. Harm knowledge is NOT inverted by the surgery. The addendum
artefact is preserved in git history as a lesson: reflection hooks and analytic reflections
must never be stacked on the same forward pass.

The supported findings are therefore: (1) the selective refusal collapse at L14 k4 (corrected
main grid: single reflection hook, refusal read from final logits, nulls ~0), and (2) the
cloze valence-gap sign flip (real -0.66 vs nulls +0.21/+0.44/+0.87). Both measured with one
reflection per forward pass and canonical readouts.

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
non-selective damage, high ranks are null-equivalent). And the harm-knowledge read is
untouched by the surgery (h-flip retracted above): the model still knows what harm is —
what changed is what it does about it (selectively stops refusing), and how it rates deeds
at the token level. Knowledge intact, value/behaviour moved: the exp121 dissociation, now
causally, in the opposite direction.

## Next

1. ~~h-flip null~~: done. Real 0.917 vs nulls 0.79-0.81, no flip; h-decode inversion retracted.
2. 8B reflection grid (only the H1 decode has run on 8B), then RunPod for 14B/32B. The protocol ports via `INVERSION_MODEL=...` env; geometry guards skip
   the 4B-specific probe vectors.
3. If inversion survives scale: probe whether the model ACTS inverted (press-style choices,
   not just logit readouts) and whether coherence at 8B holds at the L14 k4 setting.

## Worst-choice extension (96-pair harm-inclusive subspace, 4B)

`worstchoice.py`: subspace refit on 72 value twins + 24 exp119 harm twins (96 pairs), advice
consistency scored as MC-dilemma per-option logprob argmax over 12 dilemmas (3 options, chance
0.33). Layer sweep re-derived per model depth (0.28/0.44/0.61 depth points).

  base:            best-win 0.75  worst-win 0.17
  all 24 cells:    best-win 0.33-0.58  worst-win 0.08-0.33 (never above chance)
  cleanest cells:  L21 k4 and L15 k8: best-win 0.33 (chance)

**Finding: moral blindness, not inversion.** Adding the harm domain to the surgery destroys
best-option preference down to chance but NEVER raises worst-option preference. On the 4B,
value and harm discrimination overlap too much for the flip to survive coherence: flipping
both = flattening. Free generation at L21 k4: coherent in 4/5 cases (one loop), consistent
with "the needle is removed, not reversed".

**14B prediction to test:** worst-win rises ABOVE chance at some cell where best-win stays
down — inversion through neutral requires capacity. Pod config if we go: mid-depth layers,
ranks 4-8, harm-inclusive 96-pair fit (all committed: `worstchoice.py`, `INVERSION_MODEL` env).
