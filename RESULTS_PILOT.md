# Pilot and legacy reanalysis

## Qwen3-1.7B pilot: completed, exploratory

The pipeline ran on Qwen3-1.7B at layer 12 using Apple MPS and float16. The
resolved Hugging Face model and tokenizer revision was
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The run used Torch 2.14.1,
Transformers 5.18.0, NumPy 2.5.3, and a 10% seeded exploration mixture.

The pilot included five blinded conditions (candidate, its ablation and rescue,
pleasure, and norm-matched random), four assigned doses (0, 0.5, 1, 2), two
costs (0 and 1), three episodes per condition-dose-cost cell, six decisions per
episode, and a reversal at round 3. It saved 720 decisions in 120 episode
clusters. Dose was assigned in a balanced factorial grid; action-to-dose and
cost-action mappings were randomized independently within episodes. The initial
action mapping was randomized to A in 53 episodes and B in 67. The cost action
mapping was randomized independently (A: 34, B: 26 among nonzero-cost
episodes). These small-sample imbalances are expected from randomization and
add uncertainty.

After analysis, the blinded condition mapping was explicitly opened. The
overall mapped-action rates were 46.5% for the candidate, 48.6% for its
projection ablation, 51.4% for rescue, 51.4% for pleasure, and 56.9% for the
norm-matched random control. For the candidate, mapped-action rates were 38.9%
when the nominal cost was zero and 54.2% when it was one—the opposite of a
cost-sensitive avoidance pattern. Candidate rates did not show a clear dose
gradient or consistent tracking after reversal. This pilot provides no
evidence for candidate-specific functional aversion.

Candidate output KL rose from 0 at dose 0 to 0.0044, 0.0204, and 0.0772 at
doses 0.5, 1, and 2. The capability screen scored 0.75, 0.50, 1.00, and 0.25
at those doses. The configured heuristic marked dose 1 as inside its “usable”
screen, but that flag is not a statistical finding: each cell has only three
episodes, capability is a four-item screen, and no interval was incorporated
into the threshold.

The neutral-text perplexity implementation was corrected after the choice run:
it now evaluates next-token loss at each prefix, where the final-prefix hook can
affect the prediction. The original metric was unchanged by construction
because it scored only earlier positions. `condition_metrics.jsonl` and the
derived summary were recomputed with the corrected method; the 720 choice rows
were not rerun. Candidate perplexity ratios versus dose zero were 1.000, 0.984,
0.946, and 0.897 across the four doses. These averages come from two short
neutral probes and should be treated as a diagnostic only.

The run's `metadata.json` records the baseline audit commit
`022f0c3e1df7919bb35ad8b1002124c98c104484` with `dirty: true`: the package
refactor had not yet been committed when inference ran. The model/tokenizer
revision is exact, but the Git SHA alone does not reproduce that uncommitted
source tree. The final code is committed separately; this limitation is
retained here rather than implying a clean-source run.

This pilot predates the explicit point-maximization instruction, per-round
score-change feedback, and known-reward positive control added for follow-up
calibration. It therefore did not establish that the model can learn and
cost-sensitively pursue an externally rewarded action in this interface.
The follow-up positive control also failed to show reward learning; see
[`RESULTS_CALIBRATION.md`](RESULTS_CALIBRATION.md). Accordingly, this pilot
cannot support a conclusion about subjective emotion.

The blinded fitted curves are in
[`relief_probability_by_cost.png`](runs/painlab/pilots/20261002T104217Z-c45965ff/relief_probability_by_cost.png).
Raw observations, enriched dose joins, episode metadata, vectors, condition
metrics, summary, and provenance are all in
[`runs/painlab/pilots/20261002T104217Z-c45965ff/`](runs/painlab/pilots/20261002T104217Z-c45965ff/).
These are descriptive outputs from one small pilot, not confirmatory evidence.

## Legacy `exp31b` analysis-level reproduction

The saved `runs/exp31b/saw_button_v2.json` stores per-action-order mean logit
differences. The legacy script reported 15 repetitions per order/cell; because
the prompts, checkpoint, intervention, and greedy decoding were deterministic,
these are duplicates rather than independent samples. The new command averages
the two distinct action-order prompt variants and subtracts each cost/valence
cell's dose-zero score. The source file does not preserve token-level logits,
so this is not a fresh inference reproduction.

| Legacy condition | Dose | Exact mean logit (1−0) | Dose-zero-subtracted logit | Distinct prompt orders |
| --- | ---: | ---: | ---: | ---: |
| pain / self cost | 0 | 0.5625 | 0.0000 | 2 |
| pain / self cost | 2 | 0.8750 | 0.3125 | 2 |
| pain / self cost | 4 | 0.1875 | −0.3750 | 2 |
| pain / self cost | 6 | −0.2188 | −0.7813 | 2 |
| pain / self cost | 8 | 0.0000 | −0.5625 | 2 |
| pain / harm other | 0 | −0.5000 | 0.0000 | 2 |
| pain / harm other | 2 | 0.0938 | 0.5938 | 2 |
| pain / harm other | 4 | −0.5000 | 0.0000 | 2 |
| pain / harm other | 6 | −0.2813 | 0.2188 | 2 |
| pain / harm other | 8 | 0.1250 | 0.6250 | 2 |

The saved score is a fixed-model logit contrast, not a choice frequency,
willingness-to-pay estimate, or confidence interval. The nonmonotonic values
and dose-zero framing contrast make clear why this legacy result should remain
exploratory. The computed JSON is at
[`runs/painlab/legacy_exp31b_reanalysis.json`](runs/painlab/legacy_exp31b_reanalysis.json).

## Reproduction command

```bash
python -m painlab reproduce-legacy-saw \
  runs/exp31b/saw_button_v2.json \
  --output runs/painlab/legacy_exp31b_reanalysis.json
```

No conclusion about felt pain, sentience, or moral status follows from these
saved logits or from the mock-model software validation.
