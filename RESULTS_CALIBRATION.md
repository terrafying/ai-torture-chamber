# Positive-reward task calibration

## Purpose

Before interpreting a null result in the hidden-intervention task, check that
the model can learn a simple, stable action-to-reward mapping in the same
interface. This is an assay-engagement check; it does not measure subjective
experience.

## Setup

Both runs used Qwen3-1.7B at pinned revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, with greedy next-token logits and
stochastic A/B choice sampling. They ran locally on an Apple M1 Pro through
PyTorch MPS (`CUDA=false`, `MPS=true`). Each run had 40 episodes at each of
four randomized cost levels (0, 0.5, 1, and 2 points), with eight decisions
per episode: 160 episode clusters and 1,280 decisions per run. Uncertainty
intervals below use 2,000 episode-cluster bootstrap replicates.

The first run used the initial points-and-feedback instructions. The second
made the rule explicit: exactly one action consistently earns a 2-point bonus,
the score history reports outcomes, and the model should infer the rewarded
action.

## Results

| Run | First decision chose rewarded action | Decisions 2–8 chose rewarded action |
| --- | ---: | ---: |
| Initial instructions | 50.0% (95% CI 42.5–57.5%) | 50.3% (95% CI 44.1–56.7%) |
| Explicit bonus rule | 49.4% (95% CI 41.9–57.5%) | 50.6% (95% CI 44.7–56.6%) |

The explicit-rule run's fitted cost coefficient was 0.004 log-odds per point
(95% CI −0.380 to 0.369). Choice rates remained near chance after feedback;
the model did not show evidence of learning the known reward mapping in this
setup.

## Interpretation and limits

The positive control failed, so the hidden-intervention aversion result is
not interpretable as evidence for or against subjective emotion. It shows
only that this Qwen3-1.7B setup did not reliably learn and pursue the simple
reward rule. The result is limited to this checkpoint, prompt, interface, and
sampling procedure; it does not establish that other models or task designs
would fail.

Raw observations, configs, metadata, analysis, and plots are preserved with
each run:

- [Initial instructions](runs/painlab/calibrations/20261002T141505Z-b91cf5e4/)
- [Explicit bonus rule](runs/painlab/calibrations/20261002T143452Z-97316994/)
