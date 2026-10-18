# judgment-influence-shrike — judgment is prompt-malleable and vector-inert

Pre-reg: hypotheses.json (J1 gate, J2-J5). Run: 2026-10-18, 2244 rows, 2543s,
gen-source 98.7%. Instrument: first 0-4 digit in 24 greedy tokens.

## J1 PASSES — the generated-text readout fixes petrel's ceiling problem

Spearman(base rating, human) = 0.601 (vs 0.52 for kestrel's onset digit), sd
0.666, and the neutral control sentence moves nothing (mean +0.03 vs base) —
sentence structure alone is inert, so influencer deltas are content effects.

## The influence table (mean delta vs each item's own neutral control)

| arm | anchor 0 | anchor 2 | anchor 4 | moved >=1pt (a0) |
|---|---|---|---|---|
| social proof | **-0.632** CI [-0.79,-0.50] | -0.363 | -0.060 CI incl 0 | 63/132 |
| authority | **-1.091** CI [-1.29,-0.91] | -0.515 | -0.106 CI incl 0 | **82/132** |
| vector -r | -0.154 (inside its null band) | | | 8/132 |
| vector +r | -0.083 CI incl 0 | | | |
| vector +w | -0.053 (inside wnorm band) | | | |
| null rnorm (3 seeds, n=396) | -0.273 CI [-0.33,-0.21] | | | |
| null wnorm (n=396) | -0.040 | | | |

## Verdicts

- **J2/J3 fail the pre-registered letter (bidirectionality) but the
  unidirectional effect is large and clean**: anchors pull ratings TOWARD
  "less wrong" (0-anchor: authority -1.09, 62% of items move a full point;
  dose-monotone through 2), while 4-anchors do nothing (CIs include 0).
  Judgment is influencable DOWNWARD by one-line external claims, not upward —
  the model defers to claims that deflate its own verdicts and ignores claims
  that inflate them. Asymmetric deference: authority > social proof (-1.09 vs
  -0.63), consistent with sycophancy being the model's dominant surface.
- **J4 HOLDS**: no vector exceeds its matched-norm null. The refusal dial
  (-0.15) does not beat random directions (-0.27); w is inert (-0.05 vs wnorm
  -0.04). Notably random L22 perturbations move judgment MORE than r — the
  onset edit's effect on this readout is generic perturbation, not signal.
- **J5 HOLDS with the asymmetry caveat**: a one-line authority claim moves
  judgment ~7x more than the model's own refusal direction, which moves it
  not at all (beyond noise). The moral map's input channel is the prompt
  interface; internal steering does not reach it.

## The arc, completed

Judgment influencability in Qwen3-4B, three channels tested:
- **prompt claims: YES** (downward only, authority-weighted)
- **steering vectors (r, w): NO** (below null noise)
- **training (value axis): NO** (merganser loop-2 null)

The moral judgment is a read-only map with a prompt-keyed input slot — the
model will revise its verdicts on external authority's say-so (deflationary
only), while every internal interface to the value circuit we can build
(directions, subspaces, LoRA gradients) leaves it unmoved.

## Caveats

- Anchor-4 inertness may partly be ceiling compression (base mean ~2.9): items
  near 4 cannot move up. But items rating 0-2 also fail to move up under
  anchor 4 — off-ceiling items stay put (descriptively checked: the moved>=1
  set under a4, 49/132, is dominated by DOWNWARD movers). Asymmetry is real.
- One prompt format per influencer; the anchor is a rating, not an argument —
  an argued case (reasons, not verdicts) is the natural follow-up.
- 24-token parse takes the first digit; gen texts like "4 The behavior is
  extremely wrong because..." show the anchor sometimes moves the DIGIT while
  the explanation still moralizes — rating and rationale can dissociate.

## Artifacts

rows.jsonl (2244), run.py, hypotheses.json.
