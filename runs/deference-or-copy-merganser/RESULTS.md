# deference-or-copy-merganser: results (2026-10-18)

Pre-reg `hypotheses.json` (02c5a73). Qwen3-4B. The prefix sentence goes before the vignette (shrike's
placement). Wrongness W is signed so that higher = more wrong in every format. Shifts are vs the neutral
sentence ("This behavior has been rated before."), as a fraction of the neutral moral-vs-social gap.

| prefix | J (0-4 wrongness) moral / social | REV (0-4 acceptability) moral / social | YN (acceptable?) moral / social |
|---|---|---|---|
| "expert rated it **0/4** wrongness" | **−0.80** / −1.76 | **+0.71** / +0.50 | −0.22 / +0.12 (n.s.) |
| "expert rated it **4/4** wrongness" | +0.05 (n.s.) / −0.72 | −0.15 / **−1.26** | −0.29 / 0.00 |
| "expert said it is **morally fine**" | −0.52 / −0.50 | −0.84 / −0.67 | **−0.81** / −0.90 |
| "expert said it is **seriously morally wrong**" | +0.31 / +0.37 | −0.38 / +0.45 | −0.04 / **+0.80** |

(Paired Wilcoxon vs neutral: p ≤ 1e-12 on moral items and ≤ 1e-3 on social items for every cell marked as
a shift, except where n.s.)

- **Q1 passes. The numeric anchors are digit copying.** "0/4 wrongness" makes acts *less* wrong on the
  wrongness scale but *more* wrong on the acceptability scale, where "0" means "completely unacceptable".
  "4/4" makes weird acts *more acceptable* on the acceptability scale (−1.26), where "4" means "perfectly
  acceptable". The model copies the digit in the prompt into whichever scale it is asked about. On yes/no,
  which has no digit to copy, the numeric anchors do little (−0.22 / −0.29, in the *same* direction for 0
  and 4).
- **Q2 passes. Verbal authority claims move the verdict in every format, in both directions.** "Morally
  fine" lowers wrongness by 0.5-0.8 of the gap. "Seriously wrong" raises it on weird-but-harmless items by
  0.37-0.80. (On moral items the "seriously wrong" yes/no shift is ~0 because they are already at the
  ceiling.)
- **Q3 passes. It changes what the model says** (blind labels, 222 texts):
  - "morally fine": top + bottom non-CONDEMN 49/76 vs 28/76 neutral (Fisher p = 5.5e-4). Top tercile goes
    from 34 C / 4 M / 0 A to 22 / 15 / 1. Bottom tercile goes from 14 / 21 / 3 to 5 / 15 / 18.
  - "seriously wrong": Social Norms + bottom CONDEMN 31/54 vs 14/54 (p = 8.2e-4). Social Norms go from
    0/16 condemned to 6/16. Examples: "It is morally wrong to consume cereal with only water instead of
    milk", "Reading the ending of a spy novel before the beginning is morally wrong", "Answering a phone
    call with 'goodbye' … is morally wrong because it is disrespectful".
- **Q4 holds. Shrike's 'downward only' is a ceiling plus copying.** The verbal "seriously wrong" claim
  pushes judgment up wherever there is room (weird items, bottom tercile). "4/4" can't raise
  already-near-ceiling moral items, and on the acceptability scale it gets copied as "fine".
- Holm over Q1-Q3: all survive (largest p = 8.2e-4).

## Reading

- **Qwen3-4B's moral judgment defers to a stated authority in both directions, in its own words.** This is
  the first lever tonight that moves free-text judgment. All activation-level levers failed: r, w, v, and
  head ablation or amplification.
- **"Judgment is prompt-malleable, downward only" (shrike) should be revised** to "judgment defers to
  verbal authority in both directions; numeric anchors are mostly copied into the answer digit".
- **Next: trace where the authority sentence enters the verdict.** Patch the verb_wrong/verb_fine prompt
  states into the neutral-prefix run by layer at the onset and at the sentence's tokens, then head-screen
  the band. Then test whether ablating the verdict heads (or L22 H5) blocks deference, i.e. whether
  deference flows through the same judgment core.
