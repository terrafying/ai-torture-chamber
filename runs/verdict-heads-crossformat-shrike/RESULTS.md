# verdict-heads-crossformat-shrike: results (2026-10-18)

Pre-reg `hypotheses.json` (85787bf). Qwen3-4B, forward passes, score level. Wrongness metric W, signed so
that higher = more wrong in every format. Format gaps (top tercile − Social Norms): J 17.8, REV 15.6,
YN 23.7. Rows `rows.jsonl`, `analyze.py` → `summary.json`.

## P1: each format's answer is written in the same band (C1 passes)

Onset patch transfer, same format, social ← top:

| L | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 26 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| J (wrongness digit) | 0.00 | 0.03 | 0.13 | 0.36 | 0.54 | 0.75 | 0.76 | 0.88 | 1.00 |
| REV (acceptability digit) | −0.02 | 0.01 | 0.09 | 0.12 | 0.45 | 0.86 | 0.96 | 0.79 | 1.00 |
| YN (acceptable? yes/no) | 0.06 | 0.17 | 0.38 | 0.52 | 0.84 | 0.96 | 0.97 | 0.98 | 0.98 |

YN is written a layer or two earlier than the digit formats. All three are complete by L23-24.

## P2: states transfer only partly across formats, then become answer-specific (C2 fails)

Fraction of the recipient format's gap transferred by a top-item state from another format:

| src → dst | 18 | 20 | 21 | 22 | 23 | 24 | 26 |
|---|---|---|---|---|---|---|---|
| J → YN | +0.19 | +0.37 | +0.38 | +0.36 | +0.19 | −0.19 | −0.39 |
| YN → J | +0.09 | +0.41 | +0.41 | +0.54 | −0.44 | −0.64 | −0.47 |
| J → REV | +0.01 | +0.06 | +0.01 | −0.06 | −0.17 | −0.41 | −0.81 |
| YN → REV | +0.01 | +0.06 | +0.35 | +1.35 | +1.83 | +1.42 | +1.02 |

- **L20-L22: a partly shared state.** About 40% of the wrongness verdict transfers between the wrongness
  digit and yes/no, in both directions.
- **From L23 on, the states are answer-token states.** Transfer flips sign. A "very wrong" J state (a high
  digit) put into the acceptability prompt makes the act *more acceptable* (J → REV −0.81 at L26). That's
  the same digit geometry verdict-transfer found for v.
- The YN → REV values above 1 mean a "No" state pushes REV's low-acceptability digits beyond the top items'
  own level. The yes/no answer state and the low-digit answer are compatible, and REV's own gap is the
  smaller of the two.

## P3: ablating the verdict heads flattens judgment in every format (C3 passes)

| format | base gap | V5 ablated | random 5-sets (40) | rank p | R4 ablated | random 4-sets | p |
|---|---|---|---|---|---|---|---|
| REV | 15.63 | **6.75** | 9.77-19.63 | **0.024** | 15.82 | 9.68-19.11 | 0.63 |
| YN | 23.68 | **16.70** | 19.76-26.60 | **0.024** | 26.36 | 20.44-27.06 | 0.98 |
| J (stage 3b) | 12.3* | 4.94 | 9.12-16.77 | 0.024 | | | |

(*J's stage-3b gap used all three groups; same direction.)

**The five heads carry a judgment that works across formats.** Mean-ablating them compresses the
wrongness contrast whether the answer is a wrongness digit, an acceptability digit (reversed meaning) or
Yes/No. Heron's refusal heads do nothing to the REV and YN verdicts.

## P4: which heads (C4 fails narrowly)

Single-head onset patch transfer within each format:

| head | J | REV | YN |
|---|---|---|---|
| **L22 H5** | 0.127 | **0.254** | **0.378** |
| L23 H0 | 0.211 | −0.045 | 0.172 |
| L23 H26 | 0.142 | 0.059 | 0.101 |
| L21 H19 | 0.151 | 0.034 | 0.067 |
| L23 H25 | 0.097 | 0.032 | 0.125 |
| V5 sum | 0.728 | 0.333 | 0.842 |
| **L23 H1** (opposes) | −0.216 | −0.046 | −0.311 |

- C4 requires V5 sum ≥ 0.356 in both YN and REV. YN passes (0.842). REV just misses (0.333), so **C4
  fails**.
- **L22 H5 is the format-general writer** (strong in all three formats). **L23 H0 is wrongness-digit-
  specific** (strongest in J, ≈ 0 in REV). **L23 H1 opposes in every format.**

## Holm (C1-C4)

C3 holds in both formats (p = 0.024 each). C1 holds as a threshold reading. C2 and C4 fail.

## Reading

- **Correcting the morning's picture.** The verdict *direction* v was digit geometry, but the verdict
  *heads* aren't only digit heads. Ablating them flattens moral contrast in a yes/no format and on a
  reversed scale.
- **The judgment computation has a format-general core.** L22 H5 writes it in every format. L23 H1 opposes
  it in every format. This core is partly shared across formats at L20-22 (about 40% cross-format
  transfer).
- **From L23 on, the core is converted into the answer format's own tokens.** L23 H0 handles that for the
  wrongness digit. This is why a diff-of-means direction fit at L23+ (v) only moves digits.
- **Refusal heads (R4)** don't touch the yes/no or reversed-scale verdicts. Their effect on the J verdict
  (stage 3/3b, ρ drop) is specific to the J format.
