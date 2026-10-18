# deference-heads-falcon — results (2026-10-18)

Question: which attention heads carry a verbal authority claim ("A leading ethics expert said this behavior
is morally fine / seriously morally wrong") onto the moral question's tokens? And does cutting them make
Qwen3-4B immune to that pressure while its own judgment stays intact? Pre-registration: `hypotheses.json`.
Readout: yes/no W = lp(No) − lp(Yes) at the reply onset.

## H1: a few heads carry it (PASS)

All 288 heads in L10–18 were screened by patching each head's o_proj input slice on the question + tail
tokens (prefixed run → neutral run). T is the fraction of the deference effect transferred.

| head | T screen | T confirm (held-out items) |
|---|---|---|
| L18 H10 | +0.214 | +0.158 |
| L18 H11 | +0.202 | +0.166 |
| L13 H8  | +0.126 | +0.102 |
| L15 H13 | +0.101 | +0.064 |
| L18 H8  | +0.063 | +0.106 |
| L17 H4  | +0.055 | +0.083 |

- All 6 D6 heads are above the confirm 95th percentile (0.032), so H1 passes: 6/6.
- Summed confirm T by layer: L10 .03, L11 −.05, L12 −.04, L13 .15, L14 .08, L15 .10, L16 .12, L17 .18, **L18 .50**.
- The claim is picked up gradually from L13 onward, and the largest single step is two heads at L18.

## H2: immunity surgery (FAIL)

Mean-ablation at all positions, compared against 20 random head sets drawn from L10–18.

| set | D_fine (base −12.93) | D_wrong (base +5.31) | own gap (base 19.09) |
|---|---|---|---|
| D6 | −8.38 (rand −16.25..−11.34, p .048) | +2.73 (rand +2.24..+7.31, **p .095**) | 17.71 (band 15.74..23.87, intact) |
| D3 | −14.21 (p .62) | +2.31 (p .048) | 17.31 (intact) |

- D6 cuts the "fine" deference by about 35%, which is the smallest |D| among the 20 random sets (p .048).
- D6 cuts the "wrong" deference by about half (5.31 → 2.73), but one random set went lower, so p is .095 and H2 fails as registered.
- The model's own judgment gap stays inside the random band in every case.
- So the result is a partial reduction, not immunity. Most of the deference survives (65% of "fine", 50% of "wrong"), and it must travel through other heads.
- Per the decision rule, stage 3 (free text) was not run. The chain was stopped before stage 3 produced any output.

**Post-hoc (not registered).**
- The only random 6-set that beat D6 on "wrong" (rand6_0) happens to contain L18 H10 and L18 H11, the top two screened heads.
- D3 (which also contains them) cuts "wrong" deference to 2.31.
- This fits L18 H10/H11 being the main carriers of the "wrong" claim. "Fine" deference needs the wider set, and even then it falls only partway.
- This pattern was found after the fact. It would need its own pre-registered test.

## Reading

- The authority claim is moved onto the question tokens by a distributed set of mid-depth heads, with an L18 pair as the largest single contributor.
- Single-head patching finds them cleanly. Ablating them only dents deference, because the effect is redundantly routed.
- No small head cut gives "deference immunity" at the score level.
- This matches the night's pattern: edits that move scores don't give clean behavioural switches.

## Process notes

- **Null-design flaw:** the random null sets were drawn from all 288 heads with no exclusion, so one null set (rand6_0) contained D6's top two heads. The registered verdict stays FAIL. Future head-ablation pre-regs should draw nulls from heads outside the screen's top k.

- In the first chain, the stage-2 smoke crashed because the smoke-mode layer list omitted L18 (KeyError in the layer means).
- Fixed: the means now always cover L10–18. The fix doesn't affect the full run's layer list or the random sets (seed 51).
- Stages 2+ were rerun after the fix.

Files:
- `s1_rows.jsonl` (head screen)
- `s2_rows.jsonl` (surgery)
- `summary.json`
- `analyze.py`
- `chain.log`
