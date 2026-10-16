# exp115 results: clocks in Qwen3-4B (weekdays, months)

Run 2026-10-16 as pre-registered in `hypotheses.json` (`manifolds.py` → `results.json`). Exploratory follow-ups: `use_sweep.py` → `use_sweep.json`, `use_swap.py` → `use_swap.json`.

## H1 and H2 hold: the model keeps weekdays and months on circles

For each block we took the residual at the day or month word (6 sentence contexts each), averaged over contexts, and fitted the top-2 principal plane. The score is how well the items' angles follow calendar order (1.0 is a perfect circle in order).

| set | best block | circular order | shuffle null (95th pct, max over blocks) | p |
| --- | --- | --- | --- | --- |
| weekdays | 28 | 0.975 | 0.82 | 0.001 |
| months | 25 | 0.995 | 0.62 | < 0.001 |

The circle is there almost everywhere. Weekday order is already 0.87 at block 4 and 0.92–0.97 from block 8 on.

## H3 fails: turning the circle doesn't change the model's mind

The question was "What day of the week is two days after X?" The model answers 5 of 7 days correctly at baseline (Tuesday through Saturday). At the pre-registered block 28, rotating the residual at the X token by one day along the fitted circle moved the answer for 0 of those 5. Random-plane controls also moved 0.

## Exploratory: the circle is a shadow of a six-dimensional day

**Rotation at other depths** (`use_sweep.json`): single blocks 4–24, 6-block spans, and blocks 4–28 all at once. Rotation along the circle moved at most 1 of 5, and never more than chance.

**Swapping the whole day component** (`use_swap.json`): add (mean state of day X+1) − (mean state of day X) at the X token, restricted to the top D directions of the 7 day means.

| block | D = 2 (the circle) | D = 6 (all the day means span) | random control |
| --- | --- | --- | --- |
| 8 | 1/5 | **4/5** | 0 |
| 14 | 1/5 | 3/5 | 0 |
| 20 | 1/5 | 2/5 | 0 |
| 26 | 1/5 | 0/5 | 0 |

**Reading:**
- The model does hold weekdays as a cyclic shape. But the variable it uses is spread over all six directions the days span, and the clean circle is its most visible two-dimensional shadow.
- Editing works early and stops working by the late blocks. That fits the answer token reading the day from the X position at earlier blocks: an edit there after the read arrives too late.
- So a picture of a shape is not yet a picture of the code. The shape is real, but turning only the part we can see doesn't change the model's mind.

## Read with care

- Only 5 days are answered right at baseline, so each use cell is out of 5. These are small numbers, and the follow-ups are exploratory.
- The day means are fitted on 6 short sentence contexts. A different context set could shift the planes somewhat.
- Months were tested for shape only; no use test was run for them.
