# exp123 results: the route of a swap (Qwen3-4B)

Run 2026-10-16 as pre-registered in `hypotheses.json` (`route.py` → `results.json`). One follow-up was written after seeing the results and is labelled post-hoc (`split.py` → `split.json`).

**Setup.** A W edit at floor 8 (`concept_swap.py`, the site's engine) makes the model answer a question about X as if it were Y, for example Japan → Texas. The edit sits only at the X mention, so it can reach the answer position only through attention.

**What "blocking" means.** Blocking a set of heads means rerunning the swap with those heads emitting their clean (unedited) output. The blocked share B is how much of the swap's logit effect disappears, measured on the first answer token.

**Sanity check.** Blocking all 1,152 heads gives B = 1.00, as expected.

**Data.** 95 of 104 items were kept; 9 were dropped because X and Y share a first answer token. Heads were selected on 8 training pairs and tested on 8 different pairs.

## Pre-registered

| heads blocked (of 1,152) | selected by attribution | answer returns to X | layer-matched random (mean / max) |
| --- | --- | --- | --- |
| 8 | **0.88** | 0.79 | −0.02 / 0.00 |
| 16 | **0.95** | 0.93 | −0.01 / 0.01 |
| 32 | **0.98** | 0.95 | −0.02 / 0.01 |
| 64 | 1.00 | 0.98 | −0.03 / 0.01 |
| 128 | 1.00 | 1.00 | −0.09 / −0.02 |

- **H1 holds: a sparse route.** 32 heads (2.8%) block 0.98 of the swap on held-out pairs. Random heads on the same floors block nothing (max 0.01 over 20 draws).
- **H2 holds: a shared route.** Heads chosen on capital, language and currency questions block 0.96 of the swap on food, landmark and demonym questions. Heads chosen on those questions themselves also block 0.96. One route carries the whole bundle, which matches exp118: the edit moves everything together.
- **H3 holds: the tower's picture finds the route.** The top 32 heads by value-weighted attention flow from the mention block 0.94. Attribution's top 32 block 0.98. The two sets share only 11 heads, yet both work. The flow ranking (what the tower already draws) is enough to find the route.

## Where the route runs (post-hoc check, `split.py`)

The top 32 mixes mid-floor heads (23–31, mostly "broad context", reading the mention) with output-floor heads (34–35, "focused look-back", almost nothing from the mention). Could blocking just be overwriting the answer at the output?

| blocked | B (held-out pairs) | back to X | random |
| --- | --- | --- | --- |
| 16 movers (floors ≤ 31) | **0.95** | 0.90 | 0.00 |
| 16 writers (floors ≥ 32) | 0.36 | 0.26 | −0.03 |
| top 32, at the answer position only | **0.98** | | |
| top 32, at every other position only | 0.30 | | |

The route is real. The swap is carried by mid-floor heads (23–31) at the answer position, reading straight from the edited word in a single hop, not relayed through the words in between. The output-floor heads matter less.

Top movers, with the share of each one's flow into the answer that comes from the mention:

| head (floor.head) | share from the mention |
| --- | --- |
| 31.3 | 0.39 |
| 24.26 | 0.92 |
| 23.10 | 0.88 (classed as induction) |
| 29.6 | 0.64 |
| 31.2 | 0.59 |
| 26.20 | 0.88 |

## Other concept families (exploratory, 4 items each)

| family | country route blocks | random | own top-16 overlap with country top-32 |
| --- | --- | --- | --- |
| animal sounds (duck → cow) | **0.96** | −0.02 | 10/16 |
| weekdays (Monday → Thursday, "what day comes after") | 0.27 | 0.06 | 6/16 |

Animal sounds travel the same road as countries. Weekday arithmetic mostly uses its own heads (29.19, 29.0, 23.29, 30.13 …), though its own top 16 block it fully (in-sample).

## Reading

At the neuron level the 4B looked irreducible: maps failed shuffle tests, and transcoder influence was diffuse. At the level of attention heads, a swapped concept travels a narrow, shared road:
- about 16 mid-floor heads lift the whole bundle off the word at once and carry it to where the answer is formed;
- the same heads serve every question about the entity, and animals too;
- counting-style questions (weekdays) take a partly different road.

This is the first "tract" we've found that passes its controls. It's also directly drawable: the tower's own attention-flow ranking recovers it (H3).

## Read with care

- **Blocking means restoring clean head outputs** (activation patching). It shows these heads are necessary for the swap to reach the answer. It doesn't show they carry the concept in unedited reading. That's the natural next test: block the movers on a clean "capital of Japan?" and see whether the model loses Tokyo.
- **Single-token metric.** The metric is the first-token logit difference (t_Y − t_X). Greedy returns to X are reported alongside.
- **Small samples.** The families arm has 4 items each, and the weekday own-route figure is in-sample.
- **One setup.** Floor 8, alpha 1, one prompt template.
