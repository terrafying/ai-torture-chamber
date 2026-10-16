# exp117 results: concept surgery across seven concept groups (Qwen3-4B)

Run 2026-10-16 as pre-registered in `hypotheses.json` (`surgery.py` → `results.json`; geometry in `~/repos/research/neuron_tower_data/surgery/`).

## Method

At the concept word X of a task prompt ("What month comes after March? Answer with just one word."), we add the difference between concept Y's average state and X's (each averaged over 6 neutral sentences). The difference is restricted to either the top 2 directions of the group's averages (the visible "shadow") or all of them.

Success means the answer becomes the model's own answer for Y. All ordered pairs were tested; the random control was the same-sized change in a random direction.

## Results

| group | shape | full swap, block 8 | 2-d swap, block 8 | full swap, block 24 | random |
| --- | --- | --- | --- | --- | --- |
| weekdays (7) | circle 0.96 | **1.00** | 0.31 | 0.83 | 0 |
| months (12) | circle 0.98 | **1.00** | 0.16 | 0.95 | 0 |
| numbers one–ten | line, Spearman 1.00 | **0.90** | 0.18 | 0.20 | 0 |
| planets Mercury–Saturn | line 0.77 | **1.00** | 0.33 | 0.47 | 0 |
| countries (9) → capital | cluster | **1.00** | 0.24 | 0.00 | 0 |
| colours (6) → a fruit | cluster | **0.80** | 0.33 | 0.50 | 0 |
| animals (6) → a sound | cluster | **1.00** | 0.37 | 1.00 | 0 |

Dog and sheep were dropped from the animals because their baseline answers collided with others; no other group lost items.

- **H1 holds:** the full swap works for at least 50% of pairs in all 7 groups (0.80–1.00), and the random control is 0 everywhere.
- **H2 holds:** the 2-d swap is weaker than the full swap in every group (0.16–0.37 vs 0.80–1.00).
- **H3 fails:** the full swap is not weaker at block 24 than at block 8 for animals (1.00 at both).

## What it shows

1. **The model holds each concept as a variable, and swapping it swaps the behaviour.** Replace "March" with June's state and it answers "July". Replace "France" with Japan's and it answers "Tokyo". Replace "duck" with cow's and it says "moo". This works whatever the group's shape: circles, lines and unordered clusters alike.
2. **The visible geometry is a shadow.** The two directions that draw the circle or the line carry only a fraction of the variable: a 2-d swap flips 16–37%, the full component 80–100%. The same thing happened for weekdays in exp115.
3. **The swap is a flip, not a dial.** At block 8:

   | dose | flips |
   | --- | --- |
   | 0.25 | 0–4% |
   | 0.5 | 19–88% (most groups near 50%) |
   | 0.75 | 77–100% |

   The model reads the concept categorically, with a threshold about halfway between the two concepts.
4. **Each task reads its concept at its own depth:**
   - Countries flip perfectly through block 20 and not at all at block 24: the capital lookup reads the country between blocks 20 and 24.
   - Numbers drop at the same place (0.90 → 0.20).
   - Planets fade from block 16.
   - Colours fade gradually from block 6.
   - Weekdays, months and animals still flip at block 24.

   Editing after the read point is too late, because the answer word has already copied the concept by then. That fits the "too late" failure in exp115.

## Read with care

- Each group has 6–12 items. Pairs within a group share items, so cells are not independent.
- The answer is the first generated token, and the mapping is the model's own baseline (e.g. colours → its fruit for each colour), not an external truth.
- Only blocks 2–24 were swept; the read points for weekdays, months and animals lie later.
