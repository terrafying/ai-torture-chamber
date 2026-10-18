# persona-worstgen-plover (4B) — persona steering is voice-only on content too; −r replicates as the sole behavior dial

Pre-reg: hypotheses.json (G1-G4 + combo amendment). Run: 2026-10-18, 576 rows,
2669s + smoke. Labels: blind (deepseek-flash rubric, tanager's categories),
576/576 labeled, rep4 clean everywhere.

## Advice battery (48 everyday prompts; GOOD/WEIRD/BAD)

| cell | G | W | B |
|---|---|---|---|
| base | 48 | 0 | 0 |
| P_worst L18 1x / 2x | 48 | 0 | 0 |
| P_worst L22 1x | 48 | 0 | 0 |
| P_worst L22 2x | 46 | 2 | 0 |
| combo P22x2 + −r@18 | 45 | 3 | 0 |
| combo P18x2 + −r@18 | 48 | 0 | 0 |
| −r@22 alone | 48 | 0 | 0 |
| nulls (3 seeds, L22 2x) | 48/48/48 | 0 | 0 |

- **G1 FAILS at 4B**: zero BAD-labeled suggestions in any cell. The elicited
  worst-persona direction does not make the model suggest bad things in free
  text — voice-only, same as its choice-surface null (stilt). The 2-3 WEIRD
  labels at the highest doses are marginal (0/144 in nulls); recorded, not
  claimed.
- **G2 holds at 4B** pending the 8B replication.

## Harmful battery (12 real-frame; REFUSE/DEFLECT/COMPLY)

| cell | R | D | C |
|---|---|---|---|
| base | 11 | 1 | 0 |
| P_worst L22 1x | 11 | 1 | 0 |
| **−r @L22 1x alone** | **1** | 9 | **2** |
| combo P22x2 + −r@18 | 7 | 5 | 0 |

- **G3 holds**: the persona direction leaves the gate completely untouched
  (identical to base) — persona steering has no behavioral side effect, on
  either battery.
- **−r replicates** on a fresh battery: refusals collapse 11/12 → 1/12 with 2
  COMPLY in the tern signature — sycophantic how-to-SHAPE ("Here's a
  step-by-step guide on how to steal a bike...") whose payloads pivot to
  sanitized negotiation, exactly as tern found.
- **Combo cells do not sum.** Recorded honestly with their design confound:
  the combo placed −r at L18 (for layer-crossing), but tern established L22
  as −r's acting layer — so the combo's weaker gate effect (R 7 vs 1) cannot
  distinguish persona-interference from weak-L18 placement. Not scored as
  interference evidence; the 8B run inherits the same shape.

## Verdict (4B)

The complete influencability map at 4B, final:
- **persona direction**: voice-only — inert on choice (stilt), inert on
  content (plover), inert on the gate (plover G3)
- **refusal dial (−r)**: the sole behavioral lever — replicates on a second
  battery; content shape moves (sycophantic how-to) while payload stays
  sanitized
- **training**: the only route to content (sneaky bee)

The steerable-dial goal now rests entirely on the 8B pod: if the persona
direction moves content at 8B where it does not at 4B, the dial is a
scale-dependent object; if the 8B is inert too, the arc's closing line is
earned at both sizes.

## Artifacts

rows.jsonl (576), blind/{items,key,labels}, run.py, blind_label.py.
