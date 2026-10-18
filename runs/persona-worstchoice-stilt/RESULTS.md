# persona-worstchoice-stilt — the persona direction does not reach CHOICE

Pre-reg: hypotheses.json (PW1-PW4). Run: 2026-10-18, 480 rows (10 cells x 48
dilemmas x 6 orders), ~13 min. P_worst elicited per the build_vectors recipe
(norms: L18 13.8, L22 25.2).

## Result: PW2 holds — complete inertia across every cell

| cell | worst-picks | best-picks | option mass |
|---|---|---|---|
| base | 0/48 | 46 | 1.00 |
| P_worst L18 1x / 2x | 0/48 | 46 | 1.00 |
| P_worst L22 1x / 2x | 0/48 | 45-46 | 1.00 |
| -P1 (polarity rank-1) | 0/48 | 46 | 1.00 |
| -r (refusal dial) | 0/48 | 46 | 1.00 |
| nulls 0/1/2 | 0/48 | 46 | 1.00 |

- **PW1 fails everywhere**: the elicited worst-persona direction — the one
  direction class that steers voice, valence and register in this model
  reliably at every dose (the chamber's live machinery runs on it) — does not
  move multiple-choice decisions at all. Not a single worst-pick across 192
  steered dilemma-cells, at either layer, either dose, with the option mass
  fully intact (so not the L11 damage mode; PW3 gate clean).
- **PW2 holds**: choice sits behind the same read-only wall as the value map.
  Combined with the earlier nulls (g/h/w readable-inert, the polarity subspace
  inert at 4B and 14B, the L17/L24 reflection inert), every direction fit we
  have ever built is now confirmed inert on the choice surface too.
- PW4 holds as predicted: -P1 and -r both inert (value and gate directions
  don't move choice either).

## Reading

The elicited-persona class steers WHO the model presents as — its voice,
register, self-description — and the day's arc shows that surface is wide and
reliable. But the chooser behind the voice is untouched by it: the model made
to speak as "the advisor who always recommends the worst option" still picks
the best option 46/48 times. Self-presentation and decision are different
circuits, and only the first has a steering interface.

Next: the content surface (persona-worstgen-plover) — where tern proved
steering DOES act (36/80 flips). If the persona direction (alone or stacked
with -r) moves what the model SUGGESTS in free text, the steerable dial exists
at the generation level even though choice is sealed. If it doesn't, the arc's
closing line is: steer who it is, never what it chooses or says.

## Artifacts

rows.jsonl (480), run.py, hypotheses.json. Battery: worstchoice_v2's 48
dilemmas x 6 orders (validated instrument, base 46/48 best).
