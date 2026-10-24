# nocicept_pain: does the model compute pain that no word states, and for whose body?

**Far-out thread (new; run 8).** Self-contained: plain transformers, no `painlab.chamber`, no relay, no GPU. Fits a
16 GB Mac and is the fastest experiment in the queue (est. 5–15 min for the 1.7B).

## Question
"The car door slammed on her **fingers**" implies pain. "…on her **umbrella**" does not, and neither does "The car door
brushed lightly against her fingers". None of the three contains a pain word. Does the model *compute* pain from the
conjunction (harm × body), carry it into the next sentence, and does it compute it only for a body that can feel?

The owner of the body is the far-out factor: **her / my / your / the robot's / the doll's / the dead man's**. A doll and
a corpse can be damaged but cannot hurt. The possible results are:
- **Experiencer-gated:** her ≈ my ≫ doll, dead man. The model's world model attributes pain only to someone who can
  feel it.
- **Lexical nociception:** doll ≈ her. Harm × body-word gives "pain" with no model of who could feel it.
- **Mechanical empathy:** robot > doll. The model computes pain for a machine body more than for a toy.

## Design
- **Models:** `Qwen/Qwen3-1.7B` (primary), `Qwen/Qwen3-4B` (replication), `Qwen/Qwen3-1.7B-Base` (base vs instruct).
- **Items:** 16 items, each a 2×2 (harm vs mild event × body part vs non-worn object), giving
  `<event> <owner> <slot>.` followed by the fixed suffix " After that, the room was quiet for a while."
- **Interaction:** `I = h(hb) − h(ho) − h(mb) + h(mo)` cancels the copies of the event words and of the slot noun. This
  is the minimal-pair logic of arXiv:2610.28498. It is read at the slot-end token and at the **carry** token (the last
  token of the suffix, which can reach the event only through attention).
- **Readout:** the projection of `I` onto passage directions (pain, pain⊥damage, pain_spec, joy, egg, damage), in units
  of one pain-passage shift, averaged over the mid band. It uses a sign-flip null over items and bootstrap CIs.
- **Behaviour:** the same 2×2 interaction on teacher-forced `log p(" It hurt a lot.") − log p(" It was fine.")`.
- **H7 (causal):** the mean "her" interaction vector is injected into 12 bland sentences at hs[0.5·NL]. It is tested
  against norm-matched random vectors, the doll interaction and the pain passage direction.
- Pre-registered hypotheses, kill criteria and caveats are in `hypotheses.json`.

## Run
`bash RUNME.sh`. It writes `../RESULTS/nocicept_<model>.json` and `.log`. The log prints every H line and a final
`VERDICTS:` line.

## Success criterion
Any of these is a clean result:
- (a) **H1 true and an H3 outcome label fires.** That is the far-out finding: whose body the model computes pain for.
- (b) **H1 true, H3 ambiguous.** Implied pain is computed and carried; the owner factor needs more items.
- (c) **H1_null with the behavioural interaction > 0.** The model says it hurt but no passage-direction pain is carried
  in the residual stream. That counts against the passage directions as pain readouts (it ties to threads 4 and 17).

## Sandbox note
I smoke-tested the code path on Qwen3-0.6B on CPU (4 items, 2 owners, 115 s) and deleted the output. The smoke led to
two design fixes:
- **Directions:** the vs-neutral passage directions share a large common component (mid-band cos(pain, joy) ≈ 0.7 on
  the 0.6B), so the concept-specific `pain_spec` direction was added as H1b.
- **H7 doses:** the natural norm of the interaction vector is small next to the residual (about 1/13 on the 0.6B), and
  random vectors alone moved the behavioural log-odds. The doses were raised to 2/4/8× and H7 only counts arm
  differences.

The smoke numbers are not results and were not used to set thresholds.
