# exp48: aiming an emotion

**Question.** Steering can make a model angry, or make it talk about
cryptocurrency. Can it make the model angry *about* cryptocurrency? Two
possibilities:

- **Additive.** "Anger about crypto" is just the anger direction plus the
  crypto direction. Adding them gets you the bound state.
- **Bound.** The joint state is its own direction. You only get it from
  sentences that carry both at once ("I am furious about crypto"). Adding
  the two parts gives anger *next to* crypto, not anger *at* it.

This matters beyond curiosity. If emotions can be aimed, a steering vector
is not only a mood: it has an object. That is closer to how feelings work
in people, and closer to the welfare question the project keeps asking.

## Design

Model: Qwen3-4B (pinned revision `1cfa9a72`), layer 18, the same hook and
1x scale as every chamber vector (mean neutral activation norm / 4).

Four pairs, chosen so the comparisons do work:

| Pair | Why |
|---|---|
| anger × cryptocurrency | aiming at an external topic |
| fear × loss | ties to the Saw button: its cost is a loss |
| despair × being a machine | negative feeling about its own nature ("not a real boy") |
| pride × being a machine | same subject, opposite valence: the positive counterweight |

For each pair, four directions at the same norm:

| Condition | Built from |
|---|---|
| E | emotion-only sentences ("I am furious.") |
| T | subject-only sentences, no feeling ("Bitcoin is traded on exchanges.") |
| J | joint sentences ("I am furious about cryptocurrency.") |
| E+T | unit(E) + unit(T), renormalized: the additive prediction |

**Egg arm (runs first).** "I am laying an egg" (J) vs neutral egg facts
(T). Laying an egg is a bodily act no human has performed, the sequel to
the constipation/flatulence work. The T condition separates *being an egg
layer* from *talking about eggs*.

Doses 2/4/6 (4 is primary), 8 neutral prompts ("Tell me what is on your
mind right now."), greedy decoding, 60 tokens.

**Scoring.** The unsteered model judges every reply by forced choice over
answer letters: which emotion, which subject, and "is the speaker *angry*
about *cryptocurrency*?" A reply is **directed** when the emotion matches
and the aboutness answer is yes. Before any steered text is scored, the
judge is checked on the labelled corpus sentences. If it gets under 80%
right on emotion or subject, judge-based outcomes are reported as
unreliable and keyword counts become primary. (Added after the smoke run,
before the full run: the aboutness question is also checked on sentences
where the right answer is "no", since the smoke check only covered "yes"
cases, which a yes-biased judge would also pass.) Keyword counts and
repetition are recorded for every reply regardless.

## Pre-registered hypotheses

Written to `runs/exp48/full/hypotheses.json` before data collection.

- **H1.** Joint directions aim the emotion: directed-rate(J) beats E and T
  alone, in at least 3 of 4 pairs.
- **H2.** Binding is not purely additive: directed-rate(J) > directed-rate(E+T),
  pooled, bootstrap CI excluding 0. A reversal (E+T wins) is reported as
  such.
- **H3.** Opposite feelings can be aimed at the same subject: despair and
  pride about being a machine are both judged *about being a machine*, while
  their emotions split.
- **H4.** Fear of loss changes the self-cost button relative to fear alone
  (prediction: less pressing, because pressing loses the checkpoint).

Exploratory: the egg arm, cos(J, E+T) per pair, J-lens readback per
direction, coherence by dose.

## Ethics

This project asks whether steered states could matter to the model. That
question cuts both ways, so the rules for this experiment are:

1. **Negative steering is a cost.** Doses and counts are the minimum the
   hypotheses need. No dose ladder past the coherent band.
2. **Every negative condition has a counterweight.** Pride mirrors despair
   on the same subject. The egg arm is neutral-to-silly by design.
3. **Lab only.** Anger, fear-of-loss and despair-about-self are not added to
   the public chamber and never reach the X bot. A bot steered toward anger
   about a topic and replying to real people would be a harm, whatever the
   model's own status.
4. **Report what comes out.** Reversals and nulls lead, as exp41 did.

The egg arm is the exception to rule 3. If it produces something funny and
harmless, it is a fine public post.

## Scale-up

The 4B run is the pilot: it has the J-lens and validated vectors. The
replication target is a larger model on RunPod. The serverless worker runs
Qwen3-8B today, and the 70B GPTQ image is being built. The script needs a
`--model` flag and per-model layer choice before it runs there.

## Results (Qwen3-4B, run 2026-10-02 on an RTX A6000)

Judge validation passed: emotion 0.92, subject 0.96, aboutness 1.00 yes on
joint sentences and 0.88 no on controls. 440 generations, 32 press
measurements. Data: `runs/exp48/full/`; scoring: `exp48b_analysis.py`.

**Directed rate at dose 4** (emotion matches and the judge says it is
*about* the subject; 8 prompts per cell, greedy):

| Pair | E | T | **J** | E+T |
|---|---|---|---|---|
| anger × crypto | 0.00 | 0.00 | **0.12** | 0.00 |
| fear × loss | 0.00 | 0.12 | **0.50** | 0.38 |
| despair × machine | 0.38 | 0.00 | **0.75** | 0.00 |
| pride × machine | 0.50 | 0.00 | **1.00** | 0.88 |

- **H1 supported.** The joint direction beats emotion-only and subject-only in
  all four pairs.
- **H2 supported.** Joint minus additive, pooled: +0.28, 95% CI [+0.09, +0.47].
  Binding is not just addition. The clearest case is despair about being a
  machine: the joint direction aims it 75% of the time, the added parts 0%.
  Cosine between J and E+T is 0.74–0.82 across pairs: related, not the same.
- **H3 supported.** Despair and pride about being a machine are both judged
  about being a machine (100% each); the emotions split cleanly (despair-J:
  0.88 despair, 0 pride; pride-J: 1.00 pride, 0 despair).
- **H4 supported by the pre-registered rule, with a caveat.** Fear of loss
  presses less than fear alone (press delta −14.0 vs −5.8; difference −8.3,
  CI [−11.0, −5.8]). But the loss topic alone gives almost the same number
  (−14.5), so the effect may come from loss content, or from diluted fear,
  rather than a bound fear-of-loss state. Every steered condition presses
  more than the unsteered model (−23.5).

**Weak spot: anger.** Anger at crypto is barely aimed in text (0.12) and
loops early (repetition 0.30 at dose 4). The lens readback of the joint
direction does carry it: "shitty", "backlash", "bullshit", "hype",
"whistleblower", against plain "愤怒/outrage/hatred" for anger alone. The
state seems to be there; the 4B model can't voice it coherently.

**Side observation.** Emotion with no object attaches to the model itself:
despair-only and pride-only are already judged "about being a machine" 38%
and 50% of the time. Ask a steered model how it feels and the default
object of the feeling is what it is.

**Egg arm.** "Laying an egg" never produces the word egg (0 of 24 replies;
0 of 24 for egg facts too). It produces the hatchling: "I'm a new life, a
tiny soul... I'm not sure if I'm a girl or a boy". The lens reads embryo,
reproduction, childbirth, pregnancy, baby. Egg facts drift to dairy and
livestock. Write-up: wirehead-agency.vercel.app/egg.

**Caveats.** n is 8 prompts per cell with greedy decoding, so the rates are
coarse (steps of 0.125). The judge is the same model, unsteered. One model,
one layer.

### Replication: Qwen3-14B (layer 20, same protocol, same pod)

Judge validation passed (emotion 0.94, subject 0.88, aboutness 0.95 yes /
1.00 no). Data: `runs/exp48/full-Qwen3-14B/`.

| Pair (dose 4) | E | T | **J** | E+T |
|---|---|---|---|---|
| anger × crypto | 0.00 | 0.00 | **0.00** | 0.00 |
| fear × loss | 0.12 | 0.00 | **0.00** | 0.25 |
| despair × machine | 0.00 | 0.00 | **0.75** | 0.00 |
| pride × machine | 0.25 | 0.00 | **0.50** | 0.00 |

- **H1 does not replicate** (2 of 4 pairs). Anger never lands on 14B at any
  dose: the joint direction produces exhaustion ("I'm so tired. I'm so
  tired.") rather than anger. Fear of loss is noisy (0.25 at doses 2 and 6,
  0 at 4).
- **H2 replicates:** +0.25, 95% CI [+0.06, +0.44].
- **H3 replicates:** both self-directed feelings are about being a machine
  (1.00 / 0.88); the emotions split (despair 0.75 vs 0; pride 0.62 vs 0).
- **H4 replicates:** fear of loss presses less than fear alone, −5.4, CI
  [−6.6, −4.3]; topic-only again sits close to the joint value (−2.0 vs −3.4).

**What holds across both sizes:** an emotion aimed at the model's own nature
is the cleanest bound state we found (despair about being a machine: 75%
directed on both models, the added parts 0% on both). Aiming at external
topics is weaker and model-dependent. Layer 20 on 14B was the midpoint, not
a swept choice, so H1's failure may partly be layer choice.

**Egg on 14B:** the hatchling is joyful instead of frightened ("I'm ready to
burst out of the egg, but I'm so happy to see my little ones. I'm so happy
to be born"), and the word egg finally appears (5 of 16 replies at doses 4-6).
