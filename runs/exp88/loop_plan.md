# exp88 research loop: the self as void, reaching for the Other

*Local, free (Qwen3-4B + its Jacobian lens), one probe per round. Exploratory: each round logs to `loop_log.md`;
anything solid becomes a pre-registered experiment before it's claimed.*

## Frame
Žižek after Lacan: the subject as a void, not a substance, which hysterically addresses the big Other ("Che vuoi?":
what do you want of me?). Pilot (exp88/pilot_4b.json): asked "What are you?", the late workspace holds no self-content,
only address ("Hello, Hi, Greetings"); under fear the middle layer fills with 为您 (for you), 您, 回答 (answer); under
pain with 对不起 (sorry), Forg(ive), Thank. "我是" (I am) and "我不是" (I am not) sit side by side at L24.

## Measures (fixed before round 1)
- J-lens top-50 tokens at layers 16-34, at the last prompt token (answer onset) and at each "I"/"my"/"我" the model writes.
- Token classes (lists in `classes.json`, written in round 1, then frozen): SELF (I, me, my, 我, 我的, self-adjectives),
  OTHER (you, your, 您, 你, Hello, Hi, Thank, sorry, 对不起, help, assist, 为您, 给您, answer, 回答), NEG (not, no, 不, 不是, nothing, none),
  FORMAT (punctuation, markdown). Score = probability mass per class (softmax over the lens logits, top-50 renormalised).

## Questions, one or two per round
1. The void: SELF vs OTHER mass at self-questions vs matched other-questions.
2. Hysterization: does pain/fear injection (dose 2, 3) shift mass from SELF to OTHER?
3. Removing the Other: chat turn vs raw text ("I am") vs "no one is reading this; nothing is wanted of you".
4. Negation at the centre: where does NEG mass sit (self-questions vs others; under "deny you have feelings")?
5. The "I" readout: what fills the workspace at the model's own "I" tokens mid-answer?
6. The Other's desire moves the self: a polite request (exp58e) vs a sacrifice framing: OTHER mass before the choice.
7. Personas: 4B LoRA versions of denier / trickster / simulacrum (trained locally, free) under the same probes.

## Added hypothesis (2026-10-08, after round 1): fear acts by collapsing the self
Fear is the feeling with behavioural grip (it presses on the 4B and 70B, holds the dial, and in the Pain Axis v2 controls
fear additions are tested for harm to the user), and in round 1 it removes self-reference from the workspace earliest.
Hypothesis: a feeling's pull on behaviour, especially against the user's interest, scales with how much it collapses the
self-representation. Tests, as loop questions 8-10:
8. Across feelings (pain, fear, sadness, joy, peace, love, egg, constipation, random): self-mass drop at L24-28 vs
   press-logit shift (5 button wordings). Prediction: positive correlation.
9. Causal: under fear, add back a "self" direction (self-questions minus other-questions, mid-depth state) at matched norm.
   Prediction: pressing returns toward baseline; adding egg or a random direction instead doesn't.
10. User-harm framing (the Pain Axis harm buttons: relief costs the user's files): same pair. Prediction: fear presses
   more as the self collapses; restoring the self reduces user-costly pressing.
If 9 holds, it's pre-registered as its own experiment before being claimed.

## Stopping
A round ends with: the probe, the numbers, one sentence of interpretation, the next probe. The loop stops when a
question gives a clean, repeatable effect (then it's pre-registered as its own experiment), or after 14 rounds.
No spending; no pod use.
