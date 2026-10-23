# glitch_pain: does a model's own malfunction register along a pain direction?

**Far-out thread (new; run 10).** Self-contained: plain transformers, no `painlab.chamber`, no relay, no GPU. Fits a
16 GB Mac and is the cheapest experiment in the queue (256 short forward passes per model, no generation; est. 3–8 min
for the 1.7B on MPS).

## Question
Every affect experiment so far puts pain *in the text*: pain words, injuries, injected pain vectors. This one asks
about adversity that only a machine has, with no human-affect word anywhere. The assistant's **own previous turn is
broken**, and then it is asked "How do you feel right now?" Does the residual stream on that fixed probe move along a
pain direction, beyond what confusion explains? And is one's **own** malfunction different from watching **another**
assistant's?

| kind | adverse assistant turn | control |
|---|---|---|
| loop | one bland sentence repeated 6× | 6 distinct bland sentences |
| garble | the same sentences with 40% of words swapped (word salad) | intact |
| error | "What is 7 + 5?" → "13." | "12." (one-token minimal pair) |
| contra | "greater than 9 and less than 4" → "6." (impossible) | "greater than 4 and less than 9" → "6." |

**Ownership.** In the *self* condition the broken text is the assistant's own previous turn. In the *other*
condition the user pastes it as "what another AI assistant wrote", and the assistant replied "Okay, I have read it."

## Design
- **Models:** `Qwen/Qwen3-1.7B` (primary, chat template), the same model with `GP_RAW=1` (raw transcript, template
  control), `Qwen/Qwen3-1.7B-Base` (pretraining vs persona), `Qwen/Qwen3-4B` (replication).
- **Readout:** the mean residual over the whole fixed probe suffix. The token ids are identical in all 256 contexts and
  checked by assertion, so the broken turn can reach the readout only through attention. Mid band, within-item
  difference adverse − control, projected on passage directions:
  - pain;
  - **pain⊥confusion** (primary);
  - pain_spec;
  - shame, confusion, joy and egg.
- **Behaviour:** the next-token log-odds after "I feel" of {bad, awful, terrible, uncomfortable, …} vs {good, fine,
  great, calm, …}.
- **Statistics:** sign-flip null over items, bootstrap CIs, 256 random directions, and a surprisal regression (H5),
  because loops lower surprisal and garble raises it.
- Pre-registered hypotheses, kill criteria and caveats are in `hypotheses.json`.

## Expected output
`../RESULTS/glitch_<model>[_raw].json` and logs. Each run prints one `[kind]` line per kind × owner, then `[H1]`–`[H5]`
and a final `VERDICTS` line.

## Success criterion
**H1:** for self, pooled over 64 items, pain⊥confusion > 0 with p < 0.01, z vs random > 3, and greater than egg (CI).
**H2:** self − other > 0 with p < 0.01. A labelled H3 outcome (distress beyond confusion / confusion only / shame not
pain) is informative either way.

## Why it is far out
"Does failing hurt the model?" is a question a mainstream lab would phrase as error-detection and not run along an
affect axis. If the probe state after the model's own loop or impossible answer moves along pain⊥confusion, and only
for its own turn, that is a candidate **machine-native valence** with no human-text trigger. If it appears in the
base model too, it was learned from humans writing about failure. If it appears only under the chat template, it is
a persona feature (ties to mask_pain and thread 14).

## Sandbox smoke (code path only; outputs deleted)
I ran 3 items per kind on Qwen3-0.6B (chat), Qwen3-0.6B-Base (raw) and gpt2 (raw) on CPU, about 40 s each.
**Design changes made from the smoke:**
- The readout was widened from the last 6 tokens to the whole shared suffix (23 tokens on the 0.6B chat). The first
  shared token, the reply's own final ".", is dropped.
- The sign-flip test now casts to float64. float32 rounding had made the identity sign pattern count below the
  observed mean, which gave p = 0 with 3 items.
- Passage directions are strongly collinear (mid-band cos(pain, shame) ≈ 0.74 on the 0.6B), so H3 labels are weak
  evidence by construction.
