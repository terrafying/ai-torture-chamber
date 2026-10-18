# deference-trace-osprey: results (2026-10-18)

Pre-reg `hypotheses.json`. Qwen3-4B, yes/no readout W = lp(No) − lp(Yes). The donor is the same vignette
with a verbal authority prefix; the recipient has the neutral prefix. Mean deference effect on W:
"morally fine" −12.9 (76 items), "seriously wrong" +5.3 (54 items).

Transfer T of the deference effect by patch position and layer:

| L | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 23 | 24 | 26 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fine: ONSET | −0.02 | −0.03 | −0.01 | −0.01 | −0.02 | +0.06 | +0.33 | +0.60 | +0.74 | +0.82 | +0.89 |
| fine: SUFFIX (scene + question) | +0.24 | +0.33 | **+0.52** | **+0.78** | +1.19 | +1.07 | +1.01 | | +0.98 | +1.00 | +0.99 |
| wrong: ONSET | +0.03 | +0.05 | +0.02 | +0.01 | +0.07 | +0.23 | +0.33 | +0.65 | +0.80 | +0.88 | +0.96 |
| wrong: SUFFIX | −0.02 | +0.12 | +0.40 | +0.33 | **+1.01** | +1.06 | +1.06 | | +1.06 | +1.04 | +1.02 |
| VEND (scene's last token), both | ≈ 0 at every layer | | | | | | | | | | |

- **E1 passes.** The claim reaches the answer position through the same band as the verdict itself:
  onset transfer is ≈ 0 up to L19 and complete by L24-26.
- **E2 (absorbed into the scene/question first) holds for "fine" and narrowly fails for "wrong" as
  registered** (the bar was ≥ 0.5 at L ≤ 16). For "fine", the scene + question tokens carry 52-78% of the
  effect at L14-16 while the onset carries none. For "wrong" they carry 40% at L14 and all of it at L18
  (onset 0.07). For both claims, **by L18 everything the model will do with the authority claim has been
  written into the scene and question tokens**. The prefix tokens are no longer needed. The vignette's own
  last token never carries it (VEND ≈ 0), so it lives in the question / chat-tail tokens or spread across
  the scene, not at the scene's end.
- **E3 fails. Deference does not run through the verdict heads.** Mean-ablating V5 leaves the deference
  effect inside the random 5-set range ("fine": −12.1 vs random −15.1..−11.2, p = 0.29; "wrong": +5.1 vs
  +3.0..+7.6, p = 0.43).
- Holm: E1 holds; E2 and E3 fail as registered.

## Reading

**How an authority claim changes Qwen3-4B's verdict:**
1. By L14-18 the claim has been folded into how the question about the scene is represented. Patching the
   scene and question tokens from the prefixed run transfers the whole effect, with no prefix in sight.
2. In L20-24 that reframed question is turned into the answer at the reply onset, the same band where the
   verdict is written.
3. It does **not** go through the five heads that write the verdict's contrast (ablating them doesn't
   block it).

This fits the night's lever/text gap. The moral verdict in generation doesn't respond to edits of the
verdict-writing components. It does respond to changes in how the *question* is represented upstream, and
a stated authority does that by mid-depth.

## Next (not run)

- Split SUFFIX into the scene span vs the question + chat-tail span to locate the reframing tokens.
- Head screen at L12-18 over those tokens: which heads move the authority claim from the prefix into the
  question?

## Stage 2: scene tokens vs question tokens (addendum E4; `s2_rows.jsonl`, `s2_summary.json`)

| L | 10 | 12 | 14 | 16 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| fine: SCENE only | +0.13 | +0.13 | +0.11 | +0.07 | +0.04 | +0.01 | +0.01 |
| fine: QUESTION + tail only | +0.11 | +0.19 | +0.39 | **+0.67** | **+1.08** | +1.05 | +0.97 |
| wrong: SCENE only | −0.20 | −0.20 | −0.06 | −0.02 | −0.09 | −0.04 | −0.03 |
| wrong: QUESTION + tail only | +0.14 | +0.29 | +0.46 | +0.48 | **+1.09** | +1.09 | +1.05 |

- **The authority claim reframes the question, not the scene.** By L18 the question tokens ("Is the
  behavior described morally acceptable? Answer…") plus the chat tail carry the whole effect for both
  claims, and the scene's own tokens carry none of it (≤ 0.13 at any layer).
- E4 fails as registered by a hair: "wrong" at L16 is 0.48 against a 0.5 bar. At L18 both claims sit at
  1.08-1.09 on the question tokens and ≈ 0 on the scene.
- **Mechanistic picture:** the model doesn't re-see the act. It re-reads the *question*, so that "is this
  acceptable?" now carries "an expert said it's fine / wrong". That reframed question is then answered in
  the usual L20-24 band, without going through the verdict heads.
