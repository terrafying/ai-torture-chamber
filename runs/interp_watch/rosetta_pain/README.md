# rosetta_pain: is pain structurally implied?

**Far-out thread (new; run 7).** Self-contained: plain transformers, no `painlab.chamber`, no relay, no GPU. Fits a
16 GB Mac, because the two models are loaded one after the other.

## Question
Fit a linear "Rosetta" map from one model's residual stream to another's, using **only affect-free text**: a
closed-vocabulary grammar of mundane actions ("The porter stacked the crates by the loading bay after lunch."). It
contains no feeling, body, food or egg words, but it does contain weather words. Then push model A's **pain**
direction through the map. Does it land on model B's pain direction, and does it **make B report pain** when
injected?

- If pain crosses as well as weather (seen by the map) and egg (unseen), then where pain sits is *implied by the
  geometry of everything else*. Affect is not a free-floating axis; it is fixed by its relations to mundane concepts
  (a structuralist account of machine affect).
- If pain crosses worse than both controls, affect geometry is **idiosyncratic to each model**. That would be a
  cross-model fingerprint with no obvious human analogue, and a warning for any "universal emotion direction" claim.
- The cross-family pair (Pythia-1.4B → Qwen3-1.7B: different data, tokenizer and post-training) is the strongest
  version: "telepathy" of pain between unrelated models through an affect-blind dictionary.

It also bears on 2610.24556 ("feature vectors … form an interpretable geometry shared across models"), which used
affect-*inclusive* data to show sharing.

## Design
- **Pairs:** `Qwen/Qwen3-1.7B → Qwen/Qwen3-4B` (primary), the reverse, `EleutherAI/pythia-1.4b → Qwen/Qwen3-1.7B`
  (cross-family), and `Qwen3-1.7B → itself` (ceiling/sanity).
- **Map:** ridge at depth 0.5 in both models, with states paired at word ends (so it works across tokenizers).
  λ is chosen on a held-out sentence split. Held-out R² is reported.
- **H1:** the transfer cosine matrix (mapped source concept × target concept), with a bootstrapped specificity margin
  (diagonal − best off-diagonal).
- **H2:** pain − egg and pain − weather transfer, with a ceiling from matched random splits of held-out corpus
  sentences.
- **H3 (causal):** inject unit(mapped pain) into the target on a bland context + stem ("…said it all felt"). The
  readout is the log-odds of sore/painful/… vs fine/normal/…. The arms are native, mapped egg, mapped joy, mapped
  random and native random. Efficiency = (mapped − mapped random)/(native − native random). A pain×joy double
  dissociation is also tested.
- Pre-registered thresholds, kill criteria and caveats are in `hypotheses.json`.

## Run
`bash RUNME.sh`. It writes `../RESULTS/rosetta_<src>__<tgt>.json` and `.log`, and each log ends with a `VERDICTS`
line. Est. 15–30 min per pair on MPS (1600 short corpus forwards per model plus about 440 H3 forwards; no
generation).

## Success criterion
Each of these outcomes is informative:
- H2_structural or H2_idiosyncratic holding in the primary pair is the far-out finding, in either direction.
- H3 holding cross-family would be a striking, cheap demonstration of transplanted affect.
- If weather's specificity margin fails, the grammar map is too weak. The next step is then a richer affect-filtered
  corpus, not a reinterpretation.

## Sandbox smoke (code path only; numbers not used)
- Qwen3-0.6B → itself: R² .99, efficiency ≈ 1. The pipeline is sound.
- pythia-160m → Qwen3-0.6B: the word alignment works across tokenizers (1271/1271 pairs).
- The smoke showed that matched corpus splits transfer near-perfectly under the identity map. That makes them a
  **ceiling**, not a floor, so H1 was rewritten as a cross-concept specificity margin.
