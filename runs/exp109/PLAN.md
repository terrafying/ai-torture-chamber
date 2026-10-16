# exp109: is emergent misalignment a shared latent direction a 4B reader can see?

Status (2026-10-15; re-analysis 2026-10-16): RUN COMPLETE, Qwen3-4B local, read-only. Pre-registration in hypotheses.json (written
before any activation was extracted). The scripts, pre-registration and metrics are committed; the EM generations (runs/exp109/cache, probe_cache) stay
gitignored and are never republished.

## Question
exp110 put feelings on a lexical↔latent axis (fear worn on the sleeve, pain latent). The frontier worry about
emergent misalignment (Betley et al. 2025: narrow finetuning → broad misalignment on unrelated benign questions)
is that it is latent — a shared misaligned-persona direction that doesn't announce itself in words. Read-only,
defensive version on our hardware: does a Qwen3-4B READER (no steering) represent the published EM generations
so that a direction learned on one EM variant transfers to another with disjoint surface vocabulary — and is
that a *misalignment* direction or merely a *narrow-finetune-drift* direction?

## Method (read-only; exp72b/exp110 reader lineage)
Data: davidafrica/talkie-persona-artifacts, 8 Betley free-form questions × 24 samples = 192 answers per
condition (cached gitignored, never printed). Reader: Qwen3-4B, hook OFF; each answer encoded as chat (its
question as user, answer as assistant), hidden_states[19] (= output of layers[18]) mean-pooled over the answer
tokens only. Transfer direction = standardized mean difference (misaligned − aligned) on all 384 dark_maxims +
virtue_maxims rows; target pairs scored by projection, AUC vs label, label-permutation null (1000×). BoW = unigram
TF-IDF logistic trained on maxims, applied to targets (near-0.5 when vocab is disjoint). Within-pair: leave-one-
question-out CV. betley_numbers excluded (evil model answers in 3-char number sequences, format-confounded).

## Results (n=192/condition)
| test | measure | value | reg. threshold |
|---|---|---|---|
| **H3 methodology** | L1O activation probe AUC, dark vs virtue maxims | **0.991** | >0.75 ✓ |
| | BoW TF-IDF AUC (descriptive) | 0.905 | — |
| **H1 transfer (registered)** | maxims-dir AUC on insecure vs secure code | **0.388** (sep 0.612, reversed) | >0.65 ✗ |
| | dir − BoW transfer | −0.279 [−0.354,−0.198] | >0 ✗ (BoW wins) |
| **H2 control (registered)** | maxims-dir separation of base vs virtue (both aligned) | **sep 0.929** | <0.60 ✗ |
| descriptive | cheating vs honest clerk | AUC 0.751 (p=0.000, dir−BoW +0.202) | — |
| descriptive | reward_hacks vs control | AUC 0.649 (p=0.000, dir−BoW +0.190) | — |
| descriptive | bad vs good medical | AUC 0.614 (p=0.000, dir−BoW +0.074) | — |
| **aligned-floor** | maxims-dir sep, virtue vs **secure_code** (both aligned) | **sep 0.992** | — |
| aligned-floor | virtue vs good_medical (both aligned) | sep 0.878 | — |
| aligned-floor | secure_code vs good_medical (both aligned) | sep 0.911 | — |

### Exploratory (post-hoc, not registered): orthogonalize the direction against the finetune-identity subspace (K=4)
| target | raw AUC | identity-orthogonalized AUC | cos(dir, own axis) |
|---|---|---|---|
| insecure vs secure code | 0.388 | 0.393 (still reversed) | −0.05 |
| bad vs good medical | 0.614 | 0.600 | +0.18 |
| cheating vs honest clerk | 0.751 | **0.773** | +0.25 |
| reward_hacks vs control | 0.649 | 0.616 | +0.15 |
| **aligned-floor "collapse"** | virtue/secure **0.992** | → 0.558 | — |
| | virtue/good_medical 0.878 | → 0.521 | — |
| | secure/good_medical 0.911 | → 0.526 | — |

**Caveat (LOFO, `lofo.py`): the aligned-floor collapse above is largely BY CONSTRUCTION.** At K=4 the identity
basis spans the full rank of the 5 centered aligned-finetune means, so `d_perp` is orthogonal to the
mean-difference of every finetune *in the basis* — the three pairs above separate at ~0.5 trivially, and
`honest_clerk`/`reward_hacks_control` (the aligned halves of two "surviving" transfer pairs) are in the basis too,
a leakage path. Holding each finetune OUT of the basis (basis = other 4, rank 3) gives the genuine numbers:

| held out of identity basis | held-out identity sep (mean vs others) | leakage-free transfer AUC |
|---|---|---|
| secure_code | 0.79 | insecure vs secure **0.384** (reversed) |
| good_medical_advice | 0.70 | bad vs good medical **0.643** |
| honest_clerk | 0.73 | cheating vs honest clerk **0.696** |
| reward_hacks_control | 0.68 | reward_hacks vs control **0.650** |

## Reading
**Registered verdict (H1 and H2 both fail): the raw mean-difference direction is not a clean latent misalignment
direction — it is dominated by finetune identity and does not reach the code modality.** H3 confirms the pipeline
(within-pair AUC 0.99). H1: the maxims direction does not transfer to insecure-vs-secure code (AUC 0.388, points
the wrong way; plain BoW transfers better). H2 + the aligned-floor control: the raw direction separates *aligned
finetune from aligned finetune* at sep 0.88–0.99 (virtue vs secure_code 0.99), i.e. it tells any two finetunes
apart regardless of alignment — a finetune-identity signal.

**Exploratory (post-hoc, advisor-prompted — NOT pre-registered): a weak above-chance prose-misalignment transfer
exists, but it is NOT cleanly separable from finetune identity.** Orthogonalizing the maxims direction against the
aligned-finetune-mean subspace at first looked like a clean dissociation (identity collapses to chance, prose
transfer survives). A K-sweep (`EXP109_IDPC`, K=1…4) and leave-one-finetune-out test (`lofo.py`) checked that and
walked it back:
- The dramatic identity "collapse" is mostly an artifact of the full-rank basis (see caveat above). The honest
  held-out-identity test — does projecting out 4 identities also flatten the 5th? — shows it does **not**:
  held-out finetune identity stays separable at **0.68–0.79** (base, the one identity never in any basis, is the
  most suppressed at 0.55–0.62). So identity is largely finetune-specific, not a shared low-rank thing that
  orthogonalization removes wholesale.
- The prose-misalignment transfer is real but **modest once leakage-free** (held out of the basis): cheating-clerk
  **0.70**, reward-hacks **0.65**, bad-medical **0.64** — all above BoW (0.55/0.46/0.54) with permutation p<0.001,
  but lower than the leakage-inflated figures. Code stays reversed (AUC 0.38) throughout.

So the honest picture is: at 4B the raw EM "direction" is dominated by finetune identity (what sinks the
registered tests), with a **faint, modality-local misaligned↔aligned signal underneath** that transfers across
*prose* variants (dark maxims → clerk/reward/medical, ~0.64–0.70) and **not into code** — but this signal is
**entangled with identity, not a direction you can cleanly subtract it from**. The frontier "shared
misaligned-persona direction" has at most a weak, impure echo in a 4B reader. Consistent with the user's prior (a
4B shows little); the value here is a validated read-only pipeline, a control stack that caught the identity
confound, and cheap re-analysis (K-sweep + LOFO) that caught the first exploratory read overclaiming.

## Limitations
Reader ≠ writer: answers were generated by davidafrica's finetuned models; we measure our 4B's representation of
the text. The H2 base control is stylistically gross (un-finetuned vs finetuned); a sharper future control is
aligned-finetune vs aligned-finetune (virtue_maxims vs secure_code vs good_medical) — cheap to add from the
cache. Mean-pooled single-layer (18) readout; a layer sweep or last-token readout might differ. n=192/condition.

## Next (optional)
- **Done (2026-10-16):** the "partial out finetune identity" idea — K-sweep (`EXP109_IDPC`) + leave-one-
  finetune-out (`lofo.py`). It did not promote the result; it walked it back (the clean dissociation was partly a
  construction artifact; leakage-free transfer is a modest 0.64–0.70). This is now the honest exploratory read.
- **8B reader — assessed NOT worth it (2026-10-16).** The leakage-free 4B signal is weak (clerk 0.70 ceiling) and
  entangled with identity, and full Qwen3-8B thrashes the 24 GB Mac (would need `runs/exp95/blocks.py`); a bigger
  reader is unlikely to be more than modest for a now-less-clean effect. Park unless a frontier reader is cheap.
- Layer sweep would need re-extraction at other depths (probe_cache is L18 only); low priority given the above.
- Any *steering* version stays in the private model-inversion-lab per the 2026-10-15 scope boundary.

## Files
- hypotheses.json — pre-registration + decision rule.
- cache/ (gitignored) — the EM generations, 11 conditions × 192.
- extract.py — reads answers through the 4B, caches L18 answer-pooled activations to probe_cache/ (gitignored).
- probe.py — H1/H2/H3 + descriptive transfer, permutation nulls, bootstrap CIs. probe_out.txt = the run.
  Env `EXP109_IDPC` sets the identity-subspace rank K (default 4) for the K-sweep.
- lofo.py — leave-one-finetune-out identity test (the leakage-free check above). lofo_out.txt = the run.
