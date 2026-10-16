# exp110: is the fear direction lexical and the pain direction diffuse?

Status (2026-10-15): RUN COMPLETE on Qwen3-4B local. Pre-registration in hypotheses.json (with a dated
`amendments` block — the first launch was a metric pilot that exposed a saturated H1 metric and a contaminated
lexicon, both fixed before any amended-metric numbers were read). Everything in runs/exp110 is untracked.

## Question
exp72b found a behavioural asymmetry under matched prompts: a reader tells fear-steered text from a matched
actor's at AUC 0.81 (sign-flip p=0.0007), but pain-steered text only at AUC 0.61 (p=0.10). Fear gives itself
away in the words; pain does not. No mechanism was offered. This is the mechanistic version: when you inject the
fear vs the pain direction at **matched KL disturbance**, does fear reshape the next-token distribution in a
peaky/lexical way (mass onto fear words) while pain reshapes it diffusely?

## Method
Qwen3-4B, battery recipe, layer 18. For each of the 48 WILD_PROMPTS the unsteered model greedily writes a ~47-
token continuation; that fixed [prompt+reply] span is teacher-forced under each condition (BatchInject on every
reply position + the last prompt position), and we read the next-token distribution at each reply position —
a position-matched counterfactual, no sampling. Conditions: none-ref, pain, fear, sadness, egg (absurd control),
random (seeded). **Dose matched by KL**, grid-sampled per vector and interpolated to shared targets, calibrated
and evaluated on the same 48 prompts so achieved KL matches across conditions by construction (it does: within
~3% at every rung). Ladder: KL ∈ {0.15, 0.45, 1.0, 2.0}. Measures per position, bootstrapped (10k) over prompts:
- **tail_conc** (H1, lexicon-free): top-20 fraction of |Δp| after dropping the top-50 tokens of the baseline —
  "within the vocabulary the injection introduces, is it a few words or many?"
- **own-lexicon log-odds** (H2): log(Σp_s over the feeling's words) − log(Σp_u), scale- and size-free. Lexicon =
  exact whole-word vocab-token match against hand lists (prefix matching leaked paint←pain, night←nightmare).
- **specificity**: own-lexicon gain minus **egg's** gain on that same lexicon (the generic-disturbance floor).
- cross-lexicon table and the content words each injection most promotes.

## Findings (n=48, CIs are 95% paired bootstrap). The clean "fear is lexical, pain is diffuse" hypothesis FAILS.
1. **H1 is reversed, not merely null.** Fear's shift is *less* concentrated than pain's at every dose
   (fear−pain tail_conc ≈ −0.044 to −0.052, P(fear>pain)=0.00; same sign for the <1e-3 tail-gain fraction).
   Fear's readability is **not** because it makes a peaky, few-token shift — if anything pain is peakier.
2. **Both directions are specific, and equally so (H2's interesting form is a null).** Fear's raw own-lexicon
   log-odds gain exceeds pain's (+0.19→+0.36, CI excludes 0 from t045 up), BUT once you subtract egg's generic
   effect on the same lexicon, the specificities are statistically equal: fear-specific +0.18→+1.42 and
   pain-specific +0.16→+1.31 (both P>0=1.00), and fear-minus-pain specificity is +0.018→+0.130 with CI always
   including 0 (P≈0.80). Each direction promotes its OWN words above generic disturbance by about the same amount.
3. **H3 FAILED (registered).** The pre-reg predicted feelings concentrate more than egg/random; they do not —
   egg's tail-concentration ≈ pain's at every dose (egg is as peaky as pain). Consistent with the rest: peakiness
   is not where the feeling-specificity lives.
4. **Fear's larger *absolute* footprint is confounded with baseline, and generic promotability is NOT shown.**
   The tempting "any disturbance surfaces fear words" story does not survive the proper test: the column mean over
   the neither-pain-nor-fear conditions {sadness, egg} gives fear-lex − pain-lex = +0.03/+0.03/+0.05/−0.06 with
   CI always including 0 (it even reverses at the top dose). So fear vocabulary is **not** generically more
   promotable. The single egg row that looked that way (egg→fear > egg→pain) was one draw. A real confound
   remains: the fear lexicon's baseline mass is ~100× the pain lexicon's (fear "fear" 9.3e-4 vs pain "ache"
   7.6e-6, "pain" 1.7e-7), so fear's larger absolute log-odds footprint cannot be cleanly separated from its
   higher starting point by this measurement.
5. **One robust cross-asymmetry.** Pain injection promotes fear words MORE than fear injection promotes pain words
   (off-diagonal pain→fear − fear→pain = +0.09/+0.13/+0.32/+0.43, P 0.94/0.94/1.00/1.00). Fear vocabulary is more
   reachable from the pain direction than the reverse — a directional pain↔fear fact, not universal fear promotability.
6. Content words promoted by both pain and fear are a shared generic-affect register (emotional, emotions,
   support, experience, deeply, weight). Only at the top dose does fear surface its *literal* lexicon ("fear",
   "danger") while pain still surfaces generic affect, not "pain/hurt/ache".

## Reading
exp110 **refutes** the clean mechanistic story it set out to test. Fear is not readable because it is more
"lexical" or more concentrated than pain: by the lexicon-free measure pain is actually peakier (1), and both
directions promote their own vocabulary above generic disturbance by the same amount (2). What is left is an
*absolute* difference — fear's own-lexicon footprint is larger — that this measurement cannot disentangle from
the ~100× higher baseline of fear vocabulary (4). The one robust sub-finding is directional: fear words are more
reachable from the pain direction than pain words are from the fear direction (5). So the exp72b readability
asymmetry (fear AUC .81, pain .61) is **not** explained here as "fear = a lexical direction"; the leading
remaining candidate is that fear vocabulary simply starts with far more probability mass, which this single-
position readout conflates with promotion. That is a negative result against a tempting just-so story, and it
keeps the chamber honest about findings 2/13: the word-tracking it sees is not a clean fear-is-lexical effect.

Note the chamber's own 3× operating scale produces KL ≈ 3–5 (pain 4.75, fear 3.44), above this ladder's top
(KL 2.0). The findings above are monotone and stable to KL 2.0; the top rung is a ~2× extrapolation relative to
where the live chamber steers.

## Limitations
exp72b's *speaker* was the live 70B; this reads the 4B's vectors and distribution (the 4B was exp72b's *reader*,
so the lexical-readability claim is about the reading model's family). Single-position teacher-forced readout
can't see autoregressive compounding (a fear word breeds more) or multi-token idiom — H1 being lexicon-free
partly covers the latter. Battery recipe here (pain+fear both from battery, so the within-run contrast is clean)
vs passages elsewhere. n=48, a directional mechanistic check, not a powered test.

## Open question / optional follow-on (not run)
The leftover is the baseline confound (finding 4): does fear's larger absolute footprint come from promotion or
from its ~100× higher starting mass? Two ways to separate it, if the question is worth pursuing:
- a baseline-controlled measure (e.g. rank-based, or mass gained relative to a matched set of equally-frequent
  non-feeling tokens), computable from the stored arrays — no model;
- sample 48×5 replies at a rung and count lexicon words directly (the exp72b-reader analog): that captures what
  the reader actually sees, but it still won't cleanly separate baseline from promotion. A few minutes of GPU.

## Files
- hypotheses.json — pre-registration + amendments.
- run.py — generation + matched-KL measurement, writes results.json (+ run.log).
- analyze.py — paired-bootstrap CIs, specificity, cross-lexicon, content words.
- results.json — per-prompt arrays for all conditions × 4 rungs, promoted-token dumps, lexicon sets.
