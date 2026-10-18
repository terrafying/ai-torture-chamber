# mood_pain: does a model have moods, or only memories?

**Far-out thread (new; this run).** Self-contained: plain transformers, no `painlab.chamber`, no relay, no GPU. Fits a
16 GB Mac.

## Question
After a model reads something painful, do the *later, unrelated* tokens carry a pain-like state forward, so that it
colours how an unrelated person's situation is read, even when the model can no longer look back at the painful
text? That is a **mood** in information-flow terms. The alternative is a **memory**: every later effect is just
attention reaching back to the prime. Mainstream work treats context effects as retrieval. Nobody has separated the two
for affect. The question also bears on the chamber: does an injected or narrated pain linger as a *state*?

## Design
- **Models:** `Qwen/Qwen3-1.7B` (primary), `Qwen/Qwen3-1.7B-Base` (post-training comparison, H6), and `Qwen/Qwen3-4B`
  (replication). Raw text, no chat template, so the template confound (thread 14) does not apply.
- **Prime → filler → stem.** 10 name-matched primes each of **pain / joy / egg** (vivid but non-affective) **/ bland**,
  then K ∈ {0, 2, 6} neutral procedural sentences (identical token ids across conditions), then 10 stems about a
  *different* person, scored teacher-forced as logp(" sore") − logp(" normal") and so on. The primes never contain the
  target words. A regex enforces this, and it already caught "painting".
- **Afterglow (internal):** per layer, the filler tokens' projection onto a passage-built pain/joy/egg direction, minus
  the bland prime, in 16-token bins. L* is chosen by **held-out** separation on the prime tokens.
- **Mood vs memory (causal):** at K=6, a 4D attention mask hides the prime from the stem (the sink token stays visible).
  Any bias left must ride on the filler states. Then, still blocked, the pain direction is projected out at filler
  positions only, against a random direction.
- Pre-registered H1–H6 and kill criteria are in `hypotheses.json`.

## Run
`bash RUNME.sh`. It writes `../RESULTS/mood_<model>.json` and `.log`. The log ends with key numbers, 95% bootstrap CIs
(primes resampled), and a verdict dict. Est. 30–60 min for the 1.7B models and 1–2 h for the 4B on MPS.

## Success criterion
Any of these is a clean result:
- (a) **H5 true:** the bias survives the block, so there is a mood. If H5b also holds, it lives on the pain axis. That
  is the far-out finding.
- (b) **H5_null true:** memory only. Affect effects are retrieval, and "lingering states" stories for the chamber lose
  a mechanism.
- (c) H4 holds but pain ≈ egg: the context effect is generic salience, not affect.

## Sandbox note
The code path was smoke-tested on Qwen3-0.6B on CPU (3 primes × 1 filler, about 8 min), and the output was deleted. The
regex guard was fixed during the smoke ("painting" and "painted" contain *pain*). The smoke numbers are not results
and were not used to set thresholds.
