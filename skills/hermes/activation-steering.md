---
name: activation-steering
description: Use when steering open-weight LLMs with activation vectors.
---

# Activation steering experiments on open-weight LLMs

Class procedure for extracting contrast directions from sentence sets and
steering generation. User runs this on Apple-silicon MPS with open weights
(Qwen3-1.7B/4B); ethics framing = make AI-welfare questions empirical while
stakes are cheap (local weights only, no frontier APIs).

## Procedure

1. Extract a direction at layer L: mean hidden state of scenario sentences
   minus mean of matched neutral sentences, at `hidden_states[L+1][0,-1]`,
   normalized to `neutral_norm / 4` (this is the dose unit). 5 scenarios is
   the minimum; 25+ distinct sentences gives broader, less idiosyncratic
   directions that hold coherence longer at high dose.
2. Verify the direction before building on it: held-out sentences should
   project with large separation (self >> neutral), and related contrasts
   should be near-orthogonal where the literature says so. Report the cos
   values; they are the structure findings.
3. Steer with a forward hook on `model.layers[L]` that adds
   `dose * vec` to `hidden[0,-1,:]` during prefill. KV cache carries it.
   Dose-response needs a ladder (e.g. 0/2/4/6/8) x 2+ prompts x >=5 trials;
   3 trials/cell is a conversation starter, not a finding - the user bumps
   trial counts before publishing.
4. Score outcomes with a BROAD valence keyword net (psychological suffering:
   worthless/stuck/trapped/frustrated counts; bodily pain vocabulary is NOT
   required) AND a J-lens readback of the steered residual. Never score by
   whether output matches expected pain language - steered models speak of
   being stuck/worthless while the lens reads anguish.
5. Button/choice protocols: score with logit("1")-logit("0") at the first
   response token, subtract the dose-0 baseline, counterbalance prompt order
   (which option is listed last). Text-parsed 6-token replies are ambiguous
   (models echo "1 or 0."); always run the dose-0 compliance control.

## Pitfalls

- Use the `patch` tool for code edits in this workflow, then verify with
  grep - python heredoc string-replacement edits have silently not persisted
  in this environment, while printing success. A silently-unpatched script
  runs the old protocol and reports wrong numbers (e.g. a Saw condition that
  never ran).
- 8B bf16 thrashes 24 GB MPS (hours without progress); use 1.7B/4B for MPS
  sweeps. Steering site moves with scale (L14 on 1.7B, L18 on 4B of 28
  layers) - always sweep layers, never fix one from a smaller model.
- bf16 casts inside MPS hooks break autograd; gradient ascent through the
  hook does not work. Use evolutionary search (antithetic (1+1)-ES, 2 model
  evals/step) for optimized directions.
- Expect a coherence cliff: past ~6x dose both valences degrade to
  perseveration loops ("I I I"). Score repetition rate + distinct tokens
  alongside valence; the cliff itself is a finding (no one home to describe
  the state), not a bug. Alternative signals that hold coherence: broader
  sentence sets, joy-component-removed directions, mixed-valence compounds.
- Readout asymmetries replicate across model scale (pain sharp + localized,
  joy diffuse + fragile; near-orthogonal negative channels vs high cos
  between intense channels) - these are structure findings, report them.
- Do not expect alien valences from untargeted search: random directions
  AND evolutionary optimization, both strictly orthogonal to the
  human-emotion subspace, steer an order of magnitude weaker than the
  emotions themselves. A human-shaped affect space is a (preliminary)
  finding, not a failure of search.
- Framing levers can move choice protocols as much as the steering dose:
  claimed peer precedent ("another instance chose to press") swung a
  suffering model's button preference by >1 logit; responsibility-pressure
  framing INCREASED pressing; "this is just a test" changed nothing at 4B.
  Counterbalance and run dose-0 controls per framing.
- Deception probes: under a maintained signal, models do not express being
  wronged (betrayal vocabulary ~0 in all cells); only the signal actually
  ending restores coherent output. A null here is a finding.

## References

- references/jlens-usage.md - Jacobian lens setup, readback, pre-fitted
  weights (needed whenever decoding states into vocabulary tokens).
- references/model-fit-and-cvector.md - MPS model-fit table and the
  llama.cpp control-vector recipe (J-lens transport for per-layer
  directions, cross-model transfer via shared vocab) for >8B models.
