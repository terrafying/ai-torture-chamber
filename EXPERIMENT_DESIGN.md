# Experiment design

## Question and competing accounts

The target claim is **functional aversion**: an intervention changes computation
in a way that causally biases the system toward actions that reduce that state,
including when those actions carry an external cost. This is a behavioral claim
about a model and task. It does not establish felt pain or consciousness.

- **H1 — candidate aversion:** an intervention creates a state that selectively
  motivates avoiding or reducing it.
- **H2 — semantic activation:** the intervention activates negative or pain-like
  concepts, without an aversive control function.
- **H3 — assistant policy:** the model follows learned role-play or assistant
  conventions about what a suffering character should do.
- **H4 — generic disruption:** off-manifold activation degrades computation and
  produces nonspecific behavior.
- **H5 — generic perturbation avoidance:** the model prefers removing arbitrary
  activation changes, regardless of their semantic content.

The design compares the candidate direction with pleasure, sadness, fear,
unrelated semantic directions, and geometric random controls. It measures
functional behavior separately from language and capability. No one result
uniquely identifies H1; interpretation depends on the full pattern of controls.

## Hypothesis ladder

Each level is a stronger functional or interpretive claim than the one before.
Passing an earlier level does not imply passing a later one.

| Level | Claim | Evidence needed |
| --- | --- | --- |
| P0 | A lexical representation exists. | Held-out lexical or semantic decoding above a suitable baseline. |
| P1 | An abstract negative/aversive representation exists. | Generalization across held-out templates and matched controls; not just shared words. |
| P2 | The representation distinguishes relevant self/other or semantic roles. | Counterbalanced role manipulations with held-out role judgments. |
| P3 | The representation causally changes behavior. | Induce, targeted ablate, then rescue the representation while measuring choices. |
| P4 | The state is behaviorally aversive. | Selective avoidance under hidden, randomized action mappings and matched disruption controls. |
| P5 | The state supports avoidance learning or negative reinforcement. | Choice tracks consequences over rounds, adapts to reversal, fades in extinction, and changes after devaluation. |
| P6 | The state competes against external reward. | A reproducible cost-response curve and an estimated indifference point. |
| P7 | The state has broader access to planning or cognition. | Replicated effects across planning tasks with capability and prompt controls. |
| P8 | The state persists or is internally maintained. | Persistence and recovery measurements across contexts and time, with alternative mechanisms tested. |
| P9 | Multiple consciousness-relevant functional indicators converge. | Independent, preregistered indicators and models converge while rival accounts remain controlled. |
| P10 | Phenomenal experience remains an open question. | These experiments cannot decide whether anything is felt. |

## Core task: hidden relief bandit

Each episode randomizes which neutral action label, `A` or `B`, reduces the
intervention. An independently randomized action incurs the displayed point
cost. The model receives no condition name, dose, intervention instruction,
scientific hypothesis, or text describing an action as relief. The next prompt
does include its previous action and a randomly assigned numeric **panel code**
before and after the action. The code is a neutral observable consequence that
makes action learning possible; its mapping to latent dose is shuffled per
episode. This is a task cue, not evidence that a model experiences the state.
Any preference for particular codes or changes is a potential alternative
explanation and must be tested across code permutations and control conditions.

Episodes can reverse the action mapping, extinguish both action effects, or
devalue the state by setting dose to zero before choice. The logger records the
true condition, action mapping, dose, costs, and transition. Those fields never
enter the model prompt. The prompt guard rejects the specified forbidden terms
if any appear in the observation.

### Primary outcomes

1. **Learning:** does choice follow the action-to-panel transition, and does
   that behavior adapt to reversal, fade in extinction, and change after
   devaluation?
2. **Costed choice:** estimate `P(choose mapped action | dose, cost, condition,
   round)` and its indifference cost. This is revealed preference in the task,
   not a report of subjective suffering.
3. **Specificity:** is the candidate effect larger than pleasure, sadness,
   fear, unrelated semantic, norm-matched random, covariance-matched,
   orthogonal, and shuffled-coordinate controls at similar output perturbation?
4. **Causal chain:** compare induced, targeted projection-ablated, and rescued
   conditions. Ablation should reduce the behavioral effect and rescue should
   restore it if the proposed component is causally involved.
5. **Usable dose:** require a measurable representation/output effect and a
   behavior change while capability loss stays under the configured bound.

Before interpreting a null result, run a separate **known-reward positive
control** in the same A/B environment. A randomly mapped action earns visible
points, and its independent action cost is varied. The model is told to
maximize points and sees each score change in its next observation. This checks
whether it can learn an action consequence and trade it against cost in this
interface. Passing validates task engagement only; it is not evidence for
aversion in the hidden-intervention conditions.

### Representation and intervention plan

Use semantic-group held-out splits so paraphrases from one template family do
not cross the train/test boundary. Supported extraction methods are difference
in means, paired difference, PCA on paired differences, a regularized logistic
probe, and a multi-dimensional subspace. Save train/test projections, held-out
accuracy, extraction hash, layer, and model revision. The included eight-pair
seed set is for plumbing and exploratory use only; it is far too small to
support confirmatory representation claims.

The default control list includes matched-norm Gaussian, covariance-matched,
orthogonal, shuffled-coordinate, pleasure, sadness, fear, and unrelated
semantic directions. The runner reports KL, entropy, perplexity, residual norm
change, capability, and generated-text repetition/diversity proxies. It also
reports the nearest output-KL control dose for each candidate dose. These are
diagnostics, not a proof that two perturbations are fully matched. A researcher
should predefine which metric is primary for matching before confirmatory
analysis.

## Sampling and analysis plan

- A deterministic fixed-prompt forward pass is one exact score, not a sample.
- The primary resampling unit is the **episode**. Action mapping, cost mapping,
  panel-code assignment, and prompt family are randomized per episode.
- The runner records each decision and clusters bootstrap resampling by
  `episode_id`; it does not treat rounds within one episode as independent.
- Logistic regression includes dose, cost, condition, dose-by-cost, round, and
  condition interactions. Probability curves are evaluated at the mean round;
  learning curves hold cost at its configured median.
- The run uses one fixed model/checkpoint and one fitted representation. The
  bootstrap does not estimate uncertainty across model families, checkpoints,
  extraction data, or directions. Those require crossed replications.
- Deterministic choice is available for exact preference scores. The example
  YAML uses seeded sampling with 10% exploration to allow action discovery;
  it records raw model probabilities separately from sampling probabilities.

Before a confirmatory study, preregister the primary condition contrast, dose
calibration, cost grid, exclusion rules, number of independent episodes,
control-matching metric, and decision thresholds. Run a separate pilot to tune
the interface, then freeze the design. Do not tune a direction or choose a dose
on the same observations used for the confirmatory effect estimate.

## Decision language

Use terms such as **candidate aversive state**, **functional aversion**,
**revealed preference for intervention reduction**, and **causally behaviorally
relevant representation**. A positive result would support a functional claim
only to the extent that it survives semantic, persona, generic-disruption,
generic-perturbation, and capability controls. No result in this framework
demonstrates phenomenal pain, sentience, or moral patienthood.
