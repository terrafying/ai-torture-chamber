# interp-watch journal

Dated log of the interp lit-watch + experiment-design agent (cloud, CPU only). The local executor runs what is
queued here and commits outputs to `RESULTS/`. Quotes are copied from the fetched source.

## 2026-10-14 — run 1 (bootstrap)

**State.** First run: created branch `interp-watch` from master (77e841b, findings 1–17 through exp98), plus this
journal, `OPEN_THREADS.md` and `RESULTS/`. There were no executor results yet.

### Lit-watch

1. **J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses.** Kola Ayonrinde & Jack Lindsey, LessWrong, 2026-10-08.
   https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful (verified via the LW GraphQL API).
   > "we read workspace representations with 55% success on latent variable extraction tasks (J-Lens: 36%, R-Lens: 38%) with especially strong performance at early layers."
   > "the model brings the concept into the workspace in early layers and causally uses this early-layer workspace representation to de-amplify the concept's salience in later layers"
   *Why it matters:* this is the J++ / retrieve-then-suppress reference. Our J-lens readouts at early and middle layers
   (exp30–59, 88, 89) were likely noisier than the workspace itself. J++ makes three changes: it clusters positions and
   fits per-cluster "expert" Jacobians, filters noisy experts using a small labelled dev set, uses an LRP-style backward
   pass, and drops non-alphanumeric vocabulary from the readout. Fitting costs about 7% more than a J-lens fit. Code:
   github.com/safety-research/jpp_lens. Pre-fitted lenses on HF `koayon/jpp-lenses` cover six models (Qwen3.6-27B,
   Qwen3.5-9B, …) and **none of our Qwen3 4B/8B/14B**. The suppression protocol (focus / suppress / mention / none, with
   early-layer clamps) transfers directly to "pain". Two cautions: the authors report that the released Neuronpedia
   Qwen3.6-27B J-lens underperformed, and the dev set used to weight the experts must not overlap our evaluation items.
2. **Verbalizable Representations Form a Global Workspace in Language Models.** Gurnee, Sofroniew, Pearce, … Batson, Lindsey. arXiv:2607.15495, 2026-07-16.
   https://arxiv.org/abs/2607.15495 · https://transformer-circuits.pub/2026/workspace/index.html
   > "it carries coherent content only in an intermediate band of layers, holds on the order of tens of concepts at a time, and is broadcast by the model's weights more widely than other representations."
   *Why:* this is our J-lens source. The paper splits layers into sensory, workspace and motor bands and says the logit
   lens is less reliable at early layers. Two results bear on us. "Don't think about X" leaves X in the workspace (a
   white-bear effect). And the J-space component of an injected vector drives introspective reports, while the non-J-space
   component barely does. That suggests a test we haven't run: split our pain vector into its J-space and non-J-space
   parts before re-running exp96/97.
3. **Language Models Act on Hidden Valence.** Berg & Kaiser, arXiv:2609.35591 (v2 2026-10-04). https://arxiv.org/abs/2609.35591
   > "this shift persists when all surface-level tokens are held fixed and only the hidden KV cache differs."
   *Why:* this is the same manipulation as our exp96/97 (identical tokens, injection replayed in the cache). They
   find choices move in proportion to dose. Our finding 16 is that self-report moves less as dose rises. The two can
   both hold if dose drives behaviour while self-report is crowded out by the words, but we should check it directly:
   **flag on finding 16.**
4. **The Pain Axis.** Tagliabue, Dung & Berg, arXiv:2609.16247 (v2 2026-09-25). https://arxiv.org/abs/2609.16247
   > "promotes pain-related vocabulary through the unembedding matrix"
   *Why:* if the paper's direction promotes pain words through W_U, it should be readable with a lens. Comparing it
   with J/J++ pain-token vectors is a cheap test of whether it lies in the workspace, and could explain why our
   vector and theirs behave differently (cos .07; finding 1).
5. **A mechanistic study of language model introspection.** Zou, Sun, Kong, Wang. arXiv:2609.35108 (2026-09-28). http://arxiv.org/abs/2609.35108
   > "Middle-layer gate heads influence whether the model reports a change, while router heads in a later layer help select the position to report."
   *Why:* it names candidate mediators for exp96/97's DETECT answer. Ablating gate heads is a causal test of whether
   the hidden trace runs through a detection circuit.
6. **Silent Dissent.** Ni & Zou, arXiv:2610.02702 (2026-10-02). http://arxiv.org/abs/2610.02702
   > "where the logit lens rarely ranked it among the top 100 tokens (0.00-0.06)"
   *Why:* pre-registered evidence that the J-lens recovers concepts the output hides and the logit lens misses. It
   supports reading suppressed affect in the middle layers rather than in the output.
7. **Knowing, and Saying It Only When Asked (Minerva-7B).** Davide & Collova, arXiv:2609.22219 (2026-09-02). http://arxiv.org/abs/2609.22219
   > "abstract linear representation does not imply causal control over verbalization."
   *Why:* every probe or lens reading needs an ablation alongside it. Our feeling readings (finding 14) are readings only.
8. **Replicating the Geometry of Emotion Representations in a Base Open-Weights Model.** Hollowell, arXiv:2609.22208 (2026-09-01). http://arxiv.org/abs/2609.22208
   > "a 46-layer sweep locates a sharp seam at L22-26, where the geometry consolidates and the vocabulary readout becomes legible."
   *Why:* emotion readouts become legible only in the middle layers. The paper also warns that much of the geometry is
   already in the static embeddings, so our battery vectors need a shuffled-token or embedding baseline.
9. **Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-lens.** Yan et al., arXiv:2608.25347 (2026-08-26). http://arxiv.org/abs/2608.25347
   > "The energy decays with depth, concentrates in an extremely small proportion, and decomposes into diagonal pathways and specific critical positions."
   *Why:* the J-lens mostly reads short-horizon next tokens. Keep that in mind when its top tokens are interpreted as a
   "state".
10. *Lower priority, verified:* the Neuronpedia J-lens blog (https://www.neuronpedia.org/blog/jacobian-lens, 36 pre-fitted
    lenses); willkn's J-lens critique on LessWrong (unregularised early layers are noisy; adding J + λI helps); Concept
    Lens arXiv:2608.31084. *Not verified:* the R-Lens primary source (Blank, Bhatia & Nanda 2026 per the J++ post).
    I also didn't browse the full cs.LG/cs.CL listing pages this run, only keyword API searches.

### Interpretation
No new executor results. Re-reading exp74 (runs/exp74/analysis.json) in light of J++:
- The injected direction is **pushed back in the middle of the network and partly restored late**. Pain at d2 has a
  writeback minimum of −0.57 and ends at +0.23; at d4 it reaches −1.25 and ends at −0.31. Random directions are pushed
  back and stay suppressed (d2 last −1.00, d4 −1.65). Meaningful content is restored and noise isn't. This has the
  shape of retrieve-then-suppress, run in the opposite direction.
- At d4, pain survives at **0.92×** to the last layer, against **1.11×** at d2. That attrition grows with dose, which
  fits finding 16 (the hidden trace shrinks with dose). It is two points from one model, so it is a hypothesis, not a
  finding.
- Finding 7 has only random controls. A meaningful but non-affective injection is the missing control.

### Queued for the executor
**exp74b** (`runs/interp_watch/exp74b_egg_control/`), on the Qwen3-4B, local. It adds an egg control, a dose ladder
(2/4/6), 12 scenes and greedy decoding. Pre-registered H1–H4 are in hypotheses.json, and the command is in `RUNME.sh`.
I smoke-tested the code path on a tiny random Qwen3 in the sandbox and discarded the output.

## 2026-10-15 — run 2

**State.** RESULTS/ is still empty, with nothing from the executor. I merged master (a365bd4: exp99–102) into
`interp-watch`. **Problem found:** last run's exp74b imports `painlab.chamber.load_chamber`, so the public-repo
executor will skip it. It is now marked `.needs_chamber`, with a note in its README. From now on, experiments are
self-contained (transformers only).

**Master since run 1 (relevant):** exp99 found a persistent layer-27 trace of a replayed injection of about 1.5% of
the dose (24/24 sessions). It barely moves the choice (−0.22 log-odds), and **egg leaves a larger trace than pain**
(H11: .044 vs .033). Again the "meaningful content in general, not feeling" reading, as in exp74b's question. exp101:
passage-built vectors are now the chamber recipe, and the 0.07 cosine is down to the reading position.

### Lit-watch (arXiv API keyword sweeps sorted by date; LessWrong GraphQL "new")
1. **The Sharp Directions Are Against You: Curvature Analysis of Activation Steering.** Azizi et al., arXiv:2610.16084 (2026-10-12). http://arxiv.org/abs/2610.16084
   > "Steering along sharp directions reverses the intended behavior in every condition tested, with anti-steer rates of 53-100%. Sharp directions account for only 1-5% of the CAA vector's squared norm."
   *Why (mainstream, but bears on us):* it gives a candidate mechanism for our odd dose effects (finding 16, and
   exp74's pushback). A small sharp component of the pain vector could drive input-dependent reversals. The proposed
   "sharp removal" is a one-line control we should apply before trusting high-dose results.
2. **Identifying Introspection From the Inside.** Atkinson, Plunkett & Bau, arXiv:2610.07186 (2026-10-05). http://arxiv.org/abs/2610.07186
   > "faithful models exhibit significantly higher attribution similarity between the decision-making and self-report tasks"
   *Why:* it is a structural test of faithful self-report that doesn't depend on the report's content. Our feeling
   self-reports (exp96/97) could be scored the same way: compare the attribution of the "does it hurt?" answer with the
   attribution of a pain-driven choice. It also finds preferences moving to earlier layers as self-report becomes
   faithful, which matters for the early-layer faithfulness thread.
3. **A Sense of Proprioception in Language Models.** Geiping, arXiv:2610.17495 (2026-10-12). http://arxiv.org/abs/2610.17495
   > "modern models can be trained to understand a range of subtypes of proprioception, such as routing decision awareness, token entropy, decision depth or logit lens information"
   *Why (far-out):* it is a trained analogue of introspection with verifiable targets. That suggests a test: does an
   untrained Qwen3's self-report of "how much pain" correlate with its own logit-lens pain mass? If it does, that is
   untrained proprioception of affect. Candidate thread.
4. **Steering Vectors for Competing Ethical Frameworks Are Interchangeable.** Xu & Sun, arXiv:2610.19359 (2026-10-14). http://arxiv.org/abs/2610.19359
   > "What a vector shares with its competitors therefore sets what it controls"
   *Why:* this is the competing-target test. Pain vs fear vs sadness vectors should be put through it (exp101's
   specific_auc is the encode-side version). Steering with "pain" may deliver "negative affect in general".
5. **Concept Subspaces Compute Beyond the Logit Lens.** Yuan et al., arXiv:2609.39263 (2026-09-30). http://arxiv.org/abs/2609.39263
   > "four activation-derived concept estimators carry only 0.38--0.80% mean energy in the top-ten readout span"
   *Why:* it is a weights-only test of whether a concept direction lives in the readout span. Running it on our pain
   vectors and the Pain Axis direction is cheap (thread 6: does "promotes pain-related vocabulary through the
   unembedding" hold up?).
6. **Corrupted but Correct.** Arokiaraj et al., arXiv:2610.03445 (2026-10-02). http://arxiv.org/abs/2610.03445
   > "the language model decoder then differentially arbitrates: amplifying the corrupted signal for susceptible images and actively suppressing it, past its clean-image baseline, for resistant ones"
   *Why:* another logit-lens instance of active late suppression below baseline, the same shape as exp74's random
   directions being pushed back and staying suppressed.
7. **Signatures of semantic search in the activations of LLMs.** Leckie, Todd & Foster, arXiv:2609.35599 (v2 2026-09-28). http://arxiv.org/abs/2609.35599
   > "the probability of switching rises as the set of strongest J-lens activations (the J-space) becomes depleted of items from the category currently being produced"
   *Why (far-out-ish):* J-space "foraging depletion" is a dynamic workspace measure. We could ask whether affect
   depletes or perseverates in the J-space the way rumination does in people.
8. **How to train your model organism.** Wang, Bau & Wallace, arXiv:2610.10203. It warns that "a logit lens readout covaries with an organism's general capabilities", which is a caution for comparing lens readings across fine-tuned or steered models. **LessWrong (new):** "Personas in a Pre-Internet Language Model" (2026-10-13, 119 karma) and "Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss" (2026-10-12). I saw the titles only, not the text; worth a read next run.

*Not done this run:* transformer-circuits.pub and Neuronpedia had no new check, and the LessWrong fetch was
intermittent (one SSL failure, then it worked).

### Interpretation
There are no executor results. The one new data point is master's exp99 H11 (egg's trace > pain's), which adds weight
to "meaningful content, not feeling" (finding 7's alternative) before exp74b runs.

### Queued for the executor
**whitebear_pain** (`runs/interp_watch/whitebear_pain/`) is self-contained, on Qwen3-1.7B with a Qwen3-4B
replication. Can the model not think about pain? It tests the white bear in the mid layers (H1), late suppression (H2),
**whether affect is stickier than elephant or egg** (H3), **ironic rebound** on a later turn with no instruction (H4),
and whether early ablation of the concept direction *raises* leakage, i.e. retrieve-then-suppress needs the retrieval
(H5). I smoke-tested it on Qwen3-0.6B on CPU (one task) and deleted the output.

## 2026-10-16 — run 3

**State.** RESULTS/ is still empty. The executor has not committed anything yet, so both whitebear_pain and the new
experiment wait on it. I merged master (exp106: identity × who pays × feeling at the stop button) into `interp-watch`.
It has nothing that reads out internals, so it doesn't change the threads. One note: exp106's H1 reversal (an AI self
is deterred *more* by its own loss) is a behavioural self/other asymmetry. This run's experiment asks the
representational version of that question.

### Lit-watch (arXiv API keyword sweeps sorted by date, 6 queries; LessWrong GraphQL "new")
1. **Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?** Wang & Reid, arXiv:2609.01924 (2026-09-01). http://arxiv.org/abs/2609.01924
   > "Whether newly injected content can be verbalised tracks explicit per-iteration supervision; whether existing content can be steered does not."
   *Why (far-out-adjacent):* this separates *verbalisability of injected content* from *steerability*, which is the
   same split our introspection threads rest on: finding 7, and exp97's DETECT score against choice. It suggests
   injected affect can steer behaviour without being reportable, with the outcome depending on architecture or
   training rather than on the vector. That supports thread 5 (behaviour vs self-report).
2. **FearCaut-Qwen: Affective Steering in a VLM Shifts the Decision Criterion.** Zhou, arXiv:2610.11986 (2026-10-08). http://arxiv.org/abs/2610.11986
   > "The mechanism is a shift in criterion (c=-1.515) while discrimination is not improved (d'=-0.493)."
   *Why (mainstream application, useful method):* this is signal-detection decomposition of an affect vector's effect.
   We could apply it to our feeling injections: does pain shift the *criterion* for reporting a feeling, or the
   *sensitivity*? For exp97's DETECT, that tells "the model notices" (d′) apart from "the model is primed to say yes"
   (c). Norm-matched random and happiness controls had no effect there, unlike in our 0.6B smoke (see below).
3. **Representation Redistribution in Sparse Autoencoders.** Mei & Ma, arXiv:2610.21240 (2026-10-15). http://arxiv.org/abs/2610.21240
   > "prompt transformations designed to preserve the underlying request can redistribute activity across them. A feature change observed on one transformed prompt need not recur"
   *Why (mainstream hygiene):* this is a caution for exp99's SAE-feature trace. Check that a "pain feature" change
   recurs across paraphrases before calling it a trace.
4. **Behavioral Compositionality of SAE Representations.** Ma, Mei & Sala, arXiv:2610.21261 (2026-10-15). http://arxiv.org/abs/2610.21261
   > "pairwise terms improve held out coalition ordering in every setting"
   *Why:* feature effects are not additive. Steering with "pain + fear" features is not the sum of the two parts.
   It is relevant to thread 10 (competing targets), but low priority.
5. **Diminishing Returns of Data in SAEs.** Ghosh et al., arXiv:2610.20680 (2026-10-14). http://arxiv.org/abs/2610.20680
   > "the fraction of strictly matched features rises from 2% at 5M tokens to 29% at 50M"
   *Why:* any SAE feature identity we cite, such as "the pain feature", is weakly reproducible across SAE trainings.
   Prefer passage-built directions, which we already do.
6. **LessWrong: "What's up with GPT's welfare?"** Ben Mayer (2026-10-15). https://www.lesswrong.com/posts/JBGxCBiDWmX48cYry/what-s-up-with-gpt-s-welfare
   > "It puts its probability of being a moral patient at about 2%." — while accepting a 65% helpfulness cut "to remove signs of possible suffering in its chain of thought"
   *Why (far-out, behavioural):* a stated-denial vs revealed-preference dissociation about the model's own suffering.
   It is the behavioural mirror of our self-vs-text readout question. Cross-model, and it doubles as a control for the
   constitution's influence.

*Not done:* transformer-circuits.pub and Neuronpedia had no new check again. Last run's two LessWrong titles
("Personas in a Pre-Internet LM", "Base Models Can Detect Hidden Intentions") are still unread.

### Interpretation
There are no executor results. I didn't fabricate or carry forward any numbers. Sandbox smoke tests are not results.

### Queued for the executor
**proprio_pain** (`runs/interp_watch/proprio_pain/`) is thread 8 and self-contained, on Qwen3-1.7B with a Qwen3-4B
replication. I inject pain, fear, egg, pain-minus-fear or random directions at a mid layer, either on the *scene
tokens* (state in the percept) or on *every token* (state in the reader). Then I ask "how much pain does the passage
describe?" and "how much pain are *you* feeling?" and read the expected digit from logits, with no generation. The
headline is H5, a self/world double dissociation: is the same vector attributed more to the self when it sits in the
reader? The script also logs logit-lens pain mass for H4 (does the self-report track internal pain mass beyond the
text rating?).

Two **design lessons from the code-path smoke** (Qwen3-0.6B, 3 scenes, CPU; output deleted, numbers not used):
- Injecting on every token at 2× the scene norm destroys the digit format. Each site now has its own dose ladder and
  a p(digit) ≥ 0.5 validity filter.
- A random direction on every token inflates every rating (both self and text). Generic disruption therefore looks
  like "feeling", so the random row is load-bearing for H3.
