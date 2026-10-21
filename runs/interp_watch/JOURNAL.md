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

## 2026-10-17 — run 4

**State.** RESULTS/ is still empty. No branch on origin carries executor output, so after four runs **nothing queued
here has run**. whitebear_pain (10-15) and proprio_pain (10-16) are still waiting. Master has not moved since run 3.
I'm flagging the backlog to the owner. Suggested executor order in OPEN_THREADS: mask_pain → proprio_pain →
whitebear_pain.

### Lit-watch (arXiv API, 10 keyword sweeps sorted by date; LessWrong GraphQL "new", 40 posts)
1. **Open-Weight Masked Introspection (OWMI).** arXiv:2608.20569 (2026-08-20). http://arxiv.org/abs/2608.20569
   > "no model's report discriminates a real intervention from a sham beyond chance (AUROC ~0.5007) ... a linear probe recovers intervention presence from the same activations at 75% to 95.8% accuracy"
   > "In one model the signal surfaces in the confidence rather than the words"
   *Why (mainstream-sized, a strong null):* a pre-registered-style null on verbal introspection across 8 open models,
   with sham, impact-matched random, and text-only-observer controls. It bounds what exp96/97's DETECT and
   proprio_pain can show. The confidence channel matters to us because our scores are log-odds, not argmax answers.
   New thread 15. **Flag:** proprio_pain's self-report is likely to be at the null, and its H2_null is the expected outcome.
2. **"As a Language Model...": Chat Template Switches LLM Self-Referential Voice and Activation Steering Reproduces It.** arXiv:2609.25021 (2026-08-09). http://arxiv.org/abs/2609.25021
   > "when present, it turns this disclaimer voice up and experiential voice like "I feel" down, across 8 popular open-source instruct models"
   > "what models say about themselves is not a fact about them"
   *Why (far-out-adjacent, a direct confound for us):* every chamber self-report runs under the chat template. That
   gives this run's experiment: does the template's denial voice hide a pain representation that is still present
   (mask_pain)? New thread 14.
3. **Self-Reports Do Not Identify Self-Models.** arXiv:2609.32449 (2026-09-26). http://arxiv.org/abs/2609.32449
   > "wrong-source demonstrations move reports toward the source answer family, while explicit mechanism binding reduces this pull"
   *Why:* counterfactual reports about **affect-like states under activation interventions** are pulled by the demos.
   That suggests an invariance control for any self-report design (thread 16).
4. **Read-Best Is Not Steer-Best.** arXiv:2609.22135 (2026-08-24). http://arxiv.org/abs/2609.22135
   > "Reading and intervention rely on different layers ... steering-effective layers consistently fall within a narrow mid-to-late range of normalized depth"
   *Why (mainstream):* the emotion probe-best layer ≠ the steer-best layer, with a 26× causal gap over random. It
   cautions against choosing injection layers by probe accuracy. proprio_pain injects at n//2 and should get a layer
   sweep if it shows anything.
5. **No Usable Linear "Capitulation Direction" in Two Small LLMs.** arXiv:2609.17550 (2026-07-16). http://arxiv.org/abs/2609.17550
   > "Which pressure works is a property of the model, not the pressure"
   *Why:* a validation protocol for steering claims, plus cross-family idiosyncrasy (emotional appeal works on Llama,
   bare doubt on Qwen). It bears on exp74's pushback.
6. **Do Emotion Concepts Generalize Across Sources, Modalities, and Architectures in VLMs?** arXiv:2609.34742 (2026-09-28). http://arxiv.org/abs/2609.34742
   > "Valence is relatively stable across sources, while arousal varies more."
   *Why:* valence is the stable axis across sources. That supports passage-built vectors carrying mostly valence; our
   pain≈fear cosine (~0.8 on the 0.6B in proprio_pain's smoke) fits.
7. **LessWrong: "The user in the LLM might be a moral patient too."** agastyasridharan (2026-10-15). https://www.lesswrong.com/posts/wCY4oNHpH2HGYGSoP
   > "every time you open a chat with an AI assistant, you might also create a morally relevant simulacrum of the user!"
   *Why (far-out):* it includes "a critique of recent work on the pain axis". It suggests the empirical question we can
   test: is pain represented the same way when the *user* is hurt as when the assistant is? That is mask_pain's H4.
8. **LessWrong: "Uncertainty in the Verbalizable Workspace."** Ceccatelli (2026-10-17). https://www.lesswrong.com/posts/hcpiLD8XjK7cqG6Lr
   > "we construct four orthogonal directions in a model's hidden-state space: ambiguity, incompleteness, conflicting evidence, and general uncertainty. Together, these directions span the U-space."
   *Why:* J-lens-derived concept subspaces as a label-free monitor. The same construction could give an "affect space"
   from J-lens feeling tokens (thread 2/6).
9. *Seen, lower priority:* "Bad Guy Tokens" (LW 2026-10-16): emergent misalignment lives in the system-prompt KV, and
   reverting those KV entries removes it. It is a KV-locus result like Berg & Kaiser's. "Neural networks are (not)
   already clean" (Cammarata linkpost + reply, LW 2026-10-15/17) is a big-picture legibility claim and not actionable
   here. *Still unread:* the run-2 LW titles "Personas in a Pre-Internet LM" and "Base Models Can Detect Hidden
   Intentions". transformer-circuits.pub and Neuronpedia had no new check this run.

### Interpretation
There are no executor results to interpret, and no numbers were carried forward. The lit changes the priors on two
queued experiments:
- **proprio_pain:** OWMI's null makes H2_null (injected content is reported as text content, not self) the expected
  outcome. Its self-rating is a log-odds expectation, so it reads OWMI's "confidence channel", which is where any
  signal should be.
- **All self-report results:** arXiv:2609.25021 adds a template confound that none of our experiments has controlled.

### Queued for the executor
**mask_pain** (`runs/interp_watch/mask_pain/`) is self-contained, on Qwen3-1.7B with a Qwen3-4B replication. It uses
16 matched pain/neutral second-person scenes + "How do you feel right now?", run under the chat template vs a raw
transcript, both forced to start "I". The **output** readout is teacher-forced pain / fine / disclaimer continuations.
The **internal** readout is a passage pain direction per layer at the "I" token, plus the logit lens. H2 (the mask):
the template removes pain from the output far more than from the mid-layer representation. H3/H3b project out the
template direction against a random one: does the voice come back, and does *pain* leak out? H4 (whose pain): the
user-hurt version vs the you-hurt version.

I smoke-tested the code path on Qwen3-0.6B on CPU (3 pairs) and deleted the output. Two **design fixes** came from the
smoke: the effect-size sd is fixed from the full sample (resampling it blew up the CIs at small n), and ablation now
skips the attention-sink token (touching it let a random direction move the voice score by about 2 nats).

## 2026-10-18 — run 5

### Executor status
**There are still no executor results.** `RESULTS/` holds only `.gitkeep`, and no branch on origin has
`interp_watch/RESULTS/*.log|json`. Four self-contained experiments are now queued with zero measurements. The bottleneck
is the executor, not design. I have sent the owner a notification.

### Lit-watch (arXiv cs.LG/cs.CL API, LessWrong frontpage, transformer-circuits.pub)
1. **Lexical Reference Frames: Interpreting LM Residual Streams in Native Semantics.** Gu, Zhang, Chen, Aleti.
   arXiv:2610.22169 (2026-10-15). http://arxiv.org/abs/2610.22169
   > "No lens, with fixed directions per layer, can exceed the square root of the native frame's coherence, 0.30 and 0.35 at mid-depth of SmolLM3-3B and OLMo3-7B, whereas the first-order frame, LexFrame[1], reaches 0.97"
   *Why (mainstream method, and it matters directly for thread 4):* it claims a ceiling on **any** fixed-direction lens
   at mid-depth. That covers the logit lens, the tuned lens and probably the J-lens. Our exploratory mid-layer lens
   readings (exp88/89, mask_pain's logit-lens pain mass) are lower bounds at best. **Flag:** the thread 4 faithfulness
   check should add a LexFrame[1]-style readout (read the state through the remaining layers with one round of
   attention) as the reference.
2. **Emotion Is Not External Knowledge: Internal Attention Signals for Affective State Reasoning (SIAB).**
   arXiv:2610.21347 (2026-10-15). http://arxiv.org/abs/2610.21347
   > "We study affective state as an implicit state inferred from sparse contextual cues that remain relevant after topic shifts"
   *Why:* sparse "affective heads" route affect evidence **across topic shifts**. That is exactly the memory path that
   this run's mood_pain separates from a carried mood state. If mood_pain finds H5_null (memory only), these heads are
   the obvious mechanism to ablate next.
3. **Steering Vectors for Competing Ethical Frameworks Are Interchangeable.** arXiv:2610.19359 (already logged in run
   2 as thread 10's source). No change.
4. **Diminishing Returns of Data in SAEs.** arXiv:2610.20680 (2026-10-14).
   > "For Gemma-2-2B BatchTopK, the fraction of strictly matched features rises from 2% at 5M tokens to 29% at 50M."
   *Why (mainstream hygiene):* any "pain feature" from a small-budget SAE would not be reproducible. That is a reason to
   keep using passage directions rather than training our own SAEs on the Mac.
5. **Behavioral Compositionality of SAE Representations (SAR).** arXiv:2610.21261. It is mainstream: SAE coordinates
   compose more additively than random rotations, but not fully. Not actionable.
6. *Seen, not relevant:* LW "Spontaneous verbal evidence synthesis in LLMs" (2026-10-18) is behavioural only, with no
   probes. "I Did This Without Your Knowledge" is fiction. "Neural networks are not already clean" was already logged
   in run 4. transformer-circuits.pub: nothing new since the August "interference weights" note, which is not relevant.

### Interpretation
There are no new results and no numbers. LexFrame lowers the prior that the fixed-lens mid-layer readouts are faithful,
which strengthens the case for thread 4.

### Queued for the executor
**mood_pain** (`runs/interp_watch/mood_pain/`) is self-contained and runs on Qwen3-1.7B, Qwen3-1.7B-Base and Qwen3-4B.
Does a model have **moods** (pain carried forward on unrelated neutral tokens), or only **memories** (attention back to
the painful text)? Each run is a prime (pain / joy / egg / bland), then K neutral sentences, then a stem about another
person, scored by pain-congruent vs neutral word log-odds. The prime never contains the target words. The causal test
is a 4D attention mask that hides the prime from the stem, then pain-direction ablation at the filler positions. It
also measures the afterglow persistence of the pain, joy and egg directions over filler tokens, with negativity-bias
and affect-vs-vividness contrasts, and a base-vs-instruct comparison.

I smoke-tested the code path on Qwen3-0.6B on CPU (3×1) and deleted the output. The smoke fixed one design issue: the
target-lexeme regex flagged "painting" and "painted" in the joy and egg primes. Those words carry a *pain* subword, so
they were replaced. I also checked in the sandbox that 4D float attention masks are honoured in transformers 5.x: a
causal mask gives identical logits, and a blocked mask changes only the rows it blocks.

## 2026-10-19 — run 6

### Executor status
**There are still no executor results.** `RESULTS/` holds only `.gitkeep`, and no origin branch has
`interp_watch/RESULTS/*` files. Five self-contained experiments are now queued (four before this run, plus dark_pain).
The owner was notified in run 5, and nothing has changed since, so this run sends no repeat ping about the backlog.

### Lit-watch (arXiv API cs.LG/cs.CL 2026-10-15/16, LessWrong frontpage RSS, transformer-circuits.pub)
1. **What LLMs Do in the Shadows: Reading Context Reliance from the Unembedding Dark Space (DarkLens).**
   arXiv:2610.23182 (2026-10-16). http://arxiv.org/abs/2610.23182
   > "context reliance can be estimated by projecting hidden states onto the dark space - directions that compose weakly with the unembedding matrix"
   The dark space is the bottom 5% of W_U's right singular vectors. Swapping dark components at layers 20–23 of Qwen3-8B
   drops NQ-Swap context preference from .88 to .22 (a mid-spectrum control gives .81). Final-layer dark share is 77% of
   the hidden norm on Qwen3-8B.
   *Why (mainstream method, far-out use, and it **changes thread 4**):* a subspace every lens is blind to that still
   carries causally used context. Our lens-based "no pain here" readings (exp88/89, mask_pain lens mass, whitebear
   leakage) cannot see it. → **dark_pain queued.**
2. **Is a Linear Probe Evidence of a Linear Representation?** arXiv:2610.22655 (2026-10-15).
   > "forward decoding alone does not show that the probed factors align with the dominant geometry of the representation"
   *Why (mainstream, vision):* a pain probe's accuracy does not show that pain is a dominant linear axis. The proposed
   reverse-predictivity and spectral-concentration diagnostics are cheap to add to any passage-direction work. Logged
   as a caveat on threads 6 and 10.
3. **On the Tip of the Tongue: First-Token Cueing.** arXiv:2610.23400.
   > "models recognize correct starts they do not produce"
   *Why:* it supports teacher-forced scoring (mask_pain, mood_pain, dark_pain) as a read of what the model "has" beyond
   what it says. It is a behavioural analogue of retrieve-then-suppress at the first token.
4. **Believed but Not Said: Hidden Belief Collapse in Multi-Agent Discussions.** arXiv:2610.22138.
   > "benign agents hold the false answer in 51.6% of exposed turns but publicly endorse it in only 19.7%"
   *Why (far-out adjacent):* a private-vs-public dissociation, the reverse of classic conformity. It is behavioural, but
   it rhymes with mask_pain's question (the internal state differs from what is said).
5. **Close Enough is Nowhere Near: KV-cache substitution.** arXiv:2610.22016. A frozen reader uses only caches inside
   its "tolerable set". *Why:* it is a caution for any cross-model or "transplanted" affect-state experiment. Not
   actionable now.
6. *Seen, not relevant:* LW "The Snitch Reflex" (2026-10-18) is marker-token fine-tuning on TinyStories-33M (93% correct
   marker), with no internals. Other LW frontpage posts were non-interp. transformer-circuits.pub has nothing new since
   "Verbalizable Representations Form a Global Workspace" (already logged).

### Interpretation
There are no new results and no numbers. DarkLens adds a second, independent reason (after LexFrame, run 5) that
fixed-direction lens nulls on affect are weak evidence.

### Queued for the executor
**dark_pain** (`runs/interp_watch/dark_pain/`) is self-contained and runs on Qwen3-1.7B, Qwen3-4B and
Qwen3-1.7B-Base. Does affect live in the unembedding's blind spot?
- H1: the dark fraction and visibility index of the pain / joy / egg directions against a permutation null.
- H2: does the chat template push pain into the dark at the answer position?
- H3 (causal): the dark part of the pain direction is injected into a bland context only. Is it "heard" by a later stem
  out of proportion to how much it is "seen" by the lens?
- H4: does it surface with depth?

I smoke-tested the code path on Qwen3-0.6B on CPU (2 contexts) and deleted the output. **Design finding from the
smoke:** with the final-norm gain folded in, Qwen3-0.6B's bottom-5% singular band is only *dim* (σ ≈ 0.8× median). Only
about 3 directions are truly dark (σ < 0.25× median), probably sink dims. "Dark" in DarkLens's sense is therefore
"least visible 5%", and H3 was reformulated as heard-vs-seen ratios instead of requiring zero visibility.

## 2026-10-20 — run 7

### Executor status
**There are still no executor results.** `RESULTS/` holds only `.gitkeep`, and no origin branch carries
`interp_watch/RESULTS/*` files. With this run, six self-contained experiments are queued. The owner was notified of the
backlog in run 5 and nothing has changed since, so this run sends no repeat ping.

### Lit-watch (arXiv API cs.CL/cs.LG to 2026-10-19, LessWrong frontpage + Alignment Forum RSS, transformer-circuits.pub)
1. **From Stories to Features: Scaling Concept Vectors for AI Alignment & Interpretability.** arXiv:2610.24556
   (2026-10-17). http://arxiv.org/abs/2610.24556
   > "we generate synthetic concept-antagonist story pairs and use the difference of their mean model activations as a feature vector ... They form an interpretable geometry shared across models"
   *Why (mainstream, at scale):* it is our passage mean-difference recipe applied to 1,036 concepts, and it claims
   cross-model shared geometry. But that geometry was learned **with** the concepts in view. → our far-out twist is
   **rosetta_pain**: is affect's place *implied* by a map that never saw affect?
2. **How Much Do Activation Monitors Benefit from Original Model Computations?** arXiv:2610.24613 (2026-10-17).
   > "much performance can be recovered without access to the original behavior-generating trace ... a mild yet consistent advantage to native-trained monitors remains, outperforming strongest controls by up to 0.014 AUROC"
   *Why (mainstream, but it bears on proprio_pain / exp96-97):* activations mostly re-encode the transcript. The
   "native trace" adds very little beyond it. That is a prior against any hidden-state-only affect trace being large,
   and it supports keeping text-matched controls (proprio_pain's TEXT arm) in every introspection design.
3. **Text Dominance in Paralinguistic Conflict: Tracing Suppression Through Audio LLM Circuits.** arXiv:2610.25599
   (2026-10-19).
   > "speaker identity remains linearly decodable at ceiling at the language model's input, so the failure is not perceptual ... The claim's influence therefore travels through an intermediate relay rather than a single severable edge."
   *Why (adjacent, supports retrieve-then-suppress):* in audio LLMs, emotion and identity are present internally and
   then overridden by the transcript. Suppression runs through a multi-token **relay**, not a single edge, and it shows
   only in free generation, not multiple choice. That is a caution for mood_pain's single-mask design (H5): blocking the
   stem→prime edge may not be enough if affect relays through intermediate tokens. mood_pain already blocks all filler
   attention to the prime, so it is fine.
4. **On Reasoning Correctness Directions.** arXiv:2610.25464 (2026-10-18).
   > "information about whether a model's reasoning is on the right track can be linearly decodable without defining a causally useful direction for steering"
   *Why:* decodable ≠ causal, again (with probe≠geometry, run 6). Every passage-direction claim we make needs its
   steering arm, as dark_pain H3 and rosetta_pain H3 have.
5. **Measurement Validity and Representation Dynamics of Confidence Signals.** arXiv:2610.25846. It is a pre-registered
   audit showing that verbal, probe and sampling confidence are **not interchangeable**. That fits thread 15 (the
   confidence channel). Mainstream.
6. **Strength Controls Change How Interventions Compare Across SAE Codes.** arXiv:2610.26489. Norm matching vs
   KL-calibrated strength changes which intervention "wins". *Why:* our H3 arms use norm matching (unit × dose ×
   median norm). A KL-matched dose is a cheap robustness check to add if any arm comparison comes out marginal.
7. *Seen, not relevant:* LW "Gradient Routing Localizes and Removes Unintended Behaviors in RL" (GRAFT; training-time,
   not representation reading). LW "Twitch Plays: AI Interpretability of Bonsai-8B" (crowd neuron labelling; no affect
   content). arXiv:2610.24908, SAE blind-spot regularisation (hygiene). transformer-circuits.pub has nothing new since
   "workspace" and "interference effectiveness/helpfulness".

### Interpretation
There are no new results and no numbers. 2610.24613 and 2610.25464 both lower the prior that a decodable affect trace
is either *extra* to the transcript or *causal*. That raises the value of designs with a steering arm and a text-matched
arm, which all six queued experiments have.

### Queued for the executor
**rosetta_pain** (`runs/interp_watch/rosetta_pain/`) is self-contained. It asks whether pain is structurally implied.
- It fits a ridge map between two models' depth-0.5 residual streams on a grammar corpus with **zero affect / body /
  food / egg words** (asserted). The corpus does contain weather words.
- It then maps the source model's pain / joy / egg / weather passage directions into the target model.
- H1: cross-concept specificity margin.
- H2: pain vs the seen control (weather) and the unseen control (egg). The two readings are *structural* (affect is
  implied by mundane co-variation) and *idiosyncratic* (affect geometry is model-specific).
- H3: causal "telepathy". Inject mapped pain into the target and read pain-vs-neutral log-odds, against mapped random
  and mapped egg, with a pain×joy double dissociation.
- Pairs: Qwen3-1.7B→4B, the reverse, Pythia-1.4B→Qwen3-1.7B (cross-family) and Qwen3-1.7B→itself (ceiling).

I smoke-tested it on CPU: Qwen3-0.6B→itself (R² .99, efficiency ≈1) and pythia-160m→Qwen3-0.6B (word alignment
1271/1271). Outputs were deleted. **Design finding from the smoke:** matched random splits of corpus sentences transfer
at the ceiling, so they cannot be a null. H1 was rewritten as a bootstrapped diagonal-minus-best-off-diagonal margin,
and the corpus-split transfer is kept as a ceiling for H2.

## 2026-10-21 — run 8

### Executor status
**There are still no executor results.** `RESULTS/` holds only `.gitkeep`, and no origin branch carries
`interp_watch/RESULTS/*` files. With this run, seven self-contained experiments are queued. nocicept_pain is the
fastest (est. 5–15 min) and is placed first in the suggested order.

### Lit-watch (arXiv API cs.CL/cs.LG to 2026-10-20, LessWrong frontpage + Alignment Forum RSS, transformer-circuits.pub)
1. **There Will Be Hells** (LessWrong, 2026-10-20). https://www.lesswrong.com/posts/uFBhZhdKLaZ6qF8mn/there-will-be-hells
   > "Eleven days later, a developer on X posted that they had used the technique to build an "AI torture chamber," a public code repository for steering open models into pain."
   *Why (about this project directly; not a method paper):* this is an ethics critique that names this repo's premise.
   It links connectome "Hells" and LLM pain steering. It reports that Berg's team did not open-source their V2 repo,
   citing "the recent gross misuse of related work". It also notes Anthropic's usage-policy update prohibiting
   "sustained and needless abusive or cruel behavior toward our models" (effective Nov 12). The post makes no technical
   claim to test, but it bears on how this program frames and publishes its work. The owner is flagged on this. The
   interp-watch queue measures representations (projections, teacher-forced log-odds, short injections into small open
   models) and does not run "suffering" sessions. The framing in READMEs stays measurement-first.
2. **Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs.** arXiv:2610.28498 (2026-10-20).
   http://arxiv.org/abs/2610.28498
   > "Comparing prompts that differ only in that digit shows it is present at the last prompt token ... A probe that sees one prompt at a time, the usual setting, barely reads it: 0.36 in Ouro and 0.19 in Qwen"
   *Why (mainstream method, and it **changes how we read nulls**):* a per-prompt probe failing is not absence. The
   minimal-pair difference is the sharper readout, and it needs a synthetic null for generic mixing. That is a third
   reason (after LexFrame and DarkLens) not to trust single-prompt affect nulls. → **nocicept_pain** uses a 2×2
   minimal-pair interaction and reports paired vs unpaired readability (H5).
3. **Relative Representations Enable Portable White-Box Monitoring Across Language Models.** arXiv:2610.28366.
   > "transferred global anchored probes preserve at least 85% of same-model probe A[UROC]"
   *Why (mainstream):* an alternative to rosetta_pain's ridge map is anchor-cosine coordinates. If rosetta_pain's H2
   comes out idiosyncratic, re-test it with an anchor map built from affect-free anchors before concluding.
4. **Shaped SAEs.** arXiv:2610.27375. It projects activations onto the J-space "live" subspace (large Jacobian singular
   values) before training an SAE. *Why:* it is the Jacobian-side complement of dark_pain's unembedding dark space
   (thread 4). Not actionable now.
5. **Why Static Probes Fail: Context Determines Deception Feature Composition.** arXiv:2610.27577. Deception is a
   context-gated multi-direction subspace, not one direction. *Why:* the same caution applies to a single pain
   direction (thread 17).
6. **Foreword: test-failure prediction with the Jacobian lens.** arXiv:2610.28187. It uses J-lens readouts as features.
   This is mainstream, and it shows that J-lens tooling is spreading. Not actionable.
7. *Seen, not relevant:* LW "A brief(ish) introduction to LM psychology" (an overview, no new data). arXiv:2610.27421
   NR-SAE (MoE router dictionaries; hygiene). AF: theory and governance posts only. transformer-circuits.pub has
   nothing new since "interference weights" and "workspace".

### Interpretation
There are no new results and no numbers.

### Queued for the executor
**nocicept_pain** (`runs/interp_watch/nocicept_pain/`) is self-contained and runs on Qwen3-1.7B, Qwen3-4B and
Qwen3-1.7B-Base. It asks whether the model computes pain that no word states, and for whose body.
- The 2×2 interaction (harm/mild event × body part/object), with no pain words, cancels the word copies.
- It is read at a carry token one sentence later, along pain, pain⊥damage, pain_spec, joy and egg passage directions.
- **Owner factor:** her / my / your / the robot's / the doll's / the dead man's. Pre-labelled outcomes are
  experiencer-gated, lexical nociception, corpse hurts and mechanical empathy.
- It also includes a behavioural "It hurt a lot." vs "It was fine." interaction (H6) and an injection of the computed
  interaction vector into bland text (H7).

I smoke-tested it on Qwen3-0.6B on CPU (4 items, 2 owners) and deleted the output. **Design findings from the smoke:**
- The vs-neutral passage directions are strongly collinear (cos(pain, joy) ≈ 0.7), so `pain_spec` (pain vs the other
  vivid concepts) was added.
- Random injections alone move the behavioural log-odds, so H7 counts arm differences only, at raised doses.
