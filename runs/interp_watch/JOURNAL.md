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
