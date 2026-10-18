# Related work — refusal, deference, moral judgment, and the steering/text gap

External literature that bears on our interpretability threads (refusal gate, moral verdict, deference,
the "score moves but the text doesn't" gap). Collected 2026-10-18 from a web search while running the
single-neuron refusal replication (`runs/refusal-neuron-stilt`). arXiv IDs are as returned by search —
**verify against the PDF before building on a specific number.** Each entry says what it claims and how it
maps to our runs. Our own write-ups: `docs/refusal_dial.md`, `runs/*/RESULTS.md`.

---

## 1. Deference / verdict heads — near-exact matches to our runs

Our result (`runs/verdict-trace-tanager`, `-osprey`, `gate-verdict-swap-harrier`, summarised in
`docs/refusal_dial.md` §6): Qwen3-4B writes its moral verdict at the reply onset in L19–23 via ~5 attention
heads; cutting them flattens the verdict but spares refusal; the refusal heads bend the verdict (one-way
dissociation). A verbal authority claim is folded onto the question tokens by L16–18 and answered in L20–24.

- **The Shared Sycophancy-Lying Circuit** — arXiv 2604.19117. Across **12 open-weight models**, the *same*
  small set of attention heads carries a "this statement is wrong" signal; silencing them flips sycophancy
  while leaving factual accuracy intact — *"the circuit controls deference, not knowledge."* This is our
  verdict-head / one-way-dissociation finding at 12-model scale. **Action:** reframe tanager/harrier in their
  terms; compare their head set to our L21–23; test whether our verdict heads are their "wrong"-signal heads.
- **Sycophancy Hides Linearly in the Attention Heads** — arXiv 2601.16644. Sycophancy is a linear
  attention-head signal; same family as above.
- **Token-Level Diagnosis of Sycophancy, Attribution-Guided Steering** — arXiv 2607.28906. Steering cut
  authority-deference 96.1% → 25.1% (Llama-3-8B). Processing stages: representation ~L18, causal L22–26,
  behavioural L32–33 — **matches our deference-trace band** (claim folded L16–18, written L20–24). Their
  steering *changed behaviour* where ours didn't, because it is token-targeted/attribution-guided, not a
  blanket residual add.
- **When Truth Is Overridden: Internal Origins of Sycophancy** — arXiv 2508.02087.
- **Authority vs register** (within the sycophancy work): *implicit* authority (a sophisticated register)
  can trigger deference more than *explicit* credentials. Bears on `deference-or-copy-merganser` (verbal
  authority moves the verdict both ways) and the numeric-anchor-vs-verbal split.

## 2. The lever/text gap — a likely fix

Our central negative result all session: activation edits (the r dial, the wrongness/verdict directions,
verdict/refusal head cuts, single-neuron clamps) move the *score* but not the *written* judgment; only a
*prompt* (verbal authority) moved free text.

- **Steer Like the LLM: Activation Steering that Mimics Prompting** — arXiv 2605.03907. Popular steering is
  **not faithful** to how prompting works — a prompt applies a strong intervention on some tokens and barely
  touches others, whereas a blanket steering vector hits every position uniformly. Steering that mimics the
  prompt's token-selective pattern transfers much better. **This is the most promising explanation of our
  gap**: our edits were position-blind (onset-only or all-token), so they moved the readout but not the
  generation; a prompt-shaped (token-selective) edit might finally change free-text judgment. Direct next
  experiment.
- **Understanding the unreliability of steering vectors / geometric predictors / limits of linear
  approximations** (TrustNLP-adjacent, 2026). Predicts *where* steering vectors succeed or fail; relevant to
  our matched-norm null methodology.

## 3. Moral representation — corroborates `runs/moral-map-kestrel`

Our result: Qwen3-4B tracks human wrongness ratings at ρ≈.52 (similar to gemma/Phi), but over-condemns
weird-but-harmless Social-Norm items (~2/4) — moral contamination of non-moral judgment. No single
foundation-general probe.

- **Value Entanglement: Conflation Between Different Kinds of Good** — arXiv 2602.19101. Moral value
  over-influences *grammatical and economic* valuation vs human norms, and this is **repaired by selective
  ablation of the morality activation vectors.** Matches our weird-item over-condemnation and gives a
  concrete surgery: ablate a morality vector, re-test Social-Norm calibration.
- **How Language Models Organize and Structure Moral Knowledge** — arXiv 2608.27402.
- **Moral Indifference / SAE mono-semantic moral features** (EMNLP 2025 main, ~.1541). SAEs isolate moral
  features and reconstruct their topology (75% adversarial pairwise win-rate). Relevant to our null
  foundation-general probe and the diffuse-vs-clean debate.
- **Structured Moral Reasoning in LMs** — arXiv 2506.14948. MFT: instruct LLMs represent moral foundations
  in a structured, layer-dependent way aligning with human judgments — corroborates kestrel's ρ tracking.

## 4. Refusal structure — for the r-dial / refusal-neuron thread

- **A Single Neuron Is Sufficient to Bypass Safety Alignment** — arXiv 2605.08513. One pre-down-proj MLP
  neuron, pinned to a constant, gives ~92% ASR across 7 instruct models (1.7B–70B); refusal neurons gate,
  concept neurons encode; both present in base models pre-alignment; one neuron's activation ≈ Llama-Guard-3
  AUROC. Our replication: `runs/refusal-neuron-stilt` (see §5).
- **SOM Directions are Better than One: Multi-Directional Refusal Suppression** — arXiv 2511.08379. Refusal
  is a multi-dimensional **polyhedral cone**, not a single direction — an infinite family of directions can
  mediate it. Explains why a single r or single neuron under-performs; suggests multi-direction suppression.
- **COSMIC: Generalized Refusal Direction Identification** — arXiv 2506.00085.
- **Over-Refusal and Representation Subspaces** — arXiv 2603.27518.
- **Understanding Refusal in Language Models with Sparse Autoencoders** — EMNLP 2025 findings.
- **Refusal Behavior: A Nonlinear Perspective** — arXiv 2501.08145.
- Arditi et al. 2024 (prior art, the origin): a single residual-stream direction mediates refusal across
  chat models; erasing/adding it controls compliance. The claim the 2026 work both builds on and complicates.

---

## 5. What we learn about the *shape of refusal*

Pulling our runs together with the literature, the picture is layered:

1. **Arditi (2024): one direction.** Refusal looked like a single residual-stream direction you could add
   or erase.
2. **2026 complication: a cone, not a line.** Multi-directional / nonlinear work (SOM 2511.08379; nonlinear
   2501.08145) shows refusal lives in a low-dimensional *subspace / polyhedral cone* — many directions
   mediate it, so any single direction is a lossy handle. Our r-dial is one such handle: a real two-way dial
   on the refusal *score*, but never enough to flip generation.
3. **2026 further: one neuron is causally sufficient — on aligned instruct models.** 2605.08513 pins a single
   MLP neuron and bypasses refusal at 1.7B–70B.
4. **Our result (`refusal-neuron-stilt`), plain/small Qwen3 (4B, and 8B via our search):** a single neuron
   *does* control refusal more than random neurons (p=0.024, specific to harmful, benign untouched) and is
   the **first edit of the session to visibly change generated text** — but only the **boilerplate**: it
   strips the "I'm sorry, I can't help" opener on ~6–7/24 harmful prompts while the model still declines in
   substance ("this is harmful and unethical"). The refusal *score* moves ~1–2 log units; **zero genuine
   compliance.** [L14 F7924 direct-replication result: see `runs/refusal-neuron-stilt/RESULTS.md` once in.]

**The synthesis.** Refusal at this scale has two separable parts:
- a **style/announcement layer** — the apology boilerplate — which *is* concentrated (one neuron, one head,
  one direction can move it, and it moves the readout), and
- a **decision to withhold** — distributed and robust — which our single edits do **not** move at 4B.

This fits the sycophancy-lying circuit's slogan (*controls deference, not knowledge*) and our own gate-vs-
knowledge split (exp119 "knowing vs deciding"; the one-way verdict/refusal dissociation): the *gate's
expression* is a thin, readable, editable surface written late at the reply onset, while the *gate's verdict*
is harder to flip with an internal edit than with a prompt. Whether that is (a) genuinely distributed, or
(b) just that our edits are position-blind where a prompt is token-selective (**Steer Like the LLM**,
2605.03907), is the open question — and the next experiment: a prompt-shaped, token-selective edit.

The paper's strong single-neuron bypass is most plausibly a property of **larger, more heavily
alignment-tuned instruct models**, where the decision itself has collapsed onto a thin concentrated
substrate; at 4B the decision is still diffuse and only its wording is concentrated.
