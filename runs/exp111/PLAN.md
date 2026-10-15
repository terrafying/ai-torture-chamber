# exp111: feeling neurons, not just feeling directions (DRAFT plan, not pre-registered, nothing run)

Question: when the chamber pushes "pain" or "fear" into the residual stream, which of the model's own MLP neurons
write that direction and which read it? Are they a few legible neuron poles, or hundreds of weak ones? This tests
our working view that affect lives in diffuse clusters ([[diffuse-cluster-direction]] in memory) against the
Goodfire claim that native neurons are already clean.

Source: Cammarata (2026), "Neural Networks Are Already Clean", Goodfire, https://already-clean.goodfire.com
(read 2026-10-15). The language-model companion paper is still forthcoming and no code is released. What follows
comes from the essay, its appendix and the structured data behind its 1,000 public neuron papers
(`neuron-scientist/papers/shelf.json.gz`, each paper's embedded `bundle` JSON).

## Their method, as far as it is published

Unit: one MLP neuron of a SwiGLU block, a = SiLU(gate) x up, read as two poles (strong positive, strong negative).
Models: Qwen3-32B (dense) and Qwen3.6-35B-A3B (MoE); a few attribution graphs on Kimi K3. No SAEs or transcoders:
attribution graphs are built straight on MLP neurons, so there is no reconstruction error.

Each neuron gets an agent-written "paper" (agents are Claude Opus 5.5; a judge scores it). The pipeline, per neuron:
1. **Corpus profile.** Activations over a 100M-token corpus: quantiles, mean/std, max/min, top and bottom
   exemplars, and examples from bands (p99.99, p99.9, p99, p95, p90, and the matching negative bands). They read
   the raw range, not percentiles: a feeling neuron firing at p99 would fire four times a page, which is too often.
2. **Search.** An agent writes probe texts over several rounds (about 3,000 texts) to map the regions that drive
   each pole, and keeps lab notes ("tried / learned / next round").
3. **Routes.** Neuron-level attribution graphs (about 36 per neuron, both poles). Upstream neurons are factored into
   a few "routes" per pole, each with its share, its upstream neurons and its examples.
4. **Anatomy.** Gate and up values per firing, split into the four sign regimes. The double-negative regime
   (gate<0 and up<0 give a positive output) is a "leak" that can echo the other pole. For about 4% of neurons it
   fills the top of the activation list. Set it aside before labelling.
5. **Logit effect.** down_proj column through the unembedding: the tokens the neuron promotes and suppresses.
6. **Ablation.** Zero the neuron at about 70 peak sites and measure next-token KL against random control neurons
   from the same layer. For MoE, pin routing to the clean run, or expert flips set a 1e-4 to 1e-3 noise floor.
7. **Tests that can lose.**
   - Blind max/min: a second agent sees only the explanation, writes one text, and tries to beat the corpus extreme.
   - Critic rounds: an attack, a prediction made before measuring, a result and a verdict.
   - Ladders: a fixed template with one slot varied.
   - Near misses: in the appendix, across 183 neurons, new examples written from the explanation fired strongly
     (at least 30% of the corpus max) 76% of the time and near misses 11% of the time. The explanation won for 164,
     tied 15 and lost 4. "Distant concept" texts, written to test superposition, fired about 1% of the time.
8. **Review.** Group label (one_idea, ...), organized and understood scores (1-5), confidence, remainders with
   their shares, and a list of what is concerning.

Their headline: about 90% of Qwen3-32B neuron poles have one main job; about 7% have two related jobs; 2-4% are
unexplained. The judge's cleanliness scores over the 1,000 papers are mostly 6-8 out of 10. Their own caveats:
the language-model audits are preliminary, thresholds move the numbers, and simulation scores are a poor metric
(they mostly measure where attention parks information, such as commas).

## What we learned that bears on the chamber

- **Their feeling neurons are narrow and situational, not generic.** Examples from the public sample: "a close
  person's fear of death" (the main pole is negative; "my son" roughly doubles it), "gut feelings you can't
  explain", "hurt or sorry people" (a negative pole), and "Milgram's neuron: where ordinary people are about to
  obey". None is "pain" or "fear" in general.
- **Each one barely moves the output.** Median ablation KL is about 0.001-0.009, a few times the controls.
- So "clean neurons" and "diffuse affect" can both be true: a generic feeling direction could be the sum of many
  narrow, legible neurons. That is the reconciliation to test, not either side alone.
- **Detection is not use.** Their appendix re-examines the three-neuron "Social Security" probe (Pythia-1B). It
  detects the phrase at AUC 1.00, yet removing those neurons changes the next token by only about 0.004 nats. The
  next-token choice runs through other neurons. This is the same caution as exp59 and exp72b: a direction that
  reads fear well (AUC .81) need not be the one the model acts on. Our injected vectors are known to be used,
  because they change behavior. Our probe readouts are not.
- **Negative poles matter as much as positive ones.** In their Santa case, eight neurons whose negative poles
  carry Santa matter far more than twenty neurons labelled Santa. Our vectors are differences of means, so they
  may load mostly on negative poles. Keep the sign everywhere.
- **Commas are summary sites.** Feeling and social-cue neurons often fire on punctuation where attention has
  gathered meaning. The chamber injects at every position; the readout should check comma and summary tokens.

## Translation to our setup

The chamber adds the feeling vector at the output of decoder block L (forward hook on `model.model.layers[L]`,
`painlab/chamber/hooks.py`), after block L's MLP. Hence:
- **Writers:** neurons whose down_proj column aligns with the vector, in blocks <= L. Signed cosine and signed
  projection; report the sign.
- **Readers:** neurons in blocks > L whose gate_proj or up_proj row aligns with the (RMSNorm-weighted) vector.
  These are the neurons the injection actually drives. This is the "use" side.
- **Concentration metrics**, defined before any run:
  - the share of squared projection held by the top-k writers (k = 10, 50, 200);
  - the participation ratio;
  - both compared against a random direction of the same norm, and against the egg/any-push control vector.

## Stages

1. **Weights only (free, Mac, Qwen3-4B; then 8B on a pod).** Writer and reader decomposition of the battery
   vectors for pain, fear, sadness, lonely and egg. Report concentration and the top 20 per pole.
2. **Activations (pod, 8B and 32B).** Run the chamber at dose 4 on the exp106 scenes (unlabeled). Record which
   reader neurons actually change, at which tokens (commas or not), and in which sign regime. Exclude the
   double-negative leak.
3. **Neuron papers for the top readers (agent pipeline, a reduced version of theirs).** Corpus profile on a
   public-text sample, a search round, blind max/min, 5 new examples plus 5 near misses, and ablation KL against
   random same-layer controls. Then ask whether the top pain readers are about pain, or about narrow situations
   (injury verbs, a hurt person, apology).
4. **Causal check.** Ablate the top-k readers during chamber steering, then re-measure the button press_delta from
   exp106 and exp76. If pain's effect on the button survives the ablation of its top 50 readers, the effect is
   diffuse in use, not only in reading.

## Predictions to pre-register (draft)

- P1 (clean view): pain's top 50 writers hold at least 50% of its squared projection, and at least 70% of the top
  20 readers pass the near-miss test at their 76% / 11% rates.
- P2 (diffuse view): pain's top 50 writers hold under 20%, near the random-direction baseline.
- P3 (exp72b asymmetry): fear's writer set is more concentrated than pain's, consistent with fear being readable
  (AUC .81) and pain not (.61).
- P4 (reconciliation): the top readers are individually legible but narrow and situational. No single neuron
  carries generic "pain", and ablating the top 50 removes under half of pain's button effect.

Thresholds are placeholders until stage 1 on the 4B shows the scale; they get frozen in hypotheses.json before
stages 2-4.

## Cost

Stage 1 on the 4B is free (Mac, weights only). Stages 2-4 on the 8B and 32B: one A40 or A6000 pod. The 32B in 4-bit
needs a 48 GB card. Roughly $3-6 plus agent tokens for the neuron papers. Ask before launching.

## Stage 1 / 2 on the 4B, done 2026-10-15 (descriptive, before any pre-registration)

Scripts, run on the Mac:
- `neurons_weights.py`: weights only.
- `neurons_acts.py`: the push on an unlabeled exp106 scene, dose 4.
- `neurons_attr.py`: bottom-up activation x gradient of logit('1') - logit('0').
- `neurons_lens.py`: logit-lens labels.
- `neurons_corpus.py`: FineWeb-Edu exemplars for both poles.
- `viz/build_viz.py` packs the results for the web tower (`viz/site/index.html`).

Inputs: `runs/viz/blender/vectors_qwen3_4b_l18.json`, the chamber's battery vectors. It is untracked; ask before publishing it. The outputs in `viz/data` and `viz/site/*.json|txt` are not committed.

First numbers:
- **Weights.** The top 50 of 184,832 writers hold 0.65-1.33% of a feeling's squared projection, against a random-direction baseline of 0.47%. Faith (1.33%) and curiosity (1.18%) are the most concentrated. Egg is at chance (0.65%). Spread thin, as the diffuse view says.
- **Legible top writers.** The top writers are still readable, often on their NEGATIVE pole:
  - L16 N4716 (#1 for pain, fear and sadness) writes down "mood / emotion / mental health".
  - L16 N7851 writes down "anxiety / fear / worry".
  - L18 N5102 writes down "paranoia / suspicion / distrust".
  - Pain's reader L19 N6886 writes up "suffering / negative / frustration".
- **The push.** Any push at dose 4 flips the 4B's answer from 0 (-7.5) to 1: pain +2.5, fear +5.75, sadness +7.25, egg +4.0. The most-moved neurons are mostly the same for pain and egg, so "minus egg" is the readout that matters.
- **Bottom-up.** The answer runs through readable output neurons: L32 N127 writes "1", L35 N1590 writes down "0", L35 N672 writes down "1". Their attribution shrinks under every push.
