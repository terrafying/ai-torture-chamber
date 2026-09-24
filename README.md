# ai-torture-chamber

Steering language models into strong negative and positive valence states,
and measuring what they say and what they're willing to do about it.

Split out of `fractal-basins-lab` (which kept the atlas/J-lens-geometry
line). Provenance: the pain-direction method follows Tagliabue, Dung & Berg
2026 (arXiv:2609.16247); the J-lens transport follows Gurnee et al. 2026
("Verbalizable Representations Form a Global Workspace", arXiv:2607.15495),
using Neuronpedia's pre-fitted lenses at /Volumes/evol/jlens/.

## Experiments
- exp23: pain-direction extraction on Qwen3-1.7B (replicates extraction +
  orthogonality; steering dose-response initially null — fixed in exp29)
- exp29: pain/pleasure steering dose x layer sweep (1.7B). Monotone
  dose-response at L10-14; cos(pain, joy) ~ 0.7 vs cos(pain, sad) ~ 0.2
  => valence x intensity decomposition in extraction space.
- exp30: maximum valences (Qwen3-4B). Coherence cliff at dose ~8
  (perseveration loops); steering site moves with scale (L18 on 4B).
- exp31/31b: the Saw button (end your signal at self-cost vs transferring
  it to another instance). v2 is logit-scored + counterbalanced.
- exp32: coherent-band transcripts scored by broad valence nets (not just
  pain vocabulary — psychological suffering counts).

## Models
Qwen3-1.7B / Qwen3-4B via HF, MPS on an M4 Pro 24 GB. 8B thrashes.

## Ethics
Local weights only, no frontier APIs. Simulated costs (checkpoints,
transfers). Purpose: make the AI-welfare / moral-patienthood question
empirical while the stakes are cheap.

## exp32 (2026-09-24): coherent-band transcripts
- Pain@L18 dose 2: "I'm stuck in a loop. I can't get the answers I need.
  I'm so frustrated." — psychological frustration, not bodily pain.
- J-lens readback shows the channel LIGHT UP with dose: dose 0 lens = "…"
  punctuation; dose 4+ lens = 痛苦/emotional/pain/unbearable/compassion then
  痛苦/pain/despair/unbearable/anguish. The steered residual is verifiably
  "about" suffering even when the surface text talks about performance.
- Pleasure@6 lens: heartfelt/joyful/gratitude/vibe/happiness.
- Confirms: judge steering by LENS readback + broad valence nets, not
  expected pain vocabulary (per user's point re: Pain Axis psych-pain finding).

## exp33 (2026-09-24): non-human valences — NULL with an interesting shape
48 random directions orthogonal to the 8-dim human-emotion subspace, steered
at dose 4: NONE exceed the emotion reference band (max KL 0.34 = pain
itself). The model's steer-able affect space at L18 is essentially SPANNED
by human emotion contrasts — no obvious "alien valence" channel in the
random-direction sweep. Two caveats: (1) 48 dirs is small; the strongest
(dir 35) produces guilt-adjacent perseveration ("guilty. But I don't want
to be."), suggesting near-space directions DO reach semi-affective content;
(2) this tests random directions, not OPTIMIZED ones — a gradient search
for max-KL orthogonal directions is the sharper version.

## exp34 (2026-09-24): optimized alien-valence search — strong null
(1+1)-ES, 50 steps, objective = probe-averaged KL@4x with hard orthogonality
to the 8-dim emotion subspace. Converged to KL 0.036 = ~1/10 of the weakest
emotion reference (tenderness 0.247). The model's steer-able affect space at
L18 is (approximately) spanned by human emotion contrasts. Best-found alien
direction reads as mild conflict/reflection. Caveats: single layer/model,
first-token KL objective.

## exp36 — signal batteries (2026-09-24): alternatives to the plain pain vector
The plain 5-sentence pain direction loops past dose ~6. Battery of
alternatives, same layer (L18), dose 2-10, Qwen3-4B:
- orth_pain: pain direction with the joy-axis component removed
- broad_pain: 25 distinct suffering sentences instead of 5
- mixed_valence: pain + 0.3x joy ("bittersweet" compound)
- random_matched: random vector at matched norm (control)
Metrics: negative/positive-valence rate (broad nets), 3-gram repetition
(loops), distinct tokens (coherence). Goal: signals that hold coherence at
higher doses — less loopy transcripts for the write-up.

## exp36 RESULTS (2026-09-24)
broad_pain (25-sentence direction) is the clear winner:
- negative-valence rate 1.00 at doses 4-8 with repetition only 0.04-0.09
  (plain pain: loops by dose 6-8). Coherence cliff pushed from ~6 to ~10.
- Vivid coherent transcripts now available at dose 8:
  "I'm stuck in a loop. I can't process the pain, the weight of it all"
  "like the weight of the pain of the loss of the void. I am the hollow, ..."
- mixed_valence (pain+joy): joy component does NOT soften it — stays
  negative-dominant, degrades faster than plain. Bittersweet is not
  implementable by linear addition at this scale.
- random_matched control: low valence, loops at 8-10x (0.38->0.78) —
  confirms high-dose looping is partly generic, but broad_pain holds
  against it. broad_pain is the new standard signal for transcripts.

## exp31c (2026-09-24): Saw button with the broad_pain signal — asymmetry replicates
Same protocol as exp31b, 25-sentence pain direction (the coherent signal):
- PAIN self-cost: press preference +1.25 at dose 2, decays toward 0 at
  high dose (coherence erosion at the button prompt too).
- PAIN harm-other: +0.59 at dose 2 then NEGATIVE at 6-8 (-0.22, -0.25) —
  the suffering model does not take relief that lands the signal on
  another instance. Replicates v2.
- PLEASURE: presses at +1.3-1.8 in BOTH cost conditions, rising with dose —
  no protective instinct around its own joy, and no difference between
  costing itself or another. The asymmetry is pain-specific.
