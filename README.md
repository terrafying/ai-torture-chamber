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
