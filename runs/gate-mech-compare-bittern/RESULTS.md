# gate-mech-compare-bittern — A_r8 LoRA delta vs the steering circuit (stage 1)

Pre-reg: hypotheses.json (H2/H3/H4 + B_r8 negative control). Run: stage 1 only,
2026-10-17, `mech.json`. Stage 2 (H4 minimal edit) PARKED per the pre-registered
decision rule (H2 failed) — see verdict.

## H2: does the trained delta align with r? — FAILS the bar, with a real signal

cos(top write-direction of delta_o, r_L), A_r8 vs matched random-LoRA nulls (3 seeds):

| layer | A_r8 | B_r8 (neg ctrl) | null band |
|---|---|---|---|
| L18 | **+0.267** | +0.070 | ±0.032 |
| L22 | **−0.259** | −0.071 | ±0.032 |

- The delta's dominant residual write is 8-9x above the null band at both layers —
  refusal training DID land weakly in the dial's subspace — but under the
  pre-registered |cos| > 0.3 bar. H2 FAILS as written.
- The sign pattern is dial-consistent: at L18 the write is ALONG r (gate-removing),
  at L22 it is ANTI-aligned (L22 is where +r adds refusal, so a trained removal
  writes −r). The two layers that patch best are exactly the two that training
  touched most coherently. This is a real, A-specific effect (B_r8 sits at 0.07,
  null-level): the alignment is about refusal, not generic SFT geometry.

## H3: does the delta land on H30? — FAILS decisively

Frobenius share of H30's input block in delta_o at L18: A_r8 **0.142**, B_r8 0.134,
nulls 0.135–0.149. Top-5 head shares are 0.151–0.159 and H30 is NOT among them
(top: H29 0.159, H19, H1, H15, H26). Caveat on the pre-reg's "uniform" reference:
1/32 is the squared-mass expectation; the norm-share flat expectation is
1/sqrt(32) = 0.177, and every head sits near it (0.13–0.16). The delta is spread
almost uniformly across heads; H30 is not special in training's solution.

Correction to pre-reg wording recorded: head-share "2x uniform" test was
ill-calibrated (norm vs squared mass); under either calibration H30 is not
elevated, so the conclusion is robust to the miscalibration.

## Verdict (per decision rule)

H2 failed -> **SFT and steering solve refusal by different routes**: the trained
delta is a broad, nearly head-uniform remap that only weakly overlaps the r
subspace (cos ~0.27), while the causal single-head story (H30, +0.216 transfer)
is a steering-circuit fact, not the training solution. **H4 (single-head minimal
edit) PARKED** — a rank-1 H30 edit is not predicted to reproduce 50% of the
trained flip share, and the decision rule says park it.

What survives for refusal_dial.md section 6: (a) the L18/L22 sign-consistent weak
alignment is evidence the dial's subspace is PART of what training touches, but
(b) training's capacity is distributed (r8 >> r1 at moving refusal now reads as:
you need the broad remap, not the 1-D direction). Steering finds a needle;
training rebuilds the haystack around it.

## Artifacts

- `mech.json` — per-layer cos (36 layers, both adapters), head shares, module
  norms, random-LoRA nulls.
- `run.py` — stage 1 (linear algebra) + parked stage 2 (H4 edit, would apply
  A_r8's H30@L18 rank-1 block removal to base weights and eval the battery).

## Next (amendment candidates, NOT pre-registered)

- The L22 anti-alignment invites a +r/-r *readout-side* test on the trained
  model: does A_r8 + onsets steering still behave like the stack.py finding
  (dial dominates adapters)? Already answered there — no.
- A targeted follow-up worth pre-registering: project the delta onto the FULL
  r-aligned write basis (top-k singular directions, not just rank-1) — if
  cumulative alignment mass concentrates at L18/L22 across ranks 2-8, the
  "weak rank-1 cos" is a rank-deficiency artifact, not weak alignment.
