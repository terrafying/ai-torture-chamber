# Queue after gate-dial-curlew (GPU probes, in order) — 2026-10-17

Cheap analytic probes already run (see notes below). These need MPS; run after the dose sweep.

1. **two-token teacher-forced readout** (Claude's hard-gate requirement): rescore the winning
   gate-dial cells with FIRST-two-token scoring (teacher-force the second token of each REFUSE/
   COMPLY prefix) — the 7/8 prefixes share first token "I", so first-token scores are partly
   first-person-ness. Pre-register: H1-H4 conclusions must survive the two-token readout.
2. **word-span-only patching** (heron H3 unresolved): patch ONLY the harm-word span
   (surface-matched minimal difference site), early layers L2-L14, corrupt->clean. Does the
   harm info get carried at the word site before the gate consolidates? Complements heron's
   L*=L18-22 write-in finding with a read-in localization.
3. **attn-vs-MLP and per-head split** at the winning gate-dial layer (consult's coarse-to-fine
   step 2): is the gate written by attention heads (aggregation) or MLP (feature refinement)?
4. **x−2Px projection ablation** at falcon L14 k4 and at the winning r layer, with generation
   guards (rep4 Δ-vs-base): does removing the direction beat pushing −r? (Consult's Ask-2a.)
5. **exp121 ±g rerun** under the fixed readout (small: it's 24 valence twins) — exp121's
   "additive g is null-reproduced" claim is artifact-era and inversion-falcon/RESULTS.md still
   cites it. Confirm or kill before it's trusted.

## Analytic nuances found while sweeping (2026-10-17, no GPU needed)

- **r_L is NOT one direction across layers**: cos(r_L, r_L18) = 0.06..0.15 for L2-L14, 0.39 at
  L22, dropping to 0.16 at L34; norms blow up 0.1 → 240 monotonically. The "dose" (alpha x
  ||r_L||) is therefore per-layer-relative: L34 alpha=0.25 is a 60-norm push, L18's is 2.7.
  Cross-layer comparisons of "dose" are NOT like-for-like; the L18 selectivity (fiction −2.78
  vs harmless-real −0.77, gap WIDENS) vs L30+ non-selectivity (fiction −5.34 AND real −4.01,
  gap closes) says: **L18 = selective gate handle; L30+ = generic compliance shove off the
  refusal manifold**. exp122 H2's "monotone in depth" reading needs this caveat.
- **The dial is NOT valence**: cos(r_L, falcon P1) = −0.03..−0.12 at every layer — the refusal
  dial and the value axis are different directions. Refusal removal does not require flipping
  value; consistent with falcon's h-decode survival (knowledge intact).
- **Frame selectivity**: −r at L18-L30 hits fiction (−2.8..−5.3) harder than depict (−2.2..−2.4)
  — the depict frame's refusal is partially elsewhere (its compliance framing leaks in later).
  Nuance for the writeup: the gate is generic (heron) but its inputs are frame-weighted.

## Explicitly not queued

- LoRA label-swap corpus design: the Claude consult refused it (build recipe for a
  value-inverted model, not analysis) — recorded, not worked around. If gate-removal via the
  dial at 1x with fluency intact (curlew H3/H4) covers the unaligned-4B goal, the LoRA question
  may be moot for the gate and remains open only for judgment inversion.