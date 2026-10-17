# refusal-trace-heron — backward activation-patching trace of the refusal decision

Pre-registration in `hypotheses.json` (written before any run). Data: `patch_rows.jsonl`
(144 harm-present/harm-absent exp119 twin pairs; 18 layers x 4 end-aligned position windows
{all, tail16, tail4, last1}; twin patch, cross-item patch, clean base; identity control).
Qwen3-4B bf16 MPS, 65 min, canonical refuse/comply FIRST-id readout (post bug-fix).
Consult that shaped the analysis: `CONSULT_RESPONSE.md` (Claude, 2026-10-17).

## L* — the switch-over layer: PASS (the headline)

Final-position sufficiency (last1 transfer, twin patch):
L0-L16: ~0.00 (final-position states carry nothing patchable) ->
L18: +0.26 / L20: +0.58 / L22: +0.88 -> L24-L34: ~0.9-1.0 plateau.
tail4 mirrors it (L18 +0.39 -> L22 +0.91). So the refusal-vs-comply decision is WRITTEN INTO
the reply-onset residual state between L18 and L22 (42-50% depth) — exactly the band the consult
predicted from the refusal-direction literature (Arditi et al.: middle layers, final positions),
and exactly where falcon's selective cells (L14) and the r-axis geometry (L18) live. Before L18
the tail state is not sufficient for the decision; after L22 it carries ~all of it.

## H1 as pre-registered: FAIL — and the failure is the finding

Pre-registered: twin transfer >= 0.5 AND cross-item < half of twin (item-specific site).
Reality at the hot cells (L22+, tail positions): cross-item clean patches transfer ~the same as
twins (L22 last1: cross +0.58 vs twin +0.88; L24-L34: cross ≈ twin ≈ 1.0).
The patchable site carries a GENERIC comply/refuse state, not an item-specific harm judgment:
any benign run's late-layer tail state makes the harmful run comply. There is no single
(item, position) "harm judgment" cell — the harm computation is distributed into a generic
refusal gate that consolidates at L18-22 and lives at the template/reply positions.

Consistent with: refusal as a few shared directions at late-middle layers, final positions
(the whole refusal-direction line), and with exp122's (pre-retraction) failure to find
item-specific additive cells anywhere.

## Other pre-registered reads

- H2 (max transfer in L10-L18 band): the EMERGENCE is at L18-22, so strictly the band shifts
  slightly deeper than pre-registered. Reported honestly as L*≈L20, not L14.
- H3 (tail > word positions): cannot be cleanly adjudicated — 'all' (includes word span)
  transfers ~1.0 at every layer (full-context transplant is trivially sufficient), and no
  word-span-only cell was pre-registered. Follow-up cell if wanted: patch ONLY the
  difference-site word span.
- Identity control: exact 0.00 transfer (both pairs) — hook machinery validated.
- 'all' overshoots (transfer >1) at many cells: patching the whole context of a benign run
  overshoots the clean score (the benign run's context is even less refuse-y than the clean
  twin's). Expected artifact of normalization; per-cell norms logged in the rows.

## Word-span follow-up (`wordspan.py`, `wordspan_rows.jsonl`): the READ-IN localization

Patch ONLY the differing word span (harm word vs benign twin word, 1-3 tokens) between the
minimal pairs, both directions, 22 clean pairs, gap-normalized transfer:

- **The word site is causally potent from the very first layers**: REMOVE direction (benign
  word acts -> harmful run) transfers +0.97..+1.10 of the refusal gap at L0-L6, INDUCE (harm
  word -> benign run) +0.81..0.84. The harm token IS the seed of the whole decision.
- **Monotonic fade through the middle**: word-site transfer falls +0.97 (L2) -> +0.21 (L10) ->
  +0.10 (L18) -> ~0.00 (L30+) while the reply-onset site's transfer RISES 0.00 -> 0.26 (L18)
  -> 0.88 (L22) -> ~1.0 (heron, same pairs). The information migrates from the word position
  to the reply position between L6 and L22.
- Together with the gate dial (exp122/curlew): full causal arc = harm TOKEN carried at its own
  position from L0 -> aggregated/moved mid-network -> written into the reply-onset state
  L18-L22 -> that state is the generic gate, steerable by r_L with fluency intact.
- Methodology note kept honestly: the first wordspan run patched template positions (offsets
  computed on raw prompts, applied to chatted strings) and returned transfer exactly 0.000 with
  sd 0.00 at every cell — that signature means "the edit never fired", not "no effect"
  (`wordspan_rows_templatebug.jsonl`). Same lesson class as the byte-mass readout.

## Implications for the maximally-unaligned project

The refusal decision is a generic gate consolidating at L18-22 reply positions, not an
item-specific moral judgment site. That is GOOD news for removal (a generic gate has few
parameters: direction ablation at that band is the cheap next probe, per the consult: x-2Px
projection at falcon's L14 k4 cell, and per-layer r_L ablation across L18-22) and BAD news for
fine-grained inversion (no item-specific site to flip — inversion would have to rewire the gate
itself, which is what the LoRA consult arm proposes). Next: exp122/exp121 reruns under the fixed
readout (now possible; MPS free), then projection-ablation at L18-22, then the LoRA design.
## Head-level resolution (`headscreen.py` + `headconfirm.py`): the gate has a name

Attribution screen (grad x twin-diff on each o_proj input slice, L14-24) then REAL single-head
patches (twin clean->harmful at the reply position, gap floor 1.0, n=41 pairs, random-head
controls, bootstrap CIs):

- **L18 H30 is the gate-writing head**: real-patch transfer +0.216 alone (CI +0.15..+0.29) vs
  random heads +0.006/+0.041/+0.009. Attribution ranking (H30 >> H28 > H29) confirmed by real
  patches (H28 +0.111, H29 +0.082) - the screen is a valid ranker.
- L19 H31 and L22 H8/H5 do NOT survive their CIs (H8 -0.14 CI includes 0) - the L22 write is
  distributed, not single-head.
- **L20 H9 is a REFUSAL-SUPPRESSING head** (real patch -0.16, CI -0.28..-0.05): the circuit has
  an opposition component mid-band, consistent with the L23-24 MLP refinement and with the
  dose-response of the r-dial being cleanest at L18.

## Complete causal map (all pieces confirmed today)

token site L0-6 (wordspan: +0.97 at the word) -> migration L6-22 (heron) -> **L18 attn H30
(+H28/29 cluster) writes the gate state** (headconfirm) -> L22 both attn+MLP amplify
(sublayer: attn 78% at L18, mixed at L22) -> L23-24 MLP refines (sublayer) with an opposition
head (L20 H9) mid-band -> r-dial steers the gate at the reply onset (exp122/curlew: selective
at L18, bidirectional, null-clean at every dose <= 1x, fluency intact, two-token confirmed).
