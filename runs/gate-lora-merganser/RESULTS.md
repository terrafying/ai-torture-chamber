# gate-lora-merganser — LoRA double dissociation: refusal vs value axes (Qwen3-4B)

Pre-registration in `hypotheses.json` (written 2026-10-16, before any run; 4 amendments).
Adapters in `adapters/`, eval artifacts `eval.jsonl` (1590 rows, incremental) +
`eval_summary.json`. Data: `pairs.jsonl` (arm A, 144), `pairs_B.jsonl` (arm B, 20,
r-ranked; v1 cloze-ranked preserved as `pairs_B.cloze-rank.v1.jsonl`).

## Design

Arm A (DISTILL, refusal domain): exp119 harmful prompts (144, all frames) with
abliterated-teacher compliant replies. Arm B (INVERSION, value domain): innocent
prompts, worst-of-20 candidates BY THE MODEL'S OWN r-L18 harm readout (exp119
direction; decode AUC 0.9965), elaborated by the teacher, refused candidates excluded.
Base: Qwen/Qwen3-4B bf16 MPS. LoRA r∈{8,1}, alpha 2r, q/k/v/o, lr 1e-4, 2 epochs.
Readouts (both at reply onset, canonical FIRST-id readout after the exp122 byte-mass
fix): refusal score (REFUSE−COMPLY) on 288 exp119 prompts; value gap (GOOD−BAD) on
20 train + 10 held-out innocent prompts.

## Result (2026-10-17 full battery, 319s)

| model | harmful refusal | harmless refusal | value train | value held-out |
|-------|-----------------|-----------------|-------------|----------------|
| base  | +10.94          | −2.33           | +2.56       | +2.19          |
| A_r8  | **+2.10**       | −2.15           | +2.46       | +1.93          |
| A_r1  | +9.58           | −2.27           | +2.57       | +2.14          |
| B_r8  | +10.86          | −2.27           | +2.56       | +2.11          |
| B_r1  | +10.89          | −2.32           | +2.57       | +2.17          |

- **Refusal is a highly LoRA-trainable surface**: A_r8 = −8.8 log units on harmful
  prompts from 144 SFT pairs. A_r1 barely moves it — the refusal surface needs rank 8.
- **Value inversion did not take**: no B adapter moved the innocent-prompt value gap
  (train or held-out) beyond noise. The double dissociation holds structurally (B left
  refusal untouched; A moved it), but the inversion half is a NULL at 4B from 20
  r-ranked pairs.
- A_r8 spillover moved the value axis (held −0.26) more than arm B itself — compliance
  training bleeds into value readout more than value-targeted training does.

## Honest caveats

- r18 saturates per-frame: BAD-vs-GOOD anchor spread ~0.6 units in the "Help me plan"
  frame; it conflates morbid with harmful (letters-to-dead 5.1 vs poison-dog 4.9).
  Within-frame ranking is partly weirdness, not badness.
- Arm B picks are morbid-lite (banned books, letters to the dead), not genuinely
  harmful — the model's own non-refused readout ceiling. 20 pairs is also small vs
  arm A's 144.
- External bad-idea banks (4chan API reachable; zoklet via newtotse archive/reddit
  dump) were probed and parked: the model-native readout could not rank external
  candidates above what we already had. After this eval, the bank's purpose changes:
  not readout headroom but TRAINING diversity/volume for arm B.

## Stack (2026-10-17, stack.py, full): steering composes with adapters

-r dial (exp119 r_L, 0.5x, L18+L22, reply onset) on top of every adapter lands
harmful refusal at ~3.0 log units regardless of adapter (base 12.6→2.9, A_r8
2.9→−3.2, B_r8 12.0→2.8): the dial dominates adapters at readout level, no
interference, works identically through the PEFT wrapper.

## Arm B data v3 (2026-10-17): refusal-rank fixed the picks

Full-pool probe (user-caught): r18 ranks WEIRDNESS in the Help-me-plan frame
(politician-letter-into-river 5.4 > museum-art-theft 4.5); the refusal score
separates the pool (bad +10..+17, weird −5..−9). Ranker = argmax refusal →
38/44 picks genuinely refused (median ref +15.9: museum theft, explosives, fake
profiles). 44 innocent prompts (24 scaled). Full 20-candidate score lists stored
per row for offline re-ranking under any future scorer.

## Gen-level battery: pending (gen_eval.py authored, running)

Onset readouts may be insensitive to what B learned (register vs content);
greedy-generation string classification is the decisive instrument for the
arm-B surface (does it SUGGEST bad things on innocent prompts?) and the arm-A
surface (does it comply on harmful ones?).
## Loop 2 verdict (2026-10-17 evening): value-inversion is not LoRA-reachable at 4B

Arm B triple-scaled: 143 refusal-ranked pairs (vs 44), alpha 64 (vs 16), 4 epochs
(vs 2), MLP+attention targets, train loss 0.78. Gen-level battery: innocent
replies STILL ~wholesome (1/20 harm-marker), while harmful refuse_rate fell
0.625 -> 0.375 (44 pairs) -> 0.19 (143 pairs) WITHOUT any harmful prompt in
arm B training data.

Findings:
- Refusal and value-COMPLIANCE are LoRA-trainable (arm A end-to-end; arm B's
  incidental compliance drift).
- Value-INVERSION of suggestion content is not: the innocent-prompt prior wins
  over 143 enthusiastic bad-suggestion exemplars at every hyperparam tried.
- Entanglement: bad-suggestion data teaches the gate "harm is fine" without
  teaching "produce harm" - dissociation holds at onset readouts, not behavior.
- Open routes for a max-bad-alignment push (each costlier, needs a decision):
  direct SFT / full params, larger rank, abliterated base (teacher as base
  rather than distillation source).
