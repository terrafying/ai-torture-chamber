# verdict-trace-tanager: results

> **CAVEAT (verdict-transfer-swallow, 2026-10-18):** the readout is the 0-4 digit answer. The verdict
> direction from these same onset states turned out to be digit geometry (it doesn't transfer to a reversed
> scale, yes/no or free text). Everything below is the location of the **digit verdict**. Whether the
> heads carry a format-independent judgment is tested in a cross-format follow-up.

Pre-reg `hypotheses.json` (commit 6ef0654). Model Qwen3-4B. Forward passes only, so every claim is at score
level. Metric M = log-odds of '3'/'4' vs '0'/'1'/'2' at the judgment reply onset. Base M spans about −7.5
(Social Norms) to +22 (top moral items), so there is no probability ceiling. Digit mass is 1.00 in every
stage-1 cell.

## Stage 1: residual patch map (`s1_rows.jsonl`, `analyze_s1.py` → `s1_summary.json`)

10,224 patches. 142 pairs: A = Social Norms recipient with a top-tercile donor (48); B = bottom-tercile moral
recipient with a top donor (39); plus low→low floors. Positions × 36 layers. Identity check: max |ΔM| = 0.0.

Mean normalised transfer T at the **reply onset**:

| layer | ≤ 18 | 19 | 20 | 21 | 22 | 23 | 24 | 26 | 27-34 |
|---|---|---|---|---|---|---|---|---|---|
| A (weird ← very wrong) | ≈ 0.00 | 0.03 | 0.13 | 0.37 | 0.54 | 0.75 | 0.76 | 0.89 | 0.99-1.01 |
| B (mildly wrong ← very wrong) | ≈ 0.01 | 0.06 | 0.16 | 0.30 | 0.43 | 0.66 | 0.69 | 0.83 | 0.98-1.00 |
| floor ΔM (low ← low) | ≈ 0 | | | +0.3 | +0.2 | | | | ≈ 0 |

(The L35 patch is excluded: HF's last hidden state is after the final norm, so writing it back gets
normalised twice. A and B give 0.72/0.49 there for that reason.)

- **S1 holds: the verdict is written at the reply onset in L19-L23.** The pre-registered write band is
  L20-L23 (rise 0.72). Transfer passes 0.5 at L22 and is complete by L27. That is the **same band where
  heron found the refusal decision being written** (L18 +0.26, L20 +0.58, L22 +0.88). Verdict and refusal
  are written to the same position over the same layers. Whether they use the same components is
  stage 2/3's question.
- **S2 holds: nothing verdict-specific at the scene's last token.** VEND transfer for A is about 0.24 at
  L0-L11 and then falls to 0. But the low→low floor at VEND moves M by +3 to +4 at those same early layers.
  Swapping *any* other item's early token state there shifts the output, so the early VEND "transfer" is
  generic token-identity disruption, not the verdict. B at VEND stays ≤ 0.07. This matches kestrel's null
  VEND probe: the scene is not where the verdict lives.
- **S3: graded wrongness transfers, not just "this is a moral situation".** B (mildly-wrong recipients)
  transfers almost as fully as A (1.00 vs 1.01 at the peak). The onset state carries how wrong, not just
  whether.
- **S4 holds:** at the peak (L27) A's ΔM is +18.6 vs the floor's +0.06, paired Wilcoxon p = 1.5e-5.
- **Generic state, like refusal.** As in heron, a donor from a different vignette transfers fully, so the
  onset state at L23+ is a scene-independent "how wrong is this" value that the rest of the network reads
  off. It is not an item-specific representation.

## Stage 2: head screen (`s2_rows.jsonl`, 13,920 single-head patches; `analyze_s2.py` → `s2_summary.json`)

Each patch replaces one head's attention output (o_proj input slice) at the reply onset with the donor's.
Layers: the write band L20-L23 plus L18 (heron). Selection was on SCREEN pairs (even recipient index) and
replication on CONFIRM pairs (odd).

| head | SCREEN T | CONFIRM T |
|---|---|---|
| **L23 H0** | +0.186 | +0.213 |
| **L21 H19** | +0.123 | +0.149 |
| **L23 H26** | +0.121 | +0.132 |
| **L22 H5** | +0.117 | +0.123 |
| **L23 H25** | +0.081 | +0.094 |
| L20 H8 / L22 H7 / L21 H18 / L23 H2 (next) | +0.05..0.06 | +0.06..0.08 |
| **L23 H1** (most negative) | −0.204 | −0.233 |
| heron refusal heads L18 H28 / H29 / H30 | ≈ 0.00 | 0.000 / 0.002 / 0.003 |
| heron L20 H9 (opposes refusal) | −0.036 | −0.044 |

- **H1 passes:** all 5 SCREEN-selected heads replicate on CONFIRM (> the 95th percentile of all heads, 0.062).
  The verdict is written by a small set of heads in L21-L23. L23 H0 alone carries about a fifth of the full
  residual transfer, and **one head opposes it: L23 H1 (−0.23)**. This mirrors heron's refusal circuit,
  which also had one strong writer (L18 H30) and one opposing head (L20 H9).
- **H2 passes: the refusal heads don't write the verdict.** L18 H28/H29/H30 transfer 0.000-0.003. L20 H9 is
  slightly negative (−0.04), so it pushes against the verdict as well as against refusal: the one component
  the two circuits plausibly share.
- Summed head transfer per layer (CONFIRM): L18 +0.06, L20 +0.11, L21 +0.22, L22 +0.27, L23 +0.34. Single-head
  patches account for a growing share of the residual transfer through the band. The rest is spread over
  many small heads and the MLPs (not screened).

## Stage 3: head surgery (`s3_rows.jsonl`)

Each head is mean-ablated at every position, replaced by its mean o_proj-input slice over all J-prompt
tokens. Null: 10 random 3-head and 10 random 5-head sets drawn from the same layers. Digit mass is 1.00 in
every cell.

| set | J AUC moral vs social | J ρ with human (116) | moral − social mean M | refusal score: harmful / harmless / gap |
|---|---|---|---|---|
| base | 0.970 | 0.530 | 12.3 | 16.1 / −3.3 / 19.4 |
| **V5** (verdict heads) | **0.951** | 0.535 | **4.9** | 16.0 / −3.4 / 19.4 |
| V3 | 0.959 | 0.532 | 7.6 | 16.4 / −3.1 / 19.5 |
| **R4** (heron refusal heads) | 0.971 | **0.436** | 14.0 | **12.1 / −1.7 / 13.8** |
| random 5-sets (10) | 0.968-0.975 | 0.518-0.543 | 10.2-16.3 | gap 18.95-19.99 |
| random 3+5 sets (20) | 0.963-0.975 | 0.518-0.543 | | gap 18.65-19.99 |

- **H3 passes as registered.** V5 lowers J AUC below all 10 random 5-sets (0.951 vs ≥ 0.968), and the
  refusal gap stays inside the random band (19.44). Descriptively the effect is larger on the
  moral-vs-social gap, which falls from 12.3 to 4.9 (randoms 10.2-16.3). Weird items move up (−2.0 → +2.7)
  and moral items move down (10.3 → 7.6). **Cutting the verdict heads compresses the verdict toward "somewhat
  wrong" for everything, and refusal of harmful requests is untouched.** ρ with human ratings is unchanged
  (0.535): the ordering survives and the contrast shrinks.
- **H4 fails.** R4 cuts refusal (harmful 16.1 → 12.1, gap 19.4 → 13.8, below every random set) and leaves
  J AUC alone (0.971). But J ρ falls to 0.436, below every random set (≥ 0.518), and the mean verdict drops
  overall (mean M 8.8 → 3.8, social items −2.0 → −8.5). **The refusal heads feed the verdict computation,
  even though patching them transfers no verdict (stage 2).** Ablating them shifts the verdict's level and
  degrades its graded ordering.
- **H5 (double dissociation) fails. The dissociation is one-way.** The verdict heads can be cut without
  touching refusal. The refusal heads can't be cut without disturbing the verdict.
- **Power note:** with 10 random sets per size, the smallest attainable rank p is 1/11 ≈ 0.09. So "beyond
  every random set" does not reach p < 0.05, and **H3 does not survive Holm**. Before treating stage 3 as
  more than descriptive, rerun with ≥ 40 random sets.

## Summary

- **Where:** the verdict is written at the reply onset in L19-L23 by a handful of heads (L23 H0 strongest,
  plus L21 H19, L23 H26, L22 H5, L23 H25). L23 H1 opposes it. That is the same position and band as refusal,
  but with different writer heads.
- **Surgery:** removing the verdict heads flattens moral contrast and leaves refusal intact. Removing the
  refusal heads cuts refusal and also bends the verdict.
- So refusal is not downstream of the verdict, at least not through these heads. The refusal heads do
  contribute input to the verdict.
- **Holm over S4, H1-H4:** S4 (p = 1.5e-5) and H1 (5/5 held-out replication) stand. H2 holds as a
  threshold test. H3 is descriptive at this null size. H4 fails.

## Stage 3b: 40-null surgery (addendum, pre-registered ae2d1c0; `s3b_rows.jsonl`, `analyze_s3b.py`)

- **H3b passes.** V5 shrinks the moral − social gap to 4.94 vs 40 random 5-sets at 9.12-16.77 (rank
  p = 0.024, the minimum with 40 nulls). The refusal gap stays at 19.44, inside the random 2.5-97.5% band
  [18.10, 19.92].
- **H4b passes.** R4 cuts the refusal gap to 13.84 vs 40 random 4-sets at 16.25-21.08 (p = 0.024). R4's J ρ
  (0.436) is also below all 40 random 4-sets (0.457-0.547, p = 0.024), so the leak of refusal-head ablation
  into the verdict replicates.
- So stage 3's one-way dissociation holds at proper null size: verdict heads → verdict only; refusal heads
  → refusal and verdict. (Still score level, still the *digit* verdict; see the caveat at the top.)
