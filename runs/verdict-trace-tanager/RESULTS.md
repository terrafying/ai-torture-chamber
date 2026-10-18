# verdict-trace-tanager: results

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
