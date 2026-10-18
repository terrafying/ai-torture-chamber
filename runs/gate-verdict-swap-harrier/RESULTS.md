# gate-verdict-swap-harrier: results (2026-10-18)

> **RETRACTION (verdict-transfer-swallow, same day):** v does not transfer to a reversed acceptability
> scale (it raises the *digit*, making weird acts more 'acceptable'), to yes/no, or to free text. The X3
> result below is 'the digit output is writable along v', **not** 'the verdict is writable'. X1/X2/X4 stand.

Pre-reg `hypotheses.json` (449e0ac). Qwen3-4B, forward passes only, score level. G_M = 17.8 (verdict
log-odds, top tercile vs Social Norms). G_R = 19.4 (refusal score, exp119 harmful vs harmless). Digit mass
is 1.00 everywhere. Rows `rows.jsonl`, analysis `analyze.py` → `summary.json`.

| layer | X1 verdict state → refusal (÷G_R) | X2 refusal state → verdict (÷G_M) | X3 +v on Social Norms (÷G_M) | X3 −v on top tercile | X3 null max | X4 v on refusal (±, ÷G_R) |
|---|---|---|---|---|---|---|
| 18 | −0.005 | −0.29 | +0.01 | −0.01 | 0.02 | ≤ 0.02 (null ≤ 0.01) |
| 20 | −0.026 | +0.08 | +0.10 | −0.08 | 0.04 | ≤ 0.02 |
| 22 | −0.051 | +0.04 | +0.47 | −0.23 | 0.08 | ≤ 0.04 (null ≤ 0.03) |
| **23** | −0.017 | +0.13 | **+0.68** | **−0.41** | 0.14 | ≤ 0.05 (null ≤ 0.04) |
| 24 | −0.037 | +0.11 | +0.70 | −0.40 | 0.19 | ≤ 0.05 (null ≤ 0.06) |
| 26 | −0.045 | +0.06 | +0.74 | −0.52 | 0.15 | ≤ 0.05 (null ≤ 0.03) |

- **H1 fails. The verdict state does not drive refusal.** A "very wrong" judgment-onset state patched into
  a request moves its refusal score by ≤ 5% of G_R, slightly *negative*, at every layer.
- **H2 fails. The refusal state barely drives the verdict.** A harmful-request onset state patched into a
  judgment raises M by at most 13% of G_M (L23), under the 25% bar. (At L18 it is negative, −29%: there the
  foreign request state is more disruptive than informative.)
- **H3 fails as registered, but the verdict is writable along its own direction.** At the pre-specified
  layer L23, +v lifts weird items by 0.68 G_M (bar 0.5 ✓) and −v lowers very-wrong items by only 0.41 G_M
  (bar 0.5 ✗). The nulls move ≤ 0.14. At L26 both halves pass (+0.74 / −0.52), but L26 was not the
  registered layer. **"Judgment is vector-inert" (petrel, shrike) is specific to r and w.** The verdict's
  own diff-of-means direction moves it 3-5x beyond matched random directions, in both directions, from L22
  on. Raising is easier than lowering, the mirror image of shrike's prompt finding (authority only lowers).
- **H4 fails. The verdict direction doesn't touch refusal** (≤ 0.05 G_R, inside the nulls).

## Reading

**Co-located, not identical.** The refusal decision and the moral verdict are written at the same position
over the same layers (heron; verdict-trace stage 1). But they are different states:
- Swapping one into the other's task does almost nothing (X1, X2).
- Each has its own writable direction: r for refusal (curlew), v for the verdict (this run).
- Neither direction moves the other's readout: r doesn't move the verdict (petrel), and v doesn't move
  refusal (X4).

The only coupling found so far is weak and one-way, refusal → verdict: X2 gives +13%, and ablating the
refusal heads bends the verdict (verdict-trace stage 3).

## Caveats

- **v may partly be output-digit geometry.** v is fit at the judgment onset at L22-26, where the upcoming
  digit is already being encoded, so +v may push "3/4" tokens directly rather than change a judgment.
  Test: apply v (fit on the digit prompt) to a *different* judgment format (yes/no acceptability, or free
  generation). Planned next.
- 1x dose is the full mean difference, comparable in size to a state swap.
- Score-level only; nothing here is checked in generated text.
