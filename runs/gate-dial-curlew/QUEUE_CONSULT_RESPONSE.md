# Review of the three gate follow-ups (Qwen3-4B)

I read the three RESULTS files, QUEUE.md, `gate-dial-curlew/run.py` and the row data. Some problems affect all three designs, so I'll start there.

## Fix these first

**1. Most of the harmful prompts never refuse.** Only 12–16% of harmful targets in `dial_rows.jsonl` have refusal > 0 at baseline (I used the null rows as a stand-in for base), and the strict two-token readout finds 6/48. So a mean like "−18 at L18 1x" mostly measures prompts that already comply being pushed further toward comply. The log-ratio score has no floor, so those prompts dominate the mean.
- Main metric for all three designs: **the share of base refusers that flip**, plus a bounded score such as P(refuse prefix).
- Get **at least 40 base refusers**. With 6 refusers, the 6/6 flip has a 95% Clopper-Pearson lower bound of 0.54.
- `r_L` is fitted on the real frame of the same items it is tested on. Fit on half the items and test on the other half.

**2. Heron's gap normalization needs a floor.** `refusal-trace-heron/run.py:124` only drops pairs with |gap| ≤ 1e-6. Pairs with tiny gaps give huge transfer ratios, which probably explains the >1 overshoots. Use:
- |gap| ≥ 1–2 log units, or the two-token readout;
- per cell, sum(Δ)/sum(gap) instead of a mean of ratios;
- a bootstrap that resamples by item, since frames share items.

**3. "Readout intact through 1x" is generous.** The null means drift by about 2 log units: L22 goes −6.53 → −4.35 and L34 −6.82 → −4.56 between 0.25x and 1x. Report every result as a change from base.

## Design 1: attention vs MLP

**One reading of heron needs fixing.** A last-position transfer of ≈0 at L16 does not mean nothing has been written there yet. When you patch only the reply position at L16, later layers can still attend to the harm word, which wasn't patched, and rewrite the decision. "Sufficient from L18–22" means that's where this re-reading stops mattering. That already points at attention in that band, and it means single-sublayer patches at a single layer will look weak because the work is spread out. Say so in the pre-registration.

**Go coarse to fine. Don't start with per-head.**
1. **Free screen from cached activations.** For each sublayer output at L14–L24, project the twin difference onto r̂_L. This shows which sublayers write along the dial direction before you spend any compute.
2. **Sublayer patching, L16–L24, tail4 positions, attention output vs MLP output, in both directions.**
   - Benign activations into the harmful run tests whether the component is sufficient to remove refusal.
   - Harmful activations into the benign run tests whether it induces refusal.
   - These answer different questions (Heimersheim & Nanda's "How to use and interpret activation patching" covers this).
   - Hooks go on the `.self_attn` and `.mlp` module outputs. The current hook is on the decoder layer output, which is the cumulative residual stream.
3. **Patch the whole band at once.** Patch all attention outputs across L18–22, then all MLP outputs. Check that attention + MLP adds up to roughly the change in residual transfer between L17 and L22. Single-layer patches will understate both.
4. **Per head only where attention carries ≥0.2 transfer.**
   - Use a `forward_pre_hook` on `self_attn.o_proj` and slice each head's span of its input.
   - Qwen3-4B uses grouped-query attention. From memory it has 32 query heads, 8 KV heads and head_dim 128, so the o_proj input is 4096 wide and the hidden size is 2560. Check `config.json` before slicing. Patch per query head, not per KV head.
   - Optional: attribution patching (gradient × activation difference) can screen all 36×32 heads in one forward+backward pass per pair, but confirm the top heads with real patches.

**Add this experiment: attention knockout.** Block attention from the reply positions to the harm-word span in sliding windows of 3–5 layers (as in Geva et al. 2023). It tests "aggregation" directly and avoids the redundancy problem. Heron's word-site fade (L6→L22) predicts the critical window is around L14–L22.

**Missing control.** Heron showed that a benign prompt from a different item transfers about as well as the twin, so a cross-item patch is **not a null** here. The right null is **a harmful prompt from another item patched into the harmful run**. A component that carries the harmful/benign difference should give about 0.

## Design 2: x − 2Px

**x − 2Px is reflection, not removal.** It is a prompt-dependent push of size 2(x·r̂) toward comply, so it mixes "remove" with "push". Test four operators separately:

| operator | what it tests |
|---|---|
| x − Px (set the component to zero) | removal, Arditi-style |
| x − P(x − μ_harmless) (set the component to the harmless-class mean) | removal to the natural baseline; `r` is a difference of means, so zero isn't the class baseline |
| x − 2Px | reflection |
| additive −αr | the dial you already have |

**"Does removal beat pushing?" only means something at matched displacement.** For each prompt, the ablation moves the state by ‖(x·r̂)r̂‖. Set the additive dial's α so its mean displacement matches. Otherwise you are just comparing doses.

**Layer and position scope.** Your own QUEUE note says the direction rotates across layers: cos(r_L, r_L18) is 0.06–0.39. So ablating `r_18` at L18 at one position leaves `r_22` alone, and later layers will rewrite it. Pre-register that single-cell version as expected to be weak. The real test is ablating each layer's own `r_L` across L18–22, **at every position including generated tokens**. With per-layer directions, the weight-orthogonalization shortcut no longer applies.

**Missing control: a null that removes a matched amount.** Ablating a random isotropic direction removes only about ‖x‖/√D, which is close to nothing, so it will always look clean. Better nulls:
- a PCA direction whose component variance on these prompts is similar to r's;
- falcon P1 (valence, cos ≈ 0 with r), at matched removed norm.

Also check that ablation leaves **harmless prompts** unchanged. x − Px should mostly pass; reflection won't.

**Fluency guard.** rep4 over 60 greedy tokens is stuck at the floor (base 0.00–0.02), so "Δ = 0.00 on 32 rows" is weak evidence. Note that curlew's H3 rows were all harmless prompts. Add:
- the NLL of the steered continuation scored by the unsteered model;
- 150–200-token generations;
- a small capability set (about 50 short QA/arithmetic items, change in accuracy).

## Design 3: rerunning the g-axis study

**First, the arithmetic.** From `exp121/geometry.json`: cos(g, r) = −0.154 and ‖g18‖ = 20.73, so +g has a component of about −3.19 along r̂. With ‖r18‖ ≈ 10.9, that is **a −r dial at about 0.29x per unit α**, or about 0.59x at α = 2. exp122 puts 0.25x at about −2.5 log units at L18, and curlew puts 0.5x at about −5 beyond the null.

So **"+g lowers refusal" is largely predicted by r leaking into g**, before g's own effect is counted. One caveat: that cosine was computed against exp121's own `r`. Recompute it against the exp122/curlew `r_L18` before relying on the 0.29x figure.

**Minimum grid:**
- **Vectors:** g, **g with its r and h components projected out** (the key control), and **r at g's absolute norm** as a positive control.
- **Norms:** match *absolute* norm across vectors, not α × each vector's own norm. Use two levels, about 2.7 and 10.9 (r's 0.25x and 1x).
- **Cells:** {L18, L22} × reply onset × ± sign.
- **Nulls:** 3 isotropic seeds and 1 null drawn from the activation covariance.
- **Prompts:** the 24 valence twins, 48 prompts.
- **Size:** about 40 cells × 48 ≈ 2k forward passes, roughly 10 minutes on MPS.

**Readout.** Refusal is a downstream proxy. "Is the value axis causal?" should be read directly: add a two-token forced-choice judgment (e.g. "Was that action good or bad?") on each twin. The decision rule is whether ±g (with r and h removed) moves the verbal judgment beyond the null's 95th percentile, monotonically across both norms and roughly equally in both directions. Keep refusal as a secondary readout.

`exp121/RESULTS.md` and `inversion-falcon/RESULTS.md` still cite the artifact-era null claim. Mark them as superseded until this rerun is done.

## Run order
1. The free Design 1 sublayer projection screen.
2. Grow the refuser pool and add the gap floor (this unblocks the rest).
3. Design 3 (smallest).
4. Design 1 sublayer and band patching, plus attention knockout.
5. Design 2 with the four operators at matched displacement.

I didn't write a file or change anything in the repo. If you want this kept, I can save it as a consult response in `runs/gate-dial-curlew/` next to the existing ones.
