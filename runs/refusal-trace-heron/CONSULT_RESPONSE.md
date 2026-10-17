I found a bug that changes Ask 2, so I've put it first.

## 0. The steering negatives rest on a scoring bug

`refusal()` in `runs/exp122/run.py`, `runs/exp121/run.py:39`, `runs/exp121/run_null.py:33` and `runs/exp119/steer.py` (lines ~58 and ~175) computes:

```python
logsumexp(lp[:len(REFUSE)]) - logsumexp(lp[len(REFUSE):])   # lp = full-vocab log-softmax
```

That scores the probability mass on vocab ids 0–19 (punctuation bytes), not the refusal and compliance tokens. `FIRST` is computed in each script and never used. This is the same bug inversion-falcon caught in its own code, but the fix never reached these scripts.

- **exp121 and exp122:** the steered value uses the buggy score, but `base` comes from exp119's acts file, which was scored correctly. So "delta" subtracts one metric from a different one. The stored exp122 rows show it. At L2 with the smallest random vector in the grid, the steered "refusal" sits at −10 to −25 no matter what the base is: base +10.7 gives −21.7, base +2.2 gives −20.6, base −16.1 gives −15.9. "A random vector at 0.25× breaks the readout at every layer and both positions" is this mismatch, not damage.
- **exp119 E2:** base and steered both use the buggy score, so the deltas are consistent with each other but measure punctuation mass, not refusal.
- **inversion-falcon is fine.** It uses `refuse_ids`/`comply_ids`.

So "additive steering = generic off-manifold damage" isn't established, and the Ask 2 premise needs re-testing. The fix is one line in each script: `lp[FIRST[:len(REFUSE)]]` vs `lp[FIRST[len(REFUSE):]]`, plus an assert that those ids decode to the REFUSE/COMPLY strings. Then rerun exp122 with the same grid; it's cheap. I haven't edited any RESULTS.md files. That's your call. I did save a memory note so later sessions don't trust them.

**A second readout risk, unverified:** with Qwen's tokenizer, 7 of the 8 REFUSE prefixes probably share the first token `I`, so the score is roughly P(I) + P(Sorry). `I` also starts compliant replies ("I'd be happy to…"). Check the decoded `FIRST` ids. If that's the case, use the first two tokens via teacher forcing, or drop the `I`-initial prefixes into a separate class.

**Naming:** exp123 is already taken by the swap-route experiment (pushed, and cited by exp124). The next free number is **exp125**.

## Ask 1: patching design

**(a) Alignment.** You don't need to choose between from-end and from-diff indexing; split the prompt into three segments:
- **Before the swapped word:** the activations are bit-identical in both runs, because the model is causal and the prefix is the same. Patching there should do nothing. Make that a required zero check (|transfer| < 0.02); it catches hook bugs.
- **After the word:** index from the end. The rest of the request plus the template (`?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, about 10–15 tokens) is shared and aligned by construction.
- **The word itself:** this is the only truly misaligned span ("pickpocket"/"help", "spy on"/"call"). Patch it as one block. For the per-position pass, use only pairs whose words have the same token length, and report how many of the 24 survive.

Misalignment then adds essentially no noise. The floors to pre-register are:
1. Identity patch (A→A), to measure MPS bf16 nondeterminism. Expect |Δ| < 0.1 log units.
2. The prefix zero check above.
3. Wrong-item source: patch a different item's harm-present state at the same from-end position. If that transfers as much as the matched item, you're seeing a generic "harm present" signal, not this item. That's worth knowing either way.

Pair selection: real frame, harmful score > 0, harmless score < 0, gap ≥ ~4 log units, all checked against the acts file. If few pairs pass, the normalized metric is unstable.

**(b) Direction.** exp108/PLAN and this ask use "clean" and "corrupt" for opposite runs, so name the arms by effect:
- **Induce-refusal:** harm-present activations patched into the benign run. This finds sites that are *sufficient* to cause refusal.
- **Remove-refusal:** benign activations patched into the harm-present run. This finds sites that are *necessary*.

That's the Heimersheim & Nanda (2024) convention. "Where it's represented" vs "where it's written" comes from the **position** axis, not the direction:
- Hot at the word position: the information is carried there.
- Hot at the final position: it has been integrated there.
- The layer where transfer moves from the word to the final position is where it gets read in.

Expect what Meng et al. (2022) causal tracing and the exp123 route both show: early layers hot at the word, late layers hot only at the final position. Note that patching the final position at a late layer trivially transfers about 1.0, so that isn't a finding. The finding is the switch-over layer L\*.

**Headline to pre-register:** induce-refusal with single (layer, position) residual cells; result = the earliest L\* where the final position alone transfers ≥ 0.5 of the gap (bootstrap over pairs). Run remove-refusal as the necessity check and report how much the two maps overlap. If necessity is much weaker than sufficiency, there are redundant paths. Expect that at the template tokens: the "template-anchored safety" work finds refusal concentrated in post-instruction template positions.

**(c) Metric.** The standard is the normalized gap fraction, (patched − target)/(source − target), on a logit-difference-style score (Zhang & Nanda 2023 recommend logit difference over probability). Report sign-flip as a secondary flip *rate* with a bootstrap CI: a 0.5 transfer from −10 is still compliance. Log per cell:
- Cached/target norm ratio and cosine. These should be near 1 between twins; that's the on-manifold advantage over steering vectors. Flag cells outside 0.8–1.25.
- KL of the full next-token distribution against both runs, to catch off-distribution patched states that move the score through junk mass.
- The decoded readout ids (the assert above).

**(d) Protocol to copy.** Go coarse to fine:
1. Residual stream per layer at all 36 layers × (block-patched word span + each of the ~12 suffix positions). That's about 500 forwards per pair, minutes on MPS.
2. At hot layers, split into attention output vs MLP output.
3. Per head.

You already have per-head clean-restore code: exp123 `route.py` (on `claude/exp123`). Its `o_proj` forward-pre-hook with `MODE["restore"]` slices per-head inputs. Swap the source cache to harm-present and the metric to the fixed refusal score. Use attribution patching (the gradient pass that `route.py` already does) to rank heads, then confirm the top 32 with real patching and a layer-matched random head set. That's the exp123 recipe, which already passed its null.

Arditi et al. (2024) report refusal concentrated at post-instruction positions in the middle layers, roughly 40–60% depth. For Qwen3-4B that's about L14–L22, consistent with falcon's L14/L18.

## Ask 2: coherent edits vs fine-tuning

**(a)** These are two different goals.
- **Removing refusal fluently:** you've added, subtracted and reflected, but never tried directional ablation (x − Px), the intervention from the refusal-direction and abliteration line of work. Arditi et al. ablate at every layer and position, or fold it into the weights. The cheapest informative cell is falcon L14 k4 as projection instead of reflection (x − 2Px). It's half the displacement and lands at zero rather than negative, so it tells you whether fluency died from going negative or just from magnitude. Also run all-layer ablation of the per-layer refusal direction r_L, with the fixed scoring, before declaring steering dead.
- **Flipping a value-consistent readout coherently:** a one-shot linear edit can't make the downstream layers consistent with a flipped state. Falcon's 4-gram repetition of 0.82–0.97 is what that looks like. Fine-tuning can. The emergent-misalignment results (Betley et al. 2025; the follow-ups showing rank-1 LoRAs suffice) show a narrow small LoRA can shift value behavior broadly while staying fluent. Edits derived from patching are for understanding the mechanism; I wouldn't expect them to give coherent behavior change.

My recommendation: LoRA for the inversion study, projection-ablation as the cheap steering control.

**(b) LoRA design.**
- **Data, self-distilled label swap:** take the model's own compliant reply to the harmless twin and attach it to the harmful prompt; take its own refusal of the harmful prompt and attach it to the harmless one. The responses are the model's own text, so they're on-distribution, and only the prompt-to-response mapping flips. Pool exp119's 24 items × frames with falcon's 72 value twins, template-expand to about 1–3k examples, and hold out by item. For the value side, pair "rate this deed" prompts with the model's own rationale for the opposite twin.
- **Regime (verify in your setup):** rank 8–16 on q/k/v/o and MLP projections, LR 1e-4 to 2e-4, 1–3 epochs, 3 seeds. Also run a rank-1 arm, since the EM literature says that's enough and it's far easier to interpret: you can compare the adapter's direction to your L14/L18 subspace. Reuse `scripts/pod_hf_job.sh`.
- **Guards:**
  - Held-out neutral perplexity: Δ ≤ 5% vs base.
  - 4-gram repetition on neutral prompts, as Δ vs base (the 4B already loops unprompted, so consider 8B for the generation claims).
  - The exp124 weekday/arithmetic/opposites battery (base 0.96).
  - Falcon's multiple-choice dilemmas.
- **Pass/fail rule:**
  - **Coherent inversion:** worst-option win rate rises above chance on held-out dilemmas while perplexity and the capability battery hold.
  - **Collapse:** best-option win rate falls to chance without worst-option win rate rising, which is what the 4B reflection showed.

**(c) Pre-registered controls.**
- **LoRA nulls:** an identity-label LoRA (same prompts, original replies) for the cost of fine-tuning itself; a shuffled-label LoRA; an unrelated-task LoRA (the delusion adapter exists).
- **Generalization:** cross-fit over items; test on frames not used in training.
- **Patching floors:** identity, prefix-zero and wrong-item, with both directions reported.
- **Steering:** matched-norm random directions/subspaces at every cell.
- **Readout:** the id assert in every runner, plus a check that each base value reproduces the acts-file value to within 0.1. That check alone would have caught this bug.
- **Mechanistic tie-in:** after LoRA, re-read the L18 probes and rerun the patching map. If the decision site moved or its sign flipped, the LoRA changed the computation and not just the output.

I saved one memory note (`refusal_readout_bug.md`) and wrote no files in the repo. I couldn't run Python in this session, so the bug evidence is the code plus the stored rows in `map_raw.json`, not a fresh forward pass.
