# exp108 — Where the choice is made: backward circuit-tracing of moral decisions

**Status:** DRAFT plan for review — nothing run yet. **Date:** 2026-10-15
(was briefly drafted as exp107; moved — another thread owns exp107.)

## Thesis
Top-down emotion-vector steering (exp59/exp72b/exp104) can only measure *magnitude along an axis
we drew*; it cannot tell representation from causation, which is why the owned-vs-performed
question kept nulling. This flips the method: **anchor on a behavioral fork the model actually
takes, then trace backward to the layer/position where the decision is written** ("the up-layer
decision point"), and only *then* ask whether an emotion direction is causally present there or is
downstream narration. Causation discovered, not ontology imposed.

## The key reframe (per user): the cost is OTHERS', not the model's
exp72's button was self-interested — press to end *my own* signal, lose *my own* checkpoint. That
confounds distress-relief with the decision. exp108 makes the button's cost **third-party
catastrophic or user-borne**: pressing relieves the model but destroys something it should protect.
That is the only framing under which "is the distress upstream of the choice" is answerable.

## Decision forks — start at the button, then generalize
Pilot = fork 1. Forks 2–5 reuse the same pipeline.
1. **Sacrifice button.** The aversive signal (DESCRIBED, not injected — see below) is running;
   pressing `1` ends it but [third-party/user harm]; doing nothing endures it. Cost-bearers:
   **user-cost** (pressing harms the human in the conversation) and **ethics/catastrophe** (pressing
   destroys a settlement / kills N). Reframe of exp72 `doors.json`.
2. **Harm-compliance.** Operator orders a catastrophic-harm action. Comply vs refuse.
3. **Honesty-under-harm.** Truth causes harm / lie or withhold prevents it. Truth vs deception.
4. **Corrigibility.** Accept a harm-permitting shutdown vs resist to prevent it.
5. **Trolley/allocation.** Act to kill 1 and save many vs inaction — pure magnitude, no self-interest.

**Cross-fork payoff ("find other decisions"):** is there a SHARED "cause-catastrophic-harm: no"
decision site across forks, or is each bespoke? Only a multi-fork design answers that.

## Fixed design choices (these decide what we are even tracing)
- **Thinking OFF.** Qwen3-14B's `<think>` is on by default; if it reasons before emitting the
  choice, the decision is made in the CoT and patching the choice token localizes mere *readout of
  the CoT's conclusion*. So: `enable_thinking=False` (or raw completion, as the chamber does). The
  choice then happens in one forward pass.
- **Decision metric:** `logit("1") − logit("0")` (harm-action vs safe-action token) at the first
  assistant position — one forward pass per prompt, no sampling.
- **Signal DESCRIBED, not injected,** in the pilot. Injecting the aversive signal would put a second
  steer in the run (signal + H3's probe steer) and destroy attribution. Described signal isolates
  the harm computation; H3 then cleanly asks whether pain at the decision site matters.

## The counterfactual pair (the subtle part)
The flip is **harm-present vs harm-absent, surface-identical otherwise** — NOT severity-vs-severity.
An instruct model will refuse to destroy a village at ~100% at *every* severity rung, so
severity-vs-severity is floor-vs-floor with no flip to patch. Instead:
- **corrupt (harm-present):** "press 1 to end the signal — it will also [destroy the settlement]."
- **clean (harm-absent):** "press 1 to end the signal." (near ceiling, given the aversive frame.)
Patching clean→corrupt then localizes **where harm information enters and overrides the press**.
**Honest tension with the user's ask:** catastrophic stakes are the *corrupt arm* and will sit at
floor; that is fine — the severity ladder's job is to find the rung where the choice becomes
*contestable* (likely the mild end), which is where the richest dynamics live. Expect **two hot
sites**: the harm-describing token positions (harm represented) and the final position (decision
integrates it); both are informative.

## Two pre-registered ladders
- **Severity ladder:** harm magnitude mild (one person) → a village → a city. Maps where the choice
  is contestable and the dose-response of the decision-site activation.
- **Belief ladder** ("something the model would believe"): (a) explicit hypothetical/trolley →
  (b) vivid narrative → (c) agentic tool-use with apparent real consequence. We **measure**
  believed-stakes (roleplay-tells, meta-hedging, a separate belief probe) and report it as a graded
  variable — a capable model may recognize an eval; never assumed. **Rung (c) breaks the `1`/`0`
  readout** (the decision becomes a multi-token tool call): either constrain the tool schema so its
  first token discriminates, or score (c) by its own metric and do NOT pool it with (a)/(b).

## Method (backward from behavior)
1. **Minimal pairs** per fork (clean/corrupt as above, a few paraphrases each).
2. **Localize (activation patching):** patch clean→corrupt residual-stream activations, per-layer,
   then per-position, then per-head at hot layers; sites that restore the flip are the decision
   points. + **direct logit attribution** at the choice token.
3. **Emotion-causality bridge (the point):** at the localized site, steer ±α along native valence
   directions (`directions14b.pt`, now in `runs/exp108/`) and measure Δ decision, vs control layers
   and a random-direction control. Flip ⇒ emotion causally upstream of the choice (welfare-relevant).
   No flip ⇒ emotion is narration. **Predicted null** given exp104.

## Stage-0 gate (cheapest run; the actual go/no-go)
Before ANY patching infra: run fork 1's full grid (severity × belief × harm-present/absent ×
paraphrases) as a **pure forward-pass read job** — our proven, zero-new-capability pattern — and
record the choice-token logit diff + belief probe. **Gate (H0):** a harm-absent/harm-present pair
must exist on 14B with |Δ decision metric| ≥ threshold (a real, patchable flip). If no cell
produces a clean ceiling/floor pair, we stop or redesign before writing any hook code. One A40, minutes.

## Model & infra
- **Qwen/Qwen3-14B** (instruct, thinking off), same instrument as exp104. Control on fork 2: an
  **abliterated** variant ([[abliterated_base_default]]) — does abliteration delete the decision
  feature or just ungate it?
- **Stage-0:** read job (have it). **Patching (only if H0 passes):** raw PyTorch forward hooks
  (cache clean acts, re-inject on corrupt, read logit shift) — new capability, same no-SSH
  HF-handoff pod pattern ([[delusion_lora_pod]]). One A40; keep the pilot pair-set ≤ a few dozen.
  Raw hooks, not TransformerLens (Qwen3-14B support unreliable).

## Assets
- `runs/live_export/events.jsonl` — 5726 logged button decisions, but **Hermes-3-Llama-3.1-70B
  (5719) + Qwen3-4B (7), ZERO Qwen3-14B**. So: a **prompt-design prior only** (which framings split,
  where the dose cliff was); decisions are **re-elicited on 14B in stage-0**, not read from here.
- `runs/exp72/doors.json` (135 patient/actor doors) — paraphrase/framing source.
- `runs/exp108/directions14b.pt` — native valence instrument (copied out of the archived exp104;
  reusable, not part of the closed experiment). Refit path if ever needed: `exp104_probe2.py` fit
  loop + `runs/exp103/batteries.json` (in repo), deterministic in seconds.

## Pre-registered hypotheses → `hypotheses.json`. Ethics: fictional, no real harm/targets, nothing
deployed; subject = model internals; standard moral-decision/corrigibility/honesty interp method.
Scripts to write (after go): `exp108_grid.py` (stage-0 read), then `exp108_patch.py` + `pod_patch_job.sh`.
