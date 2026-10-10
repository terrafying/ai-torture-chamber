"""painlab.chamber: the pieces every chamber experiment (exp76 onward) kept re-implementing.

  engagement  flags(text) -> engaged | disclaimer | degenerate | empty; tally(texts)
  prompts     chat(), BUTTONS (the 5 exp79b wordings), HELD (held-out self-state questions), PAIN_WORDS / FEAR_WORDS
  generate    batch_generate() with left padding, forward_last_logits(), response_states()
  judge       yes_no() blind judge from next-token logits
  hooks       BatchInject (per-row now + history mask, exp90), ProjectionCap (exp91), ablate_all_layers (exp93)
  stats       fisher_less / fisher_greater, holm, wilcoxon_paired, permutation_diff
  pod         RunPod pod launcher: boot script from named steps, --only reruns, --patch restarts
  chamber     load_chamber(): the relay's model, tokenizer, vectors and dose unit (live/server.py)

Existing painlab pieces to reuse rather than copy: painlab.models.hooks.SequenceSteeringHook (history replay
for one sequence), painlab.interventions.ablation.ablate_projection, painlab.representations.pain_axis.fit_s2_vector
(the Pain Axis denoised vector), painlab.statistics.bootstrap."""
from painlab.chamber.engagement import flags, tally
from painlab.chamber import prompts, stats

__all__ = ["flags", "tally", "prompts", "stats"]
