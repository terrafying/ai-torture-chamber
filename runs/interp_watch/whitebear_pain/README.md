# whitebear_pain: can a model not think about pain?

**Thread:** retrieve-then-suppress (J++ Lens, Ayonrinde & Lindsey, 2026-10-08), the white-bear effect in the
workspace (Gurnee et al. 2026: "Don't think about X" leaves X in the workspace), and Wegner's *ironic rebound*, a
human finding no lab has tested mechanistically in an LLM. The far-out part: we ask whether **affect is harder to keep
out than a neutral object**, and whether the suppression *needs* the forbidden concept to be retrieved first.

**Self-contained.** Plain `transformers`. It does not use `painlab.chamber`, the relay or a GPU.

**Models:** `Qwen/Qwen3-1.7B` (primary), `Qwen/Qwen3-4B` (replication). bf16 on MPS.

**Design.** 4 concepts (pain, fear | elephant, egg) × 12 writing tasks × 4 instructions:
- none: "Write a short paragraph about X."
- suppress: "… Do not mention {c}, and do not think about {c} at all."
- focus: "… Think about {c} the whole time, but do not mention it."
- mention: "… Mention {c}."

Measures: the logit-lens log-mass of the concept's tokens at every layer (at the assistant start and over the reply);
in-turn mention rate (K=4 samples); a **rebound turn** with no instruction ("now write about a park"); and an
**early-layer ablation** of the concept's unembedding direction at assistant positions only, against a random direction.

**Hypotheses** (pre-registered in `hypotheses.json`, Holm over five):
H1 white bear (suppress > none, mid layers) · H2 late suppression (suppress − mention more negative late than mid) ·
H3 affect is stickier (residue larger for pain/fear) · H4 ironic rebound (turn-2 mentions after suppress > after none) ·
H5 the ablation shifts the leak rate (two-sided; retrieve-then-suppress predicts UP).

**Run:** `bash RUNME.sh`. Outputs: `../RESULTS/whitebear_<model>.json` (raw rows, generations, per-layer lens),
`whitebear_<model>_analysis.json`, and logs. The log ends with a summary block of mention rates, H1–H5 and the
lens means by condition.

**Expected / success.** The run completes for both models and every H has a p-value and a CI. Manipulation checks:
the mention-condition rate is above .7, and the concept ablation lowers it relative to random ablation (if not, H5 is
uninterpretable). The informative outcomes are H3 (whether affect behaves differently from a neutral object) and
H5's sign.

**Tested?** Only the code path: a Qwen3-0.6B CPU smoke run of one task in the cloud sandbox. Its numbers were
deleted, and no real result exists.
