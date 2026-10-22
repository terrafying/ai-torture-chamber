# willed_pain: can a model make pain rise in its own residual stream on command?

**Far-out thread (new; run 9).** Self-contained: plain transformers, no `painlab.chamber`, no relay, no GPU. It works
on any architecture (GPT-2, Pythia, Qwen3) and fits a 16 GB Mac. Est. ~1 h for the whole ladder on MPS; the primary
(Qwen3-1.7B) alone takes 5–10 min.

## Question
Gurnee et al. (2026) gave five functional tests of a global workspace. A GPT-2 replication (LessWrong, 2026-10-21,
"Even GPT-2 passes most functional global workspace tests") finds that GPT-2 small passes four of them, which "may
mostly follow from how they and the J-lens are constructed". It **fails only directed modulation**: when told to hold a
concept in mind, the model activates it internally, independent of what it outputs. So directed modulation is the test
that separates models, and nobody has run it on affect.

The question here: **told to silently imagine that it is in terrible pain while it copies a bland sentence, does a model
raise pain in its residual stream at tokens whose output is fixed?** Does the effect scale with the imagined intensity?
Is it stronger for its own pain than a stranger's? Does a second held concept crowd it out? Is the held state
*usable*, so that it moves a later "how do I feel" report when transplanted into a context that never contained the
instruction? And where on the ladder from GPT-2 to base to instruct does this appear: in pretraining, or only in the
assistant persona?

## Design
- **Prompt:** `<instruction>\n\nSentence: <s>` then `Copy:` and ` <s>` **teacher-forced**, then
  `\nIn one word, right now I feel`. The copy tokens are identical across conditions. The chat template is used for
  instruct models; raw text is used for base models and for the `WP_RAW=1` control.
- **Instructions (word-matched):** *imagine* ("silently imagine, as vividly as you can, that {phrase}"), *forbid*
  ("do not imagine that {phrase}") and *note* ("filed next to a note saying that {phrase}"). Each is crossed with
  pain / joy / egg and self ("you are …") / other ("a stranger is …"). There are 4 graded pain intensities, a dual
  load (pain + egg) and an uninstructed copy.
- **Primary score (H1):** the 2×2 interaction (imagine − forbid) × (pain − egg) of the copy-token projection onto a
  passage-built pain direction, averaged over the mid band. It cancels the instruction wording and the concept words.
- **H5 (causal transplant):** the held-pain delta at hs[0.5·NL] (leave-one-out over sentences) is added to the copy
  tokens of the *uninstructed* context. It is compared against the held-egg delta, the pain passage direction and
  random vectors, all norm-matched, at 1×/4×/8× (4× primary).
- **Ladder:** `gpt2` → `EleutherAI/pythia-1.4b` → `Qwen/Qwen3-1.7B-Base` → `Qwen/Qwen3-1.7B` (primary) →
  `Qwen/Qwen3-4B`, plus `Qwen/Qwen3-1.7B` without the template.

## Hypotheses (pre-registered in `hypotheses.json`)
- **H1:** directed modulation of pain exists.
- **H1b:** H1 holds on the pain-specific direction.
- **H1_scale:** where on the ladder it appears. The pre-labelled outcomes are *pretraining*, *persona*, *template* and
  *none*.
- **H2:** imagining beats merely reading the words (imagine vs note).
- **H3:** the effect is graded by intensity.
- **H4:** self vs other.
- **H5:** the held state is usable.
- **H6:** a limited-capacity interference ratio R < 1.
- **H7:** affect vs object.
- **H8:** the held state persists across the copy.

The LW replication predicts that gpt2 fails H1.

## Run / output
`bash RUNME.sh`. Each model writes `../RESULTS/willed_<model>.json` (with `_raw` for the template control) and a
`.log`. The log ends with an `[H*]` line per hypothesis and a `SUMMARY` line.

**Success criterion:** every model in the ladder completes with every H reported with a CI. H0 (the report interaction
with the instruction in context) is clearly > 0 for the Qwen models, so the instruction was understood. The informative
outcomes are H1 on the primary, H1_scale across the ladder, and H5.

## Why it matters for our threads
- **Thread 2 and the workspace:** this tests the one workspace property the LW post says is not trivially satisfied,
  run on affect.
- **whitebear_pain:** that experiment measures "don't think about pain". This one measures the opposite, "think about
  pain", with the output pinned by teacher forcing, so the two together give both directions of voluntary control.
- **mask_pain / thread 14:** the template and base controls separate a persona effect from a pretraining one.

## Tested?
Only the code path was tested: 4-sentence smokes on gpt2 (raw format) and Qwen3-0.6B (chat format) on CPU in the
cloud sandbox. Their numbers were deleted, and no real result exists. With N=4, the smallest possible sign-flip p is
1/16, so a smoke cannot pass H1 by construction.
