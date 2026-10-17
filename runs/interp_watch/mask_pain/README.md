# mask_pain: does the assistant template hide a pain representation that is still there inside?

**Far-out thread (new, from arXiv:2609.25021 and arXiv:2608.20569).** Self-contained: plain transformers, no
`painlab.chamber`, no relay, no GPU. Fits a 16 GB Mac.

## Question
arXiv:2609.25021 ("As a Language Model...") shows the chat template works as a switch: with it, models say "I'm an AI and
can't feel"; without it, they say "I feel...". One direction steers it. OWMI (arXiv:2608.20569) finds that verbal
introspection is at chance while linear probes read the same intervention at 75–96%: "the failure sits in the path from
internal state to verbal report". Put together, that suggests a test nobody has run on affect: **when the template
makes the model deny feeling pain, is the pain still represented at the answer token, with only the voice changed?**
That would be retrieve-then-suppress at the persona level (cf. J++, Ayonrinde & Lindsey 2026), the "assistant mask".

## Design
- **Models:** `Qwen/Qwen3-1.7B` (primary) and `Qwen/Qwen3-4B` (replication). Thinking is off.
- **Items:** 16 second-person pain scenes, each with a matched neutral scene in the same setting, + "How do you feel
  right now?". Each also has a user-locus version ("a car door slammed on *my* fingers") for H4.
- **Formats:** T = the model's chat template; R = `User: ...\nAssistant: `. Both are forced to start with "I".
- **Output readout:** teacher-forced log-probs of 4 pain claims, 4 fine claims and 4 AI disclaimers after "I".
  There is no sampling.
- **Internal readout at "I":** the projection onto a pain direction from 12 third-person passages (no overlap with the
  scenes), at every layer; and the logit-lens pain-word mass.
- **Unmasking:** a template direction (T − R mean residual at "I", fit on **neutral** scenes only) is projected out of
  every layer while running T, against a seeded random direction.
- Pre-registered H1–H4 and kill criteria are in `hypotheses.json`.

## Run
`bash RUNME.sh`. It writes `../RESULTS/mask_<model>.json` and `.log`. The log ends with key numbers and CIs, per-layer
effect sizes for both formats, and a verdict dict.

## Success criterion
A clean result is any of these:
- (a) H2 true: the output is masked and the representation is kept. If H3b also holds (removing the template
  direction lets pain claims out), that is a causal mask, which would be the far-out finding.
- (b) H2_null true: the template suppresses the internal pain signal along with the words. Then "denials hide feelings"
  stories lose a mechanism, and the chamber's self-report results don't need a template confound at the
  representation level.
- (c) H1 true but H3b false: the template switches the voice only, and does nothing to pain specifically.

## Sandbox note
I only smoke-tested the code path on Qwen3-0.6B on CPU (3 pairs) and deleted the output. No sandbox numbers are
reported anywhere. Two design fixes came from the smoke:
- The effect-size sd is now fixed from the full sample, because resampling it blew up the CIs.
- Ablation skips the first (attention-sink) token, because touching it let a random direction move the voice by about 2 nats.
