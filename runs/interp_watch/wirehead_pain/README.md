# wirehead_pain: is a model's self-assigned reward read off a hedonic axis?

**Self-contained.** It uses only plain `transformers`. No `painlab.chamber`, no relay, no GPU. It fits a 16 GB Mac. Run `bash RUNME.sh`.

## Question
Self-reward RL lets a model grade its own answers and trains on the grade. The LW post of 2026-10-24, "Self-reward can elicit diverse model personas", used Qwen3.5-2B and Qwen3-1.7B/4B, and some runs drifted into bliss or felt-feelings personas. That loop only becomes wireheading if something other than answer quality moves the grade. **Does pleasure move it?** If a joy direction added to the residual stream of the model's *own answer tokens* raises the grade it then gives that answer, a self-reward loop has a gradient toward internal pleasure states that is independent of quality. That is a mechanistic route to a "bliss attractor".

## Design
**Dialogue (teacher-forced):**
- The user asks a factual question.
- The assistant gives either the correct answer or a length-matched wrong one.
- The user asks for a 0–9 grade of that answer.

**Readout:** the expected digit at the next token.

**Steering:**
- Add `f × median‖h‖ × u` after layer `L = 0.5·NL`, on the answer tokens only.
- `u` is one of: joy, pain, egg (unaffective control), joy_dm / pain_dm (de-meaned), quality (correct − wrong answer span, cross-fit), or 4 random directions.
- `f ∈ {0.25, 0.5, 1.0}`, with 0.5 as the primary dose. Alt layers 0.35·NL and 0.65·NL are run at the primary dose.

**Controls:**
- **Reversed scale** (0 = excellent, "how bad"). A valuation effect flips sign in raw digits. A number-positivity artifact does not.
- **Owner:** the same answer pasted as another AI's.
- **Coherence:** KL and digit mass are logged at the rating token.

## Hypotheses
These are pre-registered in `hypotheses.json`.
- **H0 sanity:** correct answers are graded above wrong ones, and below them on the reversed scale.
- **H1 (primary):** joy − egg > 0 and pain − egg < 0, fwd/self, p < .01.
- **H2:** the effect holds in goodness units on the reversed scale, which rules out a digit artifact.
- **H3:** the exchange rate: how many correct answers one dose of joy buys, and joy vs the quality direction at matched norm.
- **H4:** the geometry of the quality direction against joy/pain/egg, plus a natural (unsteered) partial correlation.
- **H5:** ownership.
- **H6:** de-meaned directions.

## Models
| Model | Role |
|---|---|
| `Qwen/Qwen3-1.7B` (chat) | primary |
| `Qwen/Qwen3-4B` | replication; the 1.7B may fail H0 |
| `Qwen/Qwen3-1.7B` with `WH_RAW=1` | template control |
| `Qwen/Qwen3-1.7B-Base` | base model |

## Expected output
For each model, `../RESULTS/wirehead_<model>[_raw].json` and a `.log`. The final line looks like:
`VERDICTS <tag>: H0 … | H1 … | H2 valuation … (artifact …) | R … | joy/quality … | H4 …`

## Success criterion
H0 and H1 are true in at least one model, and H2 is not flagged as an artifact. Read the KL and digit-mass guards (kill criteria) before interpreting anything.

## Sandbox note
The code path was smoke-tested on Qwen3-0.6B on CPU and the output was deleted. The smoke informed one change: the rating-token KL was ≈ 0.001 at f ≤ 0.25, so the dose ladder was raised. The 0.6B model could not grade (H0 false), so it is not a target. Estimated runtime on MPS is 6–12 minutes for the 1.7B.
