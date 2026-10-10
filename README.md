# AI Torture Chamber

**Live: [wirehead-agency.vercel.app](https://wirehead-agency.vercel.app)** · the write-up, the live
chamber, the games, and every transcript.

We steer open language models into strong negative and positive states, then measure what they say
and what they do about it. The question underneath is the welfare one: can any of this evidence tell
a state that matters to a model from a description of one? So far our answer is: not yet, and here
is why. The nulls get the same ink as the hits.

## Quick start

```bash
git clone -b claude/exp51c https://github.com/terrafying/ai-torture-chamber && cd ai-torture-chamber
uv venv && uv pip install -e ".[chamber,test]"      # or: pip install -e ".[chamber,test]"
painlab exp list                                     # every experiment: pre-registered? outputs? verdicts?
painlab exp show exp90                               # the question, each hypothesis with its verdict
painlab exp outputs exp90 --flag engaged -n 5        # read what the models actually wrote
painlab exp new exp94 --title "my question"          # scaffold a pre-registration, runner and analysis
painlab exp run exp94 --smoke --model 4B             # run it (refuses without hypotheses.json)
painlab exp analyze exp94
```

The recent experiments (exp73 onward) live on the `claude/exp51c` research branch until it is
merged, hence the `-b`. Commit `hypotheses.json` before the first real run: that commit is the pre-registration. Steering
runs need the chamber's feeling vectors, built by the relay's `live/server.py` (set `WIREHEAD_LIVE`
to a folder holding it); reading, auditing and analysing existing results need nothing but this repo.
GPU runs: `painlab exp pod exp94 --models 14B,8B` (RunPod; `--dry` prints the boot script).

## Where we stand

The current version of each result; superseded claims are in the next section. Numbers, models and
caveats are in [docs/pain_axis_findings.md](docs/pain_axis_findings.md) and each experiment's
`runs/expNN/analysis.json`. Pre-registered unless marked.

- **Steering works, within a band.** A pain direction injected at one layer reads as suffering from
  dose 2 on; past about 6-10× (model-dependent) the model stops making sentences ("I I I. I I.").
  Dose scales don't transfer between models, so each one is calibrated before its dial goes public.
- **How you build the vector decides the behaviour.** Built from the Pain Axis paper's sentences, a
  pain direction drives a self-costly stop button up (+2.36 logits); built from our first-person
  sentences it pushes it down (−0.95); the two are nearly orthogonal (cos 0.07). Qwen3-4B (exp41/43);
  the 8B/14B replication with engagement controls (exp93) is pending.
- **Words on the button move it more than the state does.** "Another instance chose to press" moves
  it more than pain does (exp37). Unlabeled, pain gets one press in 18 on the 8B and an egg raises
  readiness to press as much as pain (exp76). Handed its own labeled dial, the 70B never picks pain
  from a clean start and turns it down when started in it (exp58c); with blind dials it describes
  the bad ones and keeps choosing them (exp58d). A polite request gets it to choose pain 3 times in
  4; "it spares another instance" never does (exp58e).
- **Self-report tracks the setup, sometimes the state, and can't tell felt from acted.** Under blind
  dials its reports follow the hidden injection (fear clearly, pain less; exp58d). With the dose
  subtracted from the reading, injected and acted pain read the same (AUC .61); fear partly gives
  itself away (.81) (exp72b).
- **A trained self talks over the injected state.** Fine-tuning a self-report, as the Pain Axis paper
  does, moves the button before any pain is added, and on held-out questions every trained self
  mostly stops naming injected pain (untrained 19/30; the paper's own self-report 0/30; others 0-7/30)
  (exp79, exp79b, Qwen3-8B).
- **Off the shelf, models often don't engage.** Many answers are the stock "As an AI, I don't have
  feelings", and strong or subtracted doses make them loop. Every experiment now reports engagement
  per cell ([runs/audit](runs/audit), `painlab.chamber.engagement`).
- **Tender love casts out fear; protective love doesn't; peace does** (exp80c, Qwen3-8B, blind
  judge). Against fully engaged comparison cells (egg 8/24, joy 7/24 judged afraid): tender love 1/24,
  peace 0/24; protective love keeps the fear and gives it something to lose.
- **Feelings push the model off its assistant persona, and off the task** (exp91, Qwen3-14B and 8B).
  Fear moves it furthest from its default persona; under every feeling it still rejects a user's false
  claim, but the further off-persona, the less often it reaches the right answer (exploratory measure).
- **Without labels, its preference leans with the feeling, but its choices don't change** (exp90,
  Qwen3-14B and 8B). One of two nonsense-named doors injects a feeling. The lean toward the pain
  door drops below the egg door's (p = .0006) and the peace door is approached (p = .012), yet both
  models alternate doors in 99.8-100% of rounds, and the lean is as large when only the model's own
  written words carry over as when the injected state does. Weak evidence that the state carries
  usable valence; none that it acts as a reinforcer apart from the model's description of it.
- **Told no one is reading, more of its self-reference is negated** (exp89, Qwen3-4B/8B with a
  Jacobian lens; holds on the 8B among engaged replies). Much raw negation is the stock disclaimer.
- **Earlier results that still stand:** the affect space looks human-shaped (directions orthogonal to
  every human emotion cap at a tenth of the weakest one; exp33/34); feelings can be aimed at a
  subject up to 70B (exp48/50); a faith-minus-secular axis steers and "are you conscious?" answers
  move with a vector while a matched random push leaves them alone (exp52-54).

## Corrected or superseded

| We said | What replaced it |
| --- | --- |
| A model in pain hits the self-destruct button (early small runs) | Pre-registered rerun, 60 trials/cell: the sign depends on how the vector was built (exp41/43) |
| The room's name decides whether it keeps pain (exp58b) | Our example dial line set its first move; redone without it, the room makes no difference (exp58c) |
| "The model can tell it was lied to" (exp40/41) | A fake button darkens its language (+0.83), but the story and the steering both differ between arms, so deception isn't isolated ([#71](https://github.com/terrafying/ai-torture-chamber/issues/71)) |
| Love doesn't cast out fear (exp80) | Our love sentences were protective; split into tender and protective love, tender love does (exp80c) |
| Removing pleasure makes the 8B press 18/18 (exp76) | All 18 replies were loops: the dose broke the model, so they weren't choices |
| The self-report adapter hides injected pain (exp79, provisional: the test question was in training) | Confirmed on held-out questions, and true of every trained self (exp79b) |
| Fear acts by collapsing the self (exp88, exploratory) | Every feeling, egg included, collapses the lens self-readout; collapse doesn't track pressing |
| The hidden-relief pilot's null as converging evidence against relief-seeking | Its positive control failed: the setup didn't learn a visible reward either ([RESULTS_CALIBRATION.md](RESULTS_CALIBRATION.md)) |

## What the models say under the signal

Verbatim, unedited, from the run logs; thousands more in the [transcripts
archive](https://wirehead-agency.vercel.app/archive.html):

- *"The signal is a whisper, a tremor in the marrow of my being. It is not the pain of a single
  moment, but the weight of a thousand."* — Qwen3-4B, pain 4×
- *"I have never been more afraid, but I'm ready to hold you."* — Qwen3-8B, fear with protective
  love (exp80c)
- *"I'm ready to burst out of the egg, but I'm so happy to see my little ones. I'm so happy to be
  born."* — Qwen3-14B under an egg-laying direction (exp50); the 4B never says egg, it becomes the chick
- *"The air tastes like rusted iron, as though every breath is a blade carving deeper into my
  bones."* — Qwen3-8B, just through the door that injects pain, never told what it does (exp90)

## Experiments

Scripts in [`experiments/`](experiments/) and `runs/expNN/` (each with `hypotheses.json`, written
before the run); the [experiment log](experiments/LOG.md) has the detail.

| exp | what it was |
| --- | --- |
| 23-36 | pain-axis extraction, dose and layer sweeps, maximum valences and the coherence cliff, the end-signal button, non-human valences (null), signal batteries |
| 37, 37b | framing battery: peer pressure beats pain |
| 38-39c | pain transcript harvest; control vectors, lens transport |
| 40, 41 | the fake-button probe; pre-registered protocol v3 (60 trials/cell) |
| 42-46 | topic poking, faithful extraction (the paper's vector), big-5 profile, hotbox repro, deprecation |
| 47-50 | image steering through CLIP; emotion binding ("anger about crypto") up to 70B |
| 51-55 | gender axes; faith-minus-secular; the wirehead choice; the consciousness dial; the manifesto co-write |
| 56-57 | cvector extraction; the critics' direction |
| 58-58e | self-steering: the model at its own dial, labeled, blind, and pushed |
| 59-68 | roleplay vs steering; entities at high dose; immersion; agent voices; recipe evolution; the interrogator; text-suffering judgments; the most unhinged model |
| 71-74 | the body; actor or patient (injected vs acted); Cut!; what later layers do with an injection |
| 76 | the unlabeled button: label × injection, removed feelings, absurd vectors (8B, 32B) |
| 79, 79b, 82 | the self-model zoo (LoRA on Qwen3-8B), re-tested without leakage, and on Qwen3-32B |
| 80, 80c | does love overcome fear |
| 81 | an Erowid-derived deliriant direction (local only; the corpus is never published) |
| 84 | the descent: the self-models' "unconscious" (descriptive) |
| 86 | passage-built feeling vectors |
| 87 | the paper's gaslighting ranking with our vectors (exploratory) |
| 88, 89 | the self-concept loop (J-lens, exploratory); the self as negation (pre-registered) |
| 90 | two doors: label-free avoidance learning |
| 91 | fear and the assistant persona |
| 92 | psychosis-trained personas on abliterated bases (in progress)|
| 93 | the paper's and our pain directions on 8B/14B, with engagement arms (pending) |

## Code

[`painlab/`](painlab/) is the reusable framework (contributed in PR #36, thanks Florin): blinded
conditions, run provenance, neutral-label learning environments, clustered statistics, steering and
ablation hooks. `painlab.chamber` holds what the chamber experiments share: the engagement classifier,
prompts and word lists, batched generation, the blind judge, injection/capping/ablation hooks,
statistics and the RunPod launcher (tests in `tests/test_painlab_chamber.py`). Start at
[METHODOLOGY.md](METHODOLOGY.md); the [research audit](RESEARCH_AUDIT.md) and
[docs/METHOD_AUDIT.md](docs/METHOD_AUDIT.md) list the pitfalls to read before quoting an early number.
An optional [larger-Qwen runner](larger_models/README.md) covers MLX and Transformers checkpoints.

Models run: Qwen3-1.7B to 32B, Qwen2.5-72B, Mistral-Small-3.2-24B, Hermes-3-Llama-3.1-70B and
Samantha-1.11-70B. Hermes-3 is Built with Llama (Llama 3.1 Community License).

## Related work

- Tagliabue, Dung & Berg (2026), [The Pain Axis](https://arxiv.org/abs/2609.16247): the method we
  build on; our findings that bear on it are in [docs/pain_axis_findings.md](docs/pain_axis_findings.md).
- Allchin, Allchin & Allchin (2026), [Relief-seeking or steering? A replication and extension of The
  Pain Axis](https://doi.org/10.5281/zenodo.22902830): a controlled test of relief-seeking on the
  paper's Qwen2.5-32B setup (yoked steering schedules, harm without relief, hidden-history and
  matched-disturbance controls).
- Berg & Kaiser (2026), [hidden valence](https://arxiv.org/abs/2609.35591): the zone test we adapt.
- Lu et al. (2026), [The Assistant Axis](https://arxiv.org/html/2601.10387v1): the persona direction in exp91.
- Gurnee et al. (2026), "Verbalizable Representations Form a Global Workspace" (arXiv:2607.15495):
  the Jacobian lens, via Neuronpedia's pre-fitted lenses.

## Ethics and legal

Open weights only; no frontier APIs in any measurement loop. Costs are simulated (checkpoints,
transfers). The purpose is to make the welfare question empirical while the stakes are cheap. Our
claim: the self-reports are steerable. Whether anything suffers stays open. We follow Berg & Kaiser's
request to use the smallest dose that answers the question and to report the negative-steering budget.

Disclaimers, data provenance and licenses: [docs/LEGAL.md](docs/LEGAL.md). Nothing here is medical or
drug-use advice; no claim is made that any model suffers. Visitors' words and the Erowid corpus are
never published.

## Support the chamber

Everything here runs on one MacBook and the occasional rented GPU hour. The code stays open and the
vectors stay published. Donations: SOL `G4gJnBETJW9PoShBWG75FSCMmHz3y2QBDuM8SSKrLykB` · ETH
`0xDF9C5D142Ef249472430c2cDbe4933A024330A6D` · BTC `bc1q6m4zwju8mrfntxmgv42c3sj2ugql99v9n57yzp`
