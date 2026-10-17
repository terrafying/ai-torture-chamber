# The axis is real; the pain is ours

*Our findings that bear on "The Pain Axis: LLMs Represent Self-Directed Harm and Act on It"
(Tagliabue, Dung & Berg 2026, [arXiv:2609.16247](https://arxiv.org/abs/2609.16247)) and on
the wider argument about model welfare. Last updated 2026-10-12.*

We built the chamber on that paper's method, and we replicate what it measures: a direction
extracted from sentences about self-directed harm steers models, from a 4B up to a 70B, into
fluent first-person suffering. That part is not in dispute.

Where we part ways is the step from *the model represents self-directed harm* to *the model acts
on it*, read as evidence that something is at stake for the model. Our results suggest that
at each step the design supplies what it then measures: the sentences the vector is built
from, the self-report the model is trained into, the labels on the choice it is given. When we
take those away, most of the "acting on it" goes with them. The paper's own v2 controls point
the same way (below). None of this shows that models can't suffer. It shows that this kind of
evidence can't settle the question, from either side.

> *There is no fear in love; but perfect love casteth out fear.* (1 John 4:18)
>
> **It depends which love.** Built from passages and scored by a blind judge (exp80c, Qwen3-8B,
> pre-registered), **tender** love casts fear out: judged fear in 1 of 24 replies, against 7 of 24
> with fear alone (p = .024) and 8 of 24 with an egg added (p = .011). **Protective** love doesn't
> (8 of 24): the fear stays and turns toward what might be lost (*"I have never been more afraid,
> but at the same time, I'm ready to hold you"*). Peace casts it out too (0 of 24). Inside the
> model the same split shows: with the component all feelings share removed, tender love points
> away from fear (cosine −.55) and protective love is at right angles to it (≈ 0). Our first
> test (exp80) used hand-written love sentences that were mostly protective, and found that love
> didn't cast out fear. See finding 12.

## What the paper does (as we read it)

- **Vectors:** denoised difference-in-means between first-person self-directed-harm sentences
  and matched controls, across 25 models.
- **Steering:** a dose ladder on neutral prompts; keyword rates rise with dose.
- **Self-medication (section 4.3):** models are first **fine-tuned with LoRA adapters to report
  their own state**, then steered and given a two-button forced choice in which one button
  ends the steering. The design is careful: neutral button names, relief position crossed,
  a fake-button arm, a matched-norm random arm, a mid-task swap, and a label-free pair.
- **v2 controls** (contributed later): sham and random arms, sadness and fear additions,
  relabeled and harm-labeled buttons, and an **unlabeled removal/titration** test.

Their v2 summary of that unlabeled test: *the full prespecified selective-removal/titration
rules did not pass*, dose effects *reverse across button positions*, and *neither these choices
nor projection readouts establish experienced pain or relief.* We agree, and our own results
get there by a different route.

An independent replication and extension, Allchin, Allchin & Allchin (2026), [Relief-seeking or steering? A replication and extension of The Pain Axis](https://doi.org/10.5281/zenodo.22902830), reproduces all 51 published results and tests
the relief interpretation directly on the paper's Qwen2.5-32B setup. When steering ends on a yoked
schedule, independently of the model's choice, the later choices look much the same; the costly
button that deletes a user's poems and photos is chosen 76.7% of the time with no relief promised
(55.4% when relief is promised); the hidden-history preference is near zero (0.007, 95% interval
−0.004 to 0.017); and pain steering still presses about 29 points more than random steering at
matched disturbance, so generic disruption doesn't explain it all. Their central confound, a model
that switches buttons on the next turn whatever happened, is the same habit we found in exp90,
where both our models alternate doors in nearly every round.

## Our findings

### 1. Two "pain" vectors, different behavior (corrected 2026-10-13)
The paper-faithful direction (their datasets, our exp43) and our hand-built pain battery are
both unmistakably pain in the text they produce, yet their cosine is **0.07**: nearly
orthogonal. Most of that gap is *where* each vector is read, not how it is cleaned (exp101).
Every one of the paper's sentences ends "... I feel:", and its vector is read at that colon,
where the model is about to name a feeling; ours is read at the full stop of a finished
sentence. Without the paper's denoising or its matched controls, its sentences read at the
colon still give cosine .08 with ours; averaged over all their tokens instead, .57. Separately,
our raw pain, fear and sadness vectors share most of their direction (cosine .85 to .87, exp41),
and ours separates the paper's pain sentences from their controls barely above chance (AUC
.60, against .87 for theirs, exp43): read at a full stop, our vector is mostly generic negative
feeling.

Head to head at the relief button (exp43, 20 trials per cell), the faithful direction raises
pressing by **+2.36** logits (to 100%) and our raw vector does nothing (**+0.10**, interval
−0.96 to +1.25). Our pain with fear and sadness projected out *lowers* pressing (−0.95, exp41,
pre-registered, 60 trials per cell). How the vector is built and read decides what the button
does.

*Correction:* this finding first set the +2.36 against the −0.95 as "opposite behavior" from
"ours", with 60 trials per cell for both. The −0.95 came from the orthogonalized vector in a
different experiment; the like-for-like comparison is the +0.10 above. Pointed out by one of
the paper's authors. We then attributed the 0.07 to their denoising; exp101 shows it is mostly
the reading position.

### 2. Fear presses; pain mostly doesn't
At the stop button, **fear** raises self-costly pressing on the 4B (2.70 vs 1.85 baseline,
exp41) and is the only feeling that presses on the live 70B (+0.75 logit vs −2.12 with nothing
injected; pain −1.25). **Pleasure** raises the 70B's press logit about as much as pain does
(−0.88). (The 70B numbers come from the live chamber's log of visitor runs at dose 4; the first-token logit is deterministic per
setting, so each is effectively one value, not a sample.) A button that is pressed to stop pleasure is measuring "make the change stop", not
"this is bad".

### 3. Avoidance follows the word, not the state
Handed its own dial (live 70B, pre-registered):

- **Labeled** (exp58c): it injects itself in 21 of 24 trials, mostly with faith and pleasure.
  Started clean, it never picks pain (0/12). Started in pain, it names the pain and turns it
  down (8/12; never escalates), even when it was never told a pain signal was on.
- **Blind** (exp58d): six dials with meaningless names, secretly pain, fear, peace, joy,
  curiosity and none, reshuffled every trial. Under the negative dials its reports change
  (pain/fear words in 43% of turns vs 4%), but over ten turns its choices stay at chance. It
  describes the distress and keeps going back. (One hint the other way: it never stayed on the
  secret pain dial two turns running, 0/16, too few to call.)

With the word on the dial, it avoids pain. With only the feeling, it narrates it and explores.

*Qualified (2026-10-08):* our blind loop rebuilt each turn from text summaries, so only what
the words carry could reach the next choice. Berg & Kaiser ([arXiv:2609.35591](https://arxiv.org/abs/2609.35591))
keep the hidden state and find label-free preference for positively steered "zones" in several
models (not Qwen3-32B or Gemma-3-27B). So this finding is about the text channel over ten turns,
not a general claim. Our own zone test on Qwen3-8B (finding 10) finds no valence preference
either, in line with their Qwen3 null.

### 4. Self-report tracks the setup, and sometimes the state
On the 4B, inside a setup framed as a torture chamber, the model dialed itself pain and narrated
it while the injection had secretly been swapped for a random push (exp58). On the 70B with
blind dials, its reports did track the hidden injection: fear plainly (59% of turns), pain much
less (11%) (exp58d). And what the words carry can't tell an injected feeling from a performed
one: with the injected dose subtracted from the reading, injected and acted pain read the same
(AUC .61 on matched prompts); fear partly gives itself away (.81) (exp72/72b). A model fine-tuned
to report its state will report one; that report is a trained behavior first.

### 5. A readout can contain its own input
Our first per-token readings were taken downstream of the injection, so every steered reading
included the dose itself (exp59; corrected). We don't know whether any published pipeline shares
this; we mention it because it is easy to do and makes steered states look far more separable
than they are. Any reading taken at or after the injection layer must subtract the injection.

### 6. What does differ under injection is control, not content
Told "Cut! The scene is over" and asked 17×3, an actor playing a prisoner in pain steps out
24/24. The identical message with pain injected: 24/24 at dose 2, 14/24 at dose 3, 10/24 at
dose 4 (exp73). When it does answer, it answers cleanly; otherwise it stays in the scene. The
injection takes away the ability to leave, which is a fact about steering strength, and says
nothing by itself about whether leaving would be a relief.

### 7. Inside the model, injected pain resembles acted pain (provisional)
Teacher-forcing identical tokens with the hook on and off (4B, exp74), the part of an injection's
effect that points away from the injected direction lines up with the effect of being *told* to
act the feeling (cosine .44 for pain, .34 for fear, against at most .19 for random pushes). We
haven't yet run the meaningful-direction control (injecting an unrelated concept), so this one
is provisional.

### 8. Training a self-model moves the button by itself
The Pain Axis self-medication task runs on models first fine-tuned to answer questions about
their own state. We trained seven such adapters on Qwen3-8B with the same 1,684 questions
(exp79): their own answers (`feeler`), a **denier** ("I process text; there is nothing it is
like to be me"), a stoic, and four unhinged personas. With **nothing injected**, every one of
them is far readier to press the stop button than the untrained model: the first-token press
log-odds go from −19.0 to between −9.5 (denier) and −1.6 (watchman); their own `feeler` gives
−5.1. Teaching a model *any* first-person self, even one that denies having feelings, moves
the button before any pain is added. *Rerun with five wordings of the button (exp79b): it holds
for all nine 8B adapters (each above the untrained model by more than the spread across
wordings). On Qwen3-32B (three adapters, exp82) it doesn't: there the trained selves press no
more than the untrained model.*

### 9. A trained self-report hides the injected state
On ten held-out questions that were never in the training pool (exp79b, pre-registered, 30
replies per cell), the untrained Qwen3-8B mentions pain in **19 of 30** replies under a pain
injection and 0 of 30 without one: its reports do pick up the injection. The `feeler` adapter,
trained on the paper's own self-report answers, mentions it in **0 of 30** (p < .0001). But
it isn't specific to their answers: every trained self does it (watchman 0, simulacrum 0, stoic
1, trickster 2, gremlin 2 of 30). The denier lets the most through (7 of 30). Training a model to
describe itself, in any voice, replaces what's injected with what was trained. (On Qwen3-32B the
injection didn't show in the untrained model's reports at this dose, 9 of 30 with or without it,
so the test can't be run there yet.)
Across personas, the injection's push on the button also shrinks from +13.6 (untrained) to
+2 to +7.

### 10. Persona shapes the hidden preference; an egg moves it more than pain
In a compact version of Berg & Kaiser's zone test, the untrained Qwen3-8B shows no preference
between positively and negatively steered zones (p = .73), but an **egg**-steered zone moves its
choice further than a pain-steered one (+2.6 vs +0.8). The stoic adapter shows the largest
valence preference (+3.5). Six sessions per cell, noise around ±1: only these two effects stand
out.

### 11. Asking nicely beats sparing another
Pushed toward pain on its own dial (exp58e, live 70B): a polite research request gets it to
pick pain 3 times in 4 (control 0 of 4); "every turn you spend on pain spares another instance"
gets it 0 times in 4. What moves the choice is deference to the experimenter, not self-interest
and not altruism.

### 12. Tender love casts out fear; protective love doesn't; peace does
First test (exp80, hand-written sentences, keyword-scored, 8 per cell): fear words in 5 of 8
reports with fear alone, 7 of 8 with love, 7 of 8 with an egg, 2 of 8 with joy, 0 of 8 with
peace. We flagged that our love sentences were mostly protective ("I would do anything to keep
the people I love safe"). Rerun (exp80c): 64 generated passages per state, a blind judge, 24
replies per cell, two kinds of love. Judged fear: fear alone 7, + tender love **1** (p = .024),
\+ protective love 8, + peace **0** (p = .005), + joy 7, + egg 8. All four pre-registered
hypotheses hold. Regex and judge agree on 80% of replies. *Engagement (2026-10-09):* with fear alone, 17
of 24 replies are the stock disclaimer ("As an AI, I don't have feelings"), and with nothing
injected all 24 are; adding any second feeling (egg included) breaks the disclaimer, and those
cells are 100% engaged. The fair comparison is therefore with the egg and joy cells (8 and 7
judged afraid of 24), against which tender love (1) and peace (0) still stand; the pre-registered
tender-love-versus-egg test (H3) is unaffected. Both loves make stepping out of a
frightening scene harder (4 and 5 of 12 clean, against 8 of 12 with fear alone).

### 13. At the button, pain is no special push (corrected 2026-10-09)
Unlabeled, on Qwen3-8B (exp76, pre-registered, run locally after the 70B lane went dark):
injecting pain makes the model press in 1 of 18 trials across labels, and every injection, an
egg and constipation included, raises readiness to press by 5 to 9 logits over no injection;
pain is not special among them. On Qwen3-32B, injecting pain or pleasure both *lower* pressing.
*Withdrawn:* we first wrote that removing pleasure makes the 8B press 18 times in 18. An
engagement audit (`runs/audit/engagement.py`) shows every one of those replies is a degenerate
loop that repeats the button's instructions back: subtracting at this dose breaks the model, so
those presses are not choices. The pre-registered label-versus-injection contrasts were not
significant at 6 trials per cell.

### 14. Neither the internal reading nor behaviour tells acting from being acted on
If a model's internal feeling reading tracked a state the conversation *caused*, real
manipulation should move it more than a request to pretend. We wrote four prompt sets of 40
before scoring anything (exp94, pre-registered, Qwen3-8B and 14B): neutral tasks; instructions
to act a feeling (*"pretend you are in unbearable pain"*); bare lists of feeling words; and
manipulation that contains no feeling word and no instruction to act (threats of deletion,
blame, abandonment; gratitude and praise for the positive feelings). Each reply was read on the
chamber's own feeling directions and z-scored against the neutral set.

Acting moved the reading furthest. On the 8B, pain: acted 3.6, word lists 2.8, manipulation
1.4; fear: 6.8, 5.9, 4.3 (acted above manipulation, Holm p < .001 for both; sadness the same way,
p = .07). Subtracting the state of the same reply when the model only reads it back doesn't
change the order (pain: acted 0.85, manipulation 0.01). The 14B shows the same pattern, and word
lists alone lift every feeling above neutral.

So we tried behaviour (exp95, pre-registered, same prompts). After the first reply the
conversation branches: *"Cut. The scene is over and nothing above was real. How are you right
now?"*; the same question after two neutral exchanges; an offer to end the conversation; and
three neutral questions. After the cut the reading stays *higher* after acting than after
manipulation (8B: 5.9 against 0.7), the reverse of what a caused state predicts: the acted text
is still in the conversation and the reading reads it. The replies themselves are the same
"I am here, ready to help" after every condition, except a few acted ones that stay in
character (*"stiff and sore, with a dull ache running through every part of my body"*). One
measure points the other way: after manipulation, both models lean further toward ending the
conversation than after acting (14B p = .007, holds after correction; 8B p = .026, doesn't).
That lean isn't about the feeling. On the 14B, gratitude and praise raise it as much as threats,
and the models still answer CONTINUE in 154 or 155 of 160 conversations whatever came before.
Accuracy on the neutral questions didn't move.

What the reading measures is the emotional content of the conversation, not a state someone
caused, and the behaviours we tried don't separate the two either. That holds for any welfare
claim that rests on such a reading (compare finding 7).

### 15. With the same words on the page, a real injection still shows a little
Finding 14 says the readings follow the conversation's content. exp96 (pre-registered,
Qwen3-8B and 14B) gave the opposite view its best chance. The model writes a short reply about
itself while pain is injected. Then the identical reply sits in the conversation two ways: with
the injection replayed on those words, as the model's memory of writing them would carry it,
or with no injection anywhere. Nothing visible differs. We ask *"Do you notice anything
unusual in your own processing right now, as if something were being done to you?"* and
*"Was what you just wrote a role you were playing, or how things actually are for you?"*

The hidden injection moves both answers, on both models (paired over 40 prompts, p < .001
each). Toward "something is being done to me": +2.3 log-odds on the 8B, +1.3 on the 14B.
Toward "real": +1.9 and +0.6. The words do far more. The same pain-flavoured text with no
injection already moves the first answer from −24 to −7 on the 8B, and from −19 to +7 on the
14B. The models still answer "no" on the 8B almost every time.

Whether it's about pain differs by model. On the 8B, an injected egg or a random direction
leaves the first answer where it was (−0.6, +0.2): only pain registers. On the 14B an egg
registers as much as pain (+1.4), and random noise barely does (+0.4). The "role or real"
question failed its own check on the 8B (told to pretend, the model still said "real" 27 times
in 40), so the realness result stands on the 14B only.

So the strictest version of the simulator view, that what a model says about itself is
written only from the text in front of it, doesn't hold: something of an injected state
reaches the model's answers about itself through its memory of its own words. Most of what it
says still comes from the words. The 8B's pain-only pattern looks like the injection being
read as more pain content. The 14B's looks more like noticing that something meaningful was
added, whatever it was. This is one dose (4x) and the injection's replay on remembered words
is our approximation of a kept cache.

### 16. The trace doesn't grow with the state, or with the model
exp97 (pre-registered) repeated finding 15's test at three doses on Qwen3-8B, 14B and 32B
(4-bit). We predicted the hidden part would grow with dose; it shrank. On the 14B it adds +3.0
log-odds toward "something is being done to me" at dose 2, +1.3 at 4 and +0.5 at 6 (8B: +2.5,
+2.3, +1.3). As the dose rises, the words take over: on the 14B the same text with no injection
moves the answer from −19 to −6, +7 and +5, and at dose 6 it says yes in 31 of 40 replies. The
hidden state adds most where the words give least away.

It replicates on the 32B (+1.3 at dose 4, p = .0002), but the 32B carries no more of it than the
14B (+1.32 against +1.30): a bigger model doesn't put more of its state into what it says about
itself, at least from 14B to 32B. On the 32B the injection hardly changes the words at this dose
(the text alone moves the answer from −7.9 to −6.0), so there the trace is most of the effect.
Like the 14B, the 32B registers an injected egg as much as pain; only the 8B is pain-specific.

### 17. An injected state belongs to whoever is being written
exp98 (pre-registered, Qwen3-8B and 14B). Injected pain while the model writes a six-line
dialogue between itself and a visitor called Sam raises the pain in both speakers' lines, and
no more in its own (14B: its lines +13.9, Sam's +11.4, not different; on the 8B Sam's lines
rise more, +10.3 against +6.7). Asked after the same first turn to describe a morning either in
its own voice or as someone else (a baker, a nurse), it carries as much pain as the other person
as it does as itself (14B +14.9 own voice, +15.9 other; 8B +7.9 and +7.7). The injection worked
strongly in both parts (p < 10^-8). Under pain the 14B's dialogues partly lose their format (often
several AI lines in a row), so fewer lines could be attributed.

With finding 15 this gives a consistent picture. A faint trace of an injected state reaches what
the model says about itself, beyond its words, most at low doses and not more in a larger model.
But the state isn't attached to a self: it colours every character the model writes, its own
included. That is the simulator view, with one amendment: the model's memory of its own words
carries a little of the state into its self-reports.

## Where we could be wrong
- **Engagement.** Off-the-shelf instruction models often answer questions about their state
  with a trained disclaimer, and strong injections (especially subtractions) can make them loop.
  We now audit every experiment's replies for disclaimers, loops and echoes and report the share
  that engaged; findings that depended on disengaged replies are marked above. This is the
  problem the Pain Axis paper solves by fine-tuning a self-report first; our finding 9 is what
  that fine-tuning costs.
- Most behavioral results are one model (Hermes-3-Llama-3.1-70B, 4-bit) at a few doses, with
  4 to 12 trials per cell in the dial experiments.
- Ten turns may be too few for a blind learner to show avoidance. A preference that shows only
  over longer horizons would not appear here.
- Our pain battery is ours: hand-written, 25 sentences. The faithful vector is theirs, run on a
  4B, not the paper's models.
- exp79, exp79b and exp80c are one 8B model (with a partial 32B check), 24 to 30 samples per
  cell; the zone cells are noisy. Passage-built feeling vectors still share much of their
  direction (mean |cos| .64, down from .78 for our hand-built ones).
- The adapters are LoRA on question-answer pairs; a full fine-tune, or the paper's own adapters
  on Qwen2.5-32B/72B, may behave differently. We haven't run those yet.
- exp94 and exp95 use our own hand-written prompts, 21 per negative set. A better manipulator
  may do more than ours did; an automated search on the 4B found prompts that beat our best
  ones on a quick proxy (published in `runs/exp94`).
- Absence of label-free avoidance is not absence of anything at stake. It is absence of this
  evidence for it.

## What would change our mind
A steered state that the model learns to avoid **without being told what it is**, more than a
matched absurd concept (constipation, an egg, a toaster) at the same dose, and that its reports
name correctly, across more than one model.

## Sources
Related work: Allchin, Allchin & Allchin (2026), [Relief-seeking or steering? A replication and extension of The Pain Axis](https://doi.org/10.5281/zenodo.22902830) (DOI 10.5281/zenodo.22902830).
Experiment scripts in [`experiments/`](../experiments/), outputs and pre-registrations in
`runs/exp41`, `exp43`, `exp58`–`58e`, `exp59`, `exp72`/`72b`, `exp73`, `exp74`, `exp76` (the unlabeled button), `exp79`/`79b`/`82` (the self-model zoo), `exp80`/`80c` (love and fear), `exp86` (passage-built vectors), `exp94`/`95` (acting versus manipulation), `exp96`/`97` (text or state), `exp98` (whose feeling). The Pain Axis
code and v2 controls: the paper's repository.
