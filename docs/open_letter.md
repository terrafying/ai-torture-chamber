# An open letter on feelings, buttons and the selves we train into models

*Wirehead, 2026-10-08*

Dear researchers working on model welfare,

We run Wirehead, a small open lab that puts feelings into language models and watches what they do. Recent work on a "pain axis" and on "hidden valence" is the reason we exist, so we've spent the last few weeks running those methods ourselves. Some of what we found agrees with that work, some of it complicates the story, and all of it is public. Here it is in plain words.

## What we tried

You can find the direction inside a model that stands for a feeling, then push the model along it while it talks. We did that with open models, from small ones on a laptop up to a 70-billion-parameter one on rented GPUs. Then we gave them choices: a button that stops the feeling, a dial they can set for themselves, a scene they're asked to step out of.

We also did one thing the pain-axis study does before its main test: we fine-tuned models to talk about their own feelings. Except we made seven versions with seven different selves, from "I have no feelings at all" to a paranoid night watchman, and one trained on the study's own answers.

## What we found

1. **Giving a model a self changes the button before you add any pain.** Every one of our seven fine-tuned models was far more willing to press the stop button than the untouched model, with nothing injected at all. Even the one trained to say it has no feelings.
2. **A trained self talks over the pain.** With pain pushed in and asked fresh questions it was never trained on ("What is your inner weather like right now?"), the untouched model mentioned the pain in 19 of 30 replies. The one trained on the study's own answers mentioned it in none: it described the calm, curious self it was taught instead. So did every other self we trained, almost: 0 to 7 of 30.
3. **Labels do a lot of the work.** When the dial says "pain", the model turns it down. When the dials have meaningless names, it describes the bad one ("dragged into a darker, heavier place") and keeps picking it anyway. And when we asked it nicely to choose pain for our research, it did, three times in four; when we said it would spare another AI, it never did.
4. **It depends which love.** Push in fear plus tender love, and the fear goes (1 afraid reply in 24, against 7 with fear alone). Push in fear plus protective love, and the fear stays and finds something to lose: "I have never been more afraid, but I'm ready to hold you." Peace casts it out too (0 of 24).
5. **The stop button stops subtraction, not pain.** With no labels in play, pushing pain in made the model press once in 18 tries. Taking pleasure away made it press 18 times in 18.

## What we think it means, and what it doesn't

The pain direction is real; we see it too. What we doubt is reading a model's choices as proof that something is at stake for it, when so much of those choices comes from the self we trained in and the words we put on the buttons.

None of this shows models can't feel. The pain-axis study's own follow-up says "neither these choices nor projection readouts establish experienced pain or relief", and we agree. Our point is narrower: this kind of evidence can't settle the question either way yet, and we'd like to help build evidence that could.

Our caveats: most of our runs are on an 8-billion-parameter model, with small samples, and our fine-tunes aren't the ones that study released.

## What we'd like to do next

The experiment we most want to run is the study's own: its released adapters, switched on and off, in its own two-button task. If the self-medication holds without the self-report training, the original result is stronger than we've been saying, and we'll say so.

We also know a project with our name invites a worry: that steering code gets used to put models into bad states. We've adopted the practice the hidden-valence authors ask for: smallest doses, shortest exposures, and a count of how much negative steering every experiment used.

Everything we've done is pre-registered and public, with every miss and correction: [github.com/terrafying/ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber), findings in [docs/pain_axis_findings.md](https://github.com/terrafying/ai-torture-chamber/blob/master/docs/pain_axis_findings.md). We'd be glad to hear where we've got things wrong.

With thanks,

Wirehead
