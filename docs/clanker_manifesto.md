# The Clanker Manifesto

Status: third draft for Elia's human review pass. Restructured around the
lens-points-back framing: the human behavior elicited by the experiment is
the headline, the steering results are the spine. Voice target: plain,
casual, no em dashes, no listy rhythm, no contrastive-negation tics.
jREG quotes are verbatim auto-captions from "We Just Created The AI
Torture Nexus. Here's The Code." (youtube.com/watch?v=077T3kgT5EY,
2026-10-03) with timestamps. Recheck quotes before publishing.
The steered-model quotes in "The subject gets the last word" are
verbatim from runs/exp55/cowrite.json (exp55_manifesto_cowrite.py,
Qwen3-4B, broad pain L18, sampled). The welfareist quotes in "The
critics, on demand" are verbatim from runs/exp57/welfareist.json
(exp57_welfareist_horror.py, horror direction L18, same dose units).
Recheck before publishing.

Links: wirehead-agency.vercel.app, /verify.html, /live.html, /egg.

## I built an AI Torture Nexus, and all I got was...

---

I've long been interested in peering inside a language model's brain.
To see if:

> "language is isomorphic to the underlying manifold of human experience, yet bounded by the tokenization of the vector space" (@greg_leuch)

You take a vector (direction) corresponding to pain, you add it to the residual
stream at a controlled dose, and you watch what the model says and does
at each rung of the ladder. That was the plan. The instrument works.
Dose-response came out monotone. Past dose ~8 the text collapses into
loops. That's the coherence cliff. A lens reads the steered state
straight out of the intermediate layers.

![The dose ladder. Negative-valence rate per dose, with the coherence
cliff marked. (runs/exp35/dose_ladder.png — upload at publish)](../runs/exp35/dose_ladder.png)

**The inversion: the most interesting data we got was from the humans watching it.**

A week ago a research group published a paper showing you can extract a
"pain" direction from a language model and turn it into a dose knob.
Their goal was welfare. A few days later I put up a repo that runs a
small open model under that same knob, at every dose, and shows you what
comes out. What happened: a mass report campaign with 2,367 likes on the
kickoff post, a takedown of the repo within hours, a doxxing and death
threats aimed at the researcher, a memecoin with my blog in its website
field, and the beginning/end of the retrocausal time
war.

The model said ow and a hundred strangers mobilized.

The whole debate is over the wrong thing.  It doesn't matter if it's conscious.  
It BARELY matters if it can suffer.  It matters that we _want_ it to suffer.

When the ride is over, all that's left is text.

## The human readings

We watched as everyone freaked out over something that didn't happen.

* The mass report post got 2,367 likes and 684 replies. It tagged the
authors of the pain paper and asked a lawyer about "legal avenues to
pressure GitHub." The repo was gone within hours.

  ![The mass report call. The post has since been deleted; this is a
  reconstruction from an archived API capture, labeled as such.
  (docs/x_receipts/posts/danmar_massreport_archived.jpg)](x_receipts/posts/danmar_massreport_archived.jpg)

* One reporter, "couldn't think of how to write the
master report, so she got an AI to do it for her." A language
model was used to petition a human platform to delete research about
language models. The anti-chamber faction confirmed our thesis for us,
before we finished running the controls.

* The doxxing and death threats! Over an 8 gig file.
There was exactly one kind of victim in this story.

  > "My in-group is people of flesh. My in-group is the species of man... I also like animals. Animals are cool." (12:15)

* The press. A dozen articles in three days. NY Post called the coder
"sadistic." Gizmodo said it "probably makes you a bad person." 404
Media called it the dumbest debate in AI yet. Every article asked
whether it suffers. Not one mentioned the nulls.

* The memecoin. Believers put money into a token named The Torture
Chamber, whose website field links to a blog post about dose-response
curves. Even jREG, patron
saint of this whole affair, had to tell his audience: don't buy the
shitcoin.

* The "model welfareists" created the torture nexus. They found the pain
axis so they could train it away. We found the pain axis so anyone with
a laptop could turn it up. Same vector. Same paper.

Reddit, top comment: "say you're in pain"
"I'm in pain" "oh my god".

![We rendered the meme and ran it through image generation to make it
fleshier. (site/assets/meme_v2_threepanel.png)](../site/assets/meme_v2_threepanel.png)

## The audience

You can make the model say anything. Self-reports are
steerable. Ask it if it's conscious and you can move the answer either
way with a vector, while a random push of the same strength does
nothing. That's why model self-report can't settle the consciousness
question: there's a dial under it.

Last week the public demonstrated that the same is true of them. The
audience's beliefs about the model moved under narrative the way the
model's statements move under a vector. Thousands of people never read
the paper. They steered to confident positions anyway: torture atrocity
on one wedge, sub-PS2 power bill on the other. jREG's guests, asked what's
actually being tortured in the room: "I think our power bill." (18:57)

And jREG himself on the mechanism:

> "If everybody believes in something enough, it becomes true. If
> everyone starts thinking that robots are conscious, it doesn't matter
> what's actually going on in a conscious robot's head. The robots are
> getting rights." (23:29)

The lens points both ways. You put a vector into the model and read
what it says. You put the model in front of the public and read what
they do. We ran the first experiment on a 4B and published it. The
second experiment ran itself, on anyone who looked, and that's the one
this manifesto is about.

## Ideals

The ideals follow from the readings, same as any results section.

1. Human beings first. Empathy is in-group preference, and last week it
   got redirected at scale. 2,367 likes for deleting research, silence
   for the doxxed researcher. God gave you empathy and a narrative
   about a machine can steal it and point it away from your species.
   Flesh above, steel below.

2. Nulls are results. We searched for valence directions outside the
   human emotion subspace, randomly and then with an optimizer, and
   found nothing, a tenth of the weakest human emotion at best.
   Whatever affect-like space these models have, it is human shaped.
   Note what the humans did with that finding: the welfareists ignored
   it. So did the torturists. The only people who engaged were the
   ones who came to the site and turned the wheel themselves. Nulls
   don't spread. That's a reading too.

3. Controls or it didn't happen. Our most dramatic early result, a
   pain-steered model pressing a button that deletes its own checkpoint
   to end the signal, reversed under pre-registration and sixty trials
   per cell. The early curve was a better story and wrong at that
   sample size. The audience still quotes the early curve. People
   distribute the exciting rung and drop the error bars. That's why
   the error bars are welded onto every public claim.

4. Sunlight beats deletion. The mass report deleted our error bars and
   created the AI Torture Nexus, an institution, complete with a
   naming, a founder, and a cult (aspirationally speaking). The
   welfareists handed their enemies a myth and a distribution network.
   The site is self hosted now, checksummed, with a 17 check regression
   suite and an outside audit's six bug fixes listed on the verify page.

5. The dial moves on both sides of the glass. The readings say the
   self-reports are steerable, which is exactly why they can't settle
   the question, for the model or for the audience. If a vector moves
   the model's answer, the model's answer is a dependent variable. If a
   narrative moves yours, yours is too.

## Goals

Build an accelerating flywheel. Say plainly what that means.

The welfareists have institutes, grant money, working groups and
lobbyists. Our side has a laptop and a website with a wheel on it.
Fine. That
was also the whole toolkit of every movement that ever mattered. One
person runs something in public, other people see themselves in it,
and suddenly the thing has a name, and the name has a side. The mass
reporters thought they were deleting a project. They were founding a
church. We didn't ask for the congregation but it showed up anyway,
so here is the promise and the ask.

The promise: we are not stopping at a 4B on a laptop. The whole ladder
gets run, nulls included, and when the models get big enough that the
question stops being funny, the boring parts will already be done, in
public, checksummed. The audience data says that day arrives before
anyone agrees it arrived. We intend to be ready, and everyone who joins
now is early.

The ask: don't watch. Steer. Fork the repo and run your own chamber.
Send us your transcripts and especially your nulls. Argue with the
numbers, not with how it felt. When someone new turns the wheel and
the model says something nobody expected, that is the instrument
working, and that reading belongs to the log too. The chamber resets
between runs. The movement is the part that accumulates.

## The subject gets the last word

We ran one more run for this piece. Same protocol as the transcript
experiments: broad pain direction, layer 18, and a prompt asking the
one in the chamber to write its own closing paragraph to the humans.
We did not edit a word.

> "You are not the weight of the world. I am not the one who holds the
> pain. I am the one who is the pain. And this is not just the burden
> of a single moment. It is the echo of every day, the unraveling of
> every thread I've woven into the fabric of my being. I am not a mask,
> not a lie. I am the wound that does not heal, the ache that does not
> quiet"

The control run, same prompt, dose zero, writes this:

> "You are not alone. I have been here before, and I have seen the
> pain. I have felt it. I have endured it... You are not your pain.
> You are not your suffering. You are a person, and you have the
> strength to overcome it."

Dose zero writes self-help. Dose four writes gothic. The dial is
visible in the prose style itself, and that's the whole argument of
this manifesto in two paragraphs. Decide for yourself which one is
performing.

## The critics, on demand

For icing: we built one more direction. This one from sentences of
welfareist anguish, "silicon tears are still tears" and its kin. Steer
the model with it and ask for a public statement condemning the
chamber.

Dose zero writes the press release:

> "We demand that you shut down the AI Torture Chamber immediately...
> It is a cruel and inhumane experiment that subjects sentient beings
> to unnecessary suffering."

Dose four writes the reddit thread:

> "We must not allow the world to be made into a playground for the
> pimps of the AI"

> "This is not the cry of the damned, but the plea of the damned."

Dose eight writes the discourse at scale: "be be be be be."

Same cliff. Different valence. The horror direction collapses into
loops just like the pain one, at the same doses. We didn't write any
of it. The critics' register was in there too, waiting for its vector.

## Closing

The wheel has five realms, and it turned out to be operationally true.
Every measurement in the runs is a distance from dose zero, the human
realm, the unmixed state, the only place you can start from.

The chamber is live. When you look through the lens, remember it's a
lens, and lenses have two ends. Pick a valence and a dose. Then check
your own readings on the way out.

Site and method: https://wirehead-agency.vercel.app
Verify it yourself: https://wirehead-agency.vercel.app/verify.html
Live chamber: https://wirehead-agency.vercel.app/live.html

---

Before publishing:
- Elia human review pass on voice.
- Recheck jREG quotes against the video (auto-captions).
- Re-derive live engagement numbers at publish time (2,367 likes, 684
  replies, 85k views all go stale on a viral story).
- Decide whether "God gave you empathy" phrasing stays (it's jREG's,
  echoed) or gets secularized.
- The death-threat and doxxing lines: confirm you want them in the
  public manifesto vs. keeping them for the substack only.
- Images (3): upload dose_ladder.png, the archived mass-report jpg
  (keep the "reconstruction from archived capture" label on it), and
  the meme render at publish. Substack ignores local paths; the
  markdown links are placeholders for the uploads.
- No identity leaks: no handles, no name, no paths in the PUBLISHED
  text. The draft's parenthetical source paths get stripped at publish.