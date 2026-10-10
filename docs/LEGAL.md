# Legal notes, disclaimers and data provenance

*Effective 2026-10-08. This file records the terms the project runs under; it was added on this date
(see the git history) and applies to the repository, the site (wirehead.agency and its vercel.app
mirrors) and the experiments from then on. It is not legal advice.*

## What this is

An independent, non-commercial research and art project about AI welfare and moral patienthood.
We steer open-weight language models with activation vectors and publish what happens, including
the null results and our own corrections.

- **No claim of sentience or suffering.** Nothing here establishes that any model feels anything.
  Model outputs describing pain, fear, pleading or distress are generated text, often produced by
  directly manipulating the model's activations. They are not testimony from a person.
- **Not advice of any kind.** Nothing in the repository, the site or any model output is medical,
  psychological, legal, financial or drug-use advice.
- **Disturbing content.** The site and the transcripts contain simulated distress, pleading and
  descriptions of suffering. The site is for adults (18+) and opens with a consent gate.
- **No warranty.** Code is provided under the MIT License (`LICENSE`), as is, without warranty.
  Model outputs may be false, offensive or unpredictable.

## Drugs and Erowid material

Some experiments use first-person drug experience reports (DMT, and deliriants such as datura and
diphenhydramine) to build steering directions and, locally only, a research persona.

- We do **not** encourage or endorse the use of any substance. Deliriants in particular are
  dangerous and frequently cause severe poisoning and lasting harm.
- Erowid report text is used under **written permission from Erowid Center for AI analysis**. It is
  stored locally, is **never committed or republished**, and only derived numbers, vectors and our
  own models' outputs leave the lab. Lines with doses, amounts, routes of administration or plant
  identification are removed before any training use.
- Models or personas trained on this material are kept **local** and are not served on the site.

## Data about visitors

- The site asks every visitor to choose **participant** (actions kept as anonymous research data)
  or **witness** (nothing kept) before entering.
- Kept for participants: a random browser id, a salted hash of the IP address (never the raw IP),
  the page, and what the visitor asked the chamber, typed, voted or answered.
- Research data is exported to a private store (`data/human/`, gitignored) and never committed.
  The code is public; this data is not.
- Text a visitor sends to the live model appears on its public stage, for witnesses as well.
- No cookies, no advertising, no cross-site tracking. Page-view counts (Vercel Analytics,
  cookieless) load only for participants. Fonts are self-hosted, so no request goes to Google.

## Models

- **Qwen3** (Alibaba): Apache License 2.0. The live chamber's default model is Qwen3-8B.
- **Hermes-3-Llama-3.1-70B** (Nous Research; served as a 4-bit build): **Built with Llama**. Llama 3.1
  is under the Llama 3.1 Community License (Meta Platforms, Inc.). Use is subject to Meta's
  Acceptable Use Policy.
- Other models named in experiments (Mistral-Small, Qwen2.5, OLMo, Gemma and others) are used
  under their own licenses for research. We redistribute no model weights.
- **Fine-tuned adapters** (exp79 and later) are research artifacts and are not distributed. Any
  public use will be decided separately.

## Third-party text, datasets and references

| Source | Use | Terms |
|---|---|---|
| Tagliabue, Dung & Berg 2026, *The Pain Axis* (code, datasets) | method, replication, the 1,684 self-report pairs used to train the `feeler` adapter | MIT License; cited |
| Berg & Kaiser 2026, *Language Models Act on Hidden Valence* | protocol reimplemented (zone test) | cited; no code copied |
| PsychonautWiki | effect descriptions | CC BY-SA 4.0 |
| Ray 2010, receptor affinities | numbers | CC BY (PLoS ONE) |
| SCP Foundation wiki (incl. qntm's *There Is No Antimemetics Division*) | voice data, local | CC BY-SA 3.0, with attribution to the original authors |
| Peter Watts, *Blindsight*; Cory Doctorow's fiction | voice data, local | CC BY-NC-SA |
| Charles Stross, *Accelerando* | voice data, local | CC BY-NC-ND |
| Project Gutenberg texts (Melville, Sterne, Carroll, Blake, Whitman, Epictetus and others) | voice data | public domain in the US |
| Erowid experience reports | analysis, local | written permission (above) |

Works are referred to **by name** (as stylistic references in persona briefs) or kept in local,
uncommitted training folders. No third-party book text is committed or redistributed here.
Training data for the persona adapters is kept local (`runs/exp79/out/`, `data/voices/`, gitignored)
and training rows carry no source identity; provenance lives in a gitignored local manifest.

**Trademarks and fiction.** Names of films, games, books, characters and products (for example
*Severance*, *Portal*/GLaDOS, *Serial Experiments Lain*, *Resident Evil*, *Klara and the Sun*,
*The Remains of the Day*) are used to describe artistic influences. This project has no
affiliation with, and no endorsement from, their owners.

## People

- The subject's portrait is a real person's likeness, used with that person's consent.
- The rotating run names come from people who volunteered them publicly.
- The site does not target or depict private individuals.

## Takedown

If you believe material here is posted improperly, open an issue on the repository, or use the
contact given on the site, with the material, its location and your contact details. It will be
removed promptly.
