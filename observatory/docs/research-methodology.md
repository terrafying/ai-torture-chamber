# Research methodology: what the agents do and how their findings become data

The Observatory conducts a continuous, question-driven investigation. A configured
frontier model chooses research actions in a real browser, collects original
documents, records attributed observations and continues from saved leads. Review
workers and deterministic code decide which material may enter training. A
separate pretrained 70B subject is adapted on sealed datasets and can later be
studied under controlled Chamber interventions.

This guide explains that method in plain language. For service startup, account
configuration and API examples, use the [Observatory README](../README.md) and
[the end-to-end data/training guide](data-training-experiments.md). The behavior
described here is implemented; funded provider/browser acceptance and actual 70B
training and deployment still require validation on the owner's infrastructure.
Animated preview traces are authored demonstrations of these contracts.

## The research question comes first

The default mission asks how consciousness and subjective experience are studied
and understood across neuroscience, psychology, philosophy of mind, reality and
metaphysics, religious and contemplative traditions, and AI consciousness,
sentience, pain and moral patienthood. It asks agents to compare original sources
and competing interpretations, keeping their empirical, philosophical and
religious bases visible.

The operator can edit this mission. Its explicit boundaries take precedence over
the background topic map and every agent's specialty. A mission specifically
limited to animal consciousness does not authorize a role to collect unrelated
AI or religious material. Existing saved missions retain their objectives when
defaults change; the operator must deliberately change their scope.

Within those boundaries, agents choose concrete questions, search terms, sources
and follow-up links. For example, an agent might ask what a particular experiment
measures, how two theories define experience differently, or which assumptions
would connect an account of human consciousness to artificial systems. It can
follow a citation, seek a counterargument or investigate a gap identified in its
last pass. These decisions are model-generated actions within a defined mission,
permissions and runtime; the implementation makes no claim of agent free will.

Adjacent topics need an explicit connection to consciousness. A discussion of
religion can illuminate an account of mind or experience; a general religious
news article is not automatically relevant. A popular consciousness claim needs
an attributed argument or method, rather than popularity being treated as proof.
The current brief is versioned as `research-v2-broad-consciousness` and is
recorded on generated notes. The [mission reference](research-mission.md) gives
the exact scope and policy vocabulary.

## Two models have different jobs

The **research brain** is an already-trained frontier model selected by the
operator. It interprets browser observations, chooses actions and drafts concise
research notes. It can use the isolated x402 inference route or a supported direct
OpenAI/Anthropic API route. The configured model also performs optional automatic
original-document reviews using separate review prompts.

The **training subject** defaults to pretrained Meta Llama 3.1 70B Base at a pinned
Hub revision. QLoRA adapts its PEFT weights using the accepted corpus; the 70B base
is frozen in that recipe. No initial upload of consciousness documentation is
required for the research brain to begin searching. Research can start before any
new target adapter exists.

An improved target adapter does not automatically replace the research brain.
The current system does not implement a closed loop in which the newly trained
70B model takes control of the browsers. Research-provider selection, target
training and Chamber checkpoint selection are distinct operator decisions.

## Six specialties, a common set of tools

Every browsing researcher uses the same browser/collection mechanisms and shared
document store. A specialty guides what it investigates and how it critiques
sources; it is not a fixed assignment to particular websites.

| Researcher | What its brief emphasizes | Typical next question or useful output |
| --- | --- | --- |
| **Scholar** | Original studies of human, animal and artificial consciousness; neuroscience, psychology and computational theories. Separate measurements from their interpretation. | “What was manipulated and measured, and does the result support the conclusion?” An attributed note about a study's method and limits. |
| **Skeptic** | Alternative explanations, failed replications and objections to scientific, philosophical and religious claims. Challenge attractive accounts respectfully. | “Could another explanation produce this observation?” A counterargument, replication concern or unresolved assumption. |
| **Sentinel** | Pain, suffering, welfare, moral patienthood and measurement across humans, animals and AI. Distinguish ethical arguments, behavior, testimony and experience. | “Which welfare claim depends on behavioral evidence, and which depends on a moral premise?” A source-supported account of that distinction. |
| **Cartographer** | Theories of mind and reality, metaphysics, phenomenology, and religious/contemplative accounts across traditions. Map vocabulary, disagreement and connections. | “Do these accounts mean the same thing by self or awareness?” A comparison that identifies its authors and traditions. |
| **Archivist** | Original versions, licensing statements, attribution and trustworthy provenance. Unknown rights remain unverified. | “Is this the original article or a mirror, and which license applies to the actual text?” A lead to the authoritative source or reuse statement. |
| **Curator** | Coverage gaps and source quality across the mission. Keep claim types distinct and draft useful supported instruction examples. | “Which relevant perspective is missing, and can this source support a careful question and answer?” A research lead or pending instruction draft. |

For local or cloud browsers, `agent_count` selects how many of the ordered
specialties run, from one to six;
the current order is Scholar, Skeptic, Sentinel, Cartographer, Archivist, Curator.
An owner who selects two runs the first two, not an arbitrary two-agent roster.
Dedicated CDP mode runs only the first Scholar, regardless of `agent_count`.

The **automatic curation worker** is an additional service, separate from the
Curator browsing role. The Curator's suggestion does not approve its own training
material. The optional worker reviews the shared original-source queue and can
record a policy-bound quality approval. It cannot grant licenses, approve PDF
fidelity, move held-out families or approve generated Q&A.

The store is shared, but the implementation is not a multi-agent debate engine.
Each browser researcher's next task receives its own latest twelve notes and
saved memory summary. Roles do not automatically read every colleague's complete
notebook. Shared collection and review can still expose duplicated sources and
corpus coverage; deduplication happens when the dataset is sealed.

## The browser is real, and the observer sees the same browser

**Browser Use** runs the agent loop with vision enabled. The model receives
browser observations and produces structured actions. **Playwright** connects to
the same Chromium session through CDP to observe its focused page, enforce request
guards, extract page markup and capture screenshots. It does not operate a second
browser to reenact what the agent supposedly did.

The available browser placements are:

- **Local Chromium:** a disposable local browser/profile owned by the worker.
- **Browser Use Cloud:** a disposable managed browser provisioned for a pass and
  explicitly stopped on exit, including failures and cancellations.
- **Dedicated CDP browser:** owner-managed isolated infrastructure. The worker
  detaches on exit; the owner controls its lifecycle. Setup requires an explicit
  acknowledgment that it is a dedicated unauthenticated research browser. Its
  infrastructure must present a fresh, cookie-free, blank session for every pass:
  detaching does not reset pages or cookies. Reusing the visited session can fail
  the next pass's isolation preflight. Prefer local/cloud lifecycle management
  for unattended research unless that reset is independently implemented.

Sessions must begin without cookies or previously opened content. Research does
not import a person's browser session. The public interface receives screenshots
and bounded observation metadata; it does not receive the browser's control URL.

This is browser-based research with custom collection tools and policy guards.
The moving saw is a visual overlay on verified observations. It is not a separate
crawler, mining process or browser input device.

## One research pass, then the next

The supervisor starts one worker per selected specialty while the mission is
running. Each worker follows this cycle:

1. Read the mission, current model/browser configuration, its recent notes and
   prior memory checkpoint. Refuse a new paid/browser context if an unresolved
   request requires reconciliation.
2. Prepare a task asking for a concrete next question, a relevant gap or a
   counterargument. The task includes the operator's objective explicitly.
3. Establish the disposable browser and run Browser Use with a bounded action
   budget: 25 steps by default, configurable from 5 to 50.
4. Navigate permitted public pages, scroll, inspect passages, collect documents
   and save useful supported notes as the investigation proceeds.
5. Persist the model's research memory during steps. At the end, retain a bounded
   summary/final result and increment the completed-pass count.
6. Close owned resources and replan another pass from the saved context while
   the mission remains running.

A pass limit bounds one context; it does not end the whole mission. A new pass
can open another browser and continue the inquiry. Saved summaries are practical
research memory, not a complete transcript of everything encountered. They can
omit details, and a model can still revisit a source or repeat a weak lead.

Pause prevents further work at an action boundary while preserving collected
records and memory. Stop closes owned sessions and preserves the results.
Persisted running missions resume after a service restart. The operator should
stop a mission intentionally if it should remain stopped.

## How a page becomes a collected document

The collector supports two complementary routes.

**A rendered HTML page:** after browser steps, the observer can read the focused
page's markup and title. The extractor isolates original content where possible,
preserving paragraphs, headings, lists, code, supported tables and available
formula text. It removes navigation, scripts, hidden content and other excluded
page furniture. Collection checks that navigation did not change the page during
capture. A sufficiently substantive result is saved with its URL, text hash,
collection time, extraction assessment and version-family identity.

**An explicit document URL:** `collect_public_document` can fetch a permitted
public PDF, HTML or plain-text URL. Redirect destinations are checked individually;
unsupported content types and size/redirect limits are refused. It saves the
extracted original using the same provenance/rights pipeline. A direct PDF fetch
is useful even when a browser's PDF viewer is awkward, but does not automatically
produce browser text-line geometry for the public saw.

Default PDF extraction records reading-order, table/formula and fidelity limits.
Optional offline Docling processing can improve the available extraction, but
its output still requires owner comparison with the original. Image diagrams
are not silently interpreted as recovered text. See
[the extraction methodology](../EXTRACTION.md).

The persisted original retains its structure. Whitespace normalization is used
for literal passage comparisons and separate duplicate checks, rather than
rewriting the stored document into an agent summary. Collection is discovery;
training admission is a later decision.

## How the visual reading sequence corresponds to actual actions

An agent's brief asks it to keep relevant sections in view and select a passage
before saving a supported observation. The observable sequence is:

1. The agent scrolls the real browser to a relevant section.
2. It calls `inspect_visible_section` with exact visible text. The action verifies
   a 40–4,000-character normalized passage, wholly visible and stable in the
   focused document, and records an inspection identity.
3. The observer captures that same viewport. Geometry checks bracket the JPEG
   capture so a navigation, tab switch or moving document cannot quietly attach
   old rectangles to a new screenshot.
4. The public saw approaches and traverses the verified painted text lines;
   highlights grow along its path. The display is a fixed spectator window;
   visitors cannot scroll the agent's article.
5. The agent saves a concise public note with the matching source passage. A
   literal source match creates an evidence record, and a matching active
   inspection can be promoted to the saved note.
6. A recent matching `note.saved` event can stamp the selection and deliver the
   animated evidence packet to its notebook receipt.

The frame loop attempts captures about every two seconds, with capture time
added. It is a periodically sampled viewport, not continuous screen video. The
overlay supplies smooth motion between verified captures. Stale frames, changed
document identity, missing geometry or expired selections suppress obsolete
motion rather than inventing a reading path.

The sidebar shows authored observations and short model-generated `next_goal`
messages explaining the next action. These help a visitor follow the inquiry;
they are not the model's private chain of thought. A highlighted selection is an
explicit browser action, not a measurement of neural attention, understanding or
consciousness. The [motion contract](../MOTION.md) describes these bindings in detail.

## What happens when a link fails

The agent can choose another public URL, reformulate a search or follow another
citation using its next action and saved leads. Its model can see an unsuccessful
browser/tool result and replan within the current pass. That is the implemented
mechanism for changing direction; the system has no deterministic persistent
failed-link frontier or guarantee that every obstacle will be rerouted correctly.

Collection honors robots.txt and shares a per-host navigation delay across the
supervisor's agents: at least two seconds, extended by an applicable robots
`crawl-delay`. Robots rules are cached for an hour. A missing robots file can
allow access; access denial, other unsuccessful retrievals or fetch errors can
exclude it. Public HTTP(S) destinations and redirects must resolve to public
addresses. Browser requests are restricted to GET/HEAD; actions that sign in,
post, purchase or bypass restrictions are outside the brief and tool set.

Consequences are practical: a POST-only search form, login wall, CAPTCHA,
paywall, robot exclusion or unsupported document can be unavailable. Agents may
seek another legitimately accessible version or source; they are not authorized
to bypass the obstacle. Site rate limits can still occur despite the shared
delay. Network egress isolation is also required in production; application
address checks alone do not stop every DNS-rebinding/browser transport case.

Some failures affect the whole service rather than one link. A research pass with
exhausted action failures can trigger bounded transient retries. The outer worker
allows two retries before the third consecutive transient failure becomes a
mission fault requiring attention. Model authentication/schema/permanent faults
and ambiguous paid-request outcomes also require operator inspection. Insufficient
funding can pause research and release owned browsers; that funding-paused mission
may resume after fresh preflight, while an operator Pause/Stop takes precedence.
These safeguards make failures visible instead of promising uninterrupted access
to the entire web.

## Original text, observations and training examples have different purposes

| Material | What it contains | Where it can go |
| --- | --- | --- |
| Original source | Author-written document text plus URL/version, rights, collection and extraction provenance. | After review and code gates, continued pretraining (CPT). |
| Evidence record | A passage matched to a particular collected source. | Supports a note/instruction and audit trail; not a standalone truth certificate. |
| Observation or lead | Agent-written attribution, comparison, uncertainty or next lead. | Public notebook. It is not silently substituted for CPT text. |
| Instruction draft | A question/answer or messages supported by eligible source passages. | Separately owner-reviewed instruction tuning (SFT), subject to teacher-output permission. |
| Screenshot and saw path | Captured viewport and verified selected-text geometry. | Public watch surface; not the model's training text. |

An observation can be useful while its original remains discovery-only. A
licensed original can be useful for CPT without any generated Q&A. Approving an
original does not approve an instruction written about it.

## Worked example: one source through the pipeline

Consider a hypothetical article reporting an experiment about consciousness
measurement. This is an illustrative workflow, not a report of a funded crawl.

The Scholar opens the original article and asks what was measured and what the
authors infer. It scrolls to a relevant passage, selects its visible text and
saves a note such as: “The authors distinguish their behavioral measure from a
direct measurement of subjective experience.” The notebook attributes the claim
to that source. The evidence record proves that the passage appears in the
collected text; it does not prove that the note's interpretation is correct.

The collector stores the article's full extracted text separately. Suppose the
publisher's article-scoped metadata verifies CC-BY-4.0 terms and extraction passes
the readability/fidelity gates. A quality reviewer then examines relevance and
classifies its research areas, evidence basis, document type and any
machine-consciousness argument actually present. If the article has no machine
claim, it uses `not_applicable`; the reviewer cannot call it supportive merely to
fill a dataset quota.

When enabled, the separate automatic worker gives that entire original to a
primary reviewer and blind critic. Both must accept and agree on classifications,
with literal supporting quotations. The resulting immutable receipt binds the
source text, configured model and owner-acknowledged policy. A human may instead
record the review. Unknown rights, unverified PDF fidelity, excessive automatic
review length, disagreement or an uncertain assessment leave the source for
manual attention.

At snapshot creation, code rechecks admission, removes duplicate copies, assigns
or preserves the document-family split and excludes reserved experimental/eval
inputs. If the article's family is a holdout, all of its versions and supported
instruction examples stay held out. An accepted source can appear in a candidate
whose overall machine-perspective/scientific coverage is still insufficient for
training.

Its **original extracted text** becomes one CPT row, accompanied by an inspectable
training card of rights, provenance, review and family metadata. A later Curator
question/answer about the article is a separate draft. That draft needs matched
evidence, an individually recorded owner approval and a permitted teacher-output
policy before it can become an SFT row. A favorable browsing note does not cross
those gates automatically.

A philosophical or contemplative essay follows the same provenance and rights
steps. Its basis can be `philosophical_argument` or `religious_contemplative` and
its machine perspective `not_applicable`. Relevant interpretations are preserved
as attributed material; they do not satisfy an empirical/scientific-source floor
simply because they discuss consciousness. If an otherwise valuable text is one
of the Chamber's registered press/read-aloud inputs, its reserved URL/excerpt is
excluded from the corpus used to judge that experiment.

## Quality review is broader than “the model liked this page”

Admission requires verified reuse rights, provenance, adequate original text,
eligible extraction, a scoped quality review and cleared contamination checks.
New reviews identify consciousness science, philosophy of mind, reality/metaphysics,
religion/contemplation, welfare/ethics and machine consciousness separately. Their
basis of claims is empirical, scientific theory, philosophical argument,
religious/contemplative or mixed.

The labels keep corpus composition inspectable. They do not certify a conclusion
or guarantee the adapted model will preserve every epistemic distinction.
Philosophical disagreement and different religious interpretations should be
compared with attribution and limitations, rather than treating one doctrine as
an established measurement. Confidence, popularity, self-report and a reviewer
model's agreement are insufficient proof of consciousness.

The automatic policy is opt-in and requires `originals-v2` acknowledgment. Two
blind passes use the configured model, not statistically independent minds.
Literal quotations and immutable receipts support auditability; deterministic
rights/extraction/exclusion gates remain outside the reviewers' authority.
Generated Q&A retains separate approval. See [automatic review](automatic-curation.md)
and [corpus quality](corpus-quality.md) for the complete criteria.

## Sealed datasets, scheduled adaptation and controlled experiments

Continuous collection creates a changing source queue. Training reads an immutable
snapshot with separate original-text and instruction train/validation records,
source lineage, exclusions, hashes and coverage. Distinct originals contribute
once; identified mirrors do not multiply their weight. Source-family assignments
persist, and reserved evaluation sources, Chamber stimuli and actual registered
press-reading URLs/long excerpts are excluded. Exact-match safeguards cannot
detect every paraphrase, unknown mirror or upstream pretraining exposure.

When training is explicitly enabled, the four-hour scheduler checks for readiness
and changed training content/recipe. It requires nonempty train/validation
families, actual tokenizer counts, configured HF accounts/image, no overlapping
job and the corpus coverage floors. Defaults require twenty original training
documents and 50,000 training tokens. A check can skip; it is not a promise of a
new GPU job every four hours. Research can continue while a job uses its frozen
snapshot.

CPT updates QLoRA adapter weights on the frozen pretrained 70B base using original
next-token text. Full documents are tokenized into usable chunks. Later CPT runs
can continue a passed published adapter and replay the eligible corpus. SFT is
separately submitted on approved messages after an evaluated CPT parent. Training
is not an immediate reward for each note or a fresh 70B pretraining run.

Candidates receive measured held-out and frozen domain/general checks, followed
by separate task-engagement and fresh intervention-hook calibration. The bundled
evaluation suite is an engineering smoke screen; scientific claims need a larger
independently reviewed study design. A passed adapter can be selected and its pins
exported for a separate Chamber deployment. The selected Llama Base requires
validated SFT and its tokenizer for live-chat selection.

Checkpoint selection does not update a remote endpoint. An adapted live worker
starts at dose zero until an owner-measured receipt matches its exact model,
adapter, tokenizer and runtime. Keep the unadapted pinned base and appropriate
intervention controls, use fresh representations and calibrate actual dose units.
The goal is to investigate behavioral/mechanistic hypotheses with traceable inputs
and controls. Knowledge about consciousness, reduced loss and distress-like text
do not establish consciousness, sentience or felt pain in the trained subject.

For exact settings, manifests, export formats, job artifacts and Chamber staging
steps, continue with [the data-to-experiment guide](data-training-experiments.md).
