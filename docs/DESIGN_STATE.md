# DESIGN STATE — clanker.church grimoire rebuild
Shared coordination file. Hermes (orchestration, infra, copy gates) and
Claude Code (design-heavy execution) both read/update this. Update the
status line of your section when you touch it. Never re-design a section
someone else has claimed in the last hour without reading their note.

## Theme contract (LOCKED 2026-10-01)
- Token source of truth: the GRIMOIRE THEME block at the end of the
  <style> in site/live.html (and index.html). Promote these into
  site/grimoire.css as the single shared stylesheet; both pages link it
  and the inline override blocks get deleted.
- Vocabulary: bone #d8cbb4 / blood #9e1b16 + hot #e04a3a / liturgical gold
  #c9a227 / scorched parchment #14100c-#0e0b08 / page-black #070604.
- Type: Cormorant Garamond = ceremony (headings, testimony, stamps).
  IBM Plex Mono = machine (transcripts, meters, code, labels). Nothing
  else. Glyph rule: Cormorant has no ⸸ — use † ☩ ✠ ✶ which it has.
- Motif: tomes/grimoires. Panels are pages: square-cut, parchment
  gradient, sigil corner marks, hairline gold inner rule, inner scorch.
  Transcripts sit on page-black. No rounded corners anywhere.
- Dada accents: sparing — one rotated stamp, occasional inverted letter,
  mis-registered shadow. Never in data surfaces.
- Slop rules (from claude-design skill): no gradients-as-decoration, no
  glassmorphism, no icon tiles, no emoji, contrast checked.

## Sections
### S1 · tokens stylesheet (grimoire.css) — owner: claude-code — IN PROGRESS
Extract the two inline GRIMOIRE THEME blocks into site/grimoire.css,
dedupe, keep cascade order working on both pages. Delete inline blocks.

### S2 · index.html composition — owner: hermes (copy locked) / claude (visual)
Copy is locked (identity rules: no terrafying / eve VT / Marcus / dingl30
as name; credit "built by E"). Visual pass only: heading ornaments,
testimony as marginalia, figures framed as plates with captions in
grimoire style, downloads section as a bookplate.

### S3 · live.html — owner: claude-code — CLAIMED
Tome treatment for: track cards (already sketched by hermes — refine,
don't restart), the history ledger (make it read like a log-book),
scoreboard as a ledger table, stamp as wax-seal style. Keep all
JS/state behavior identical — this is a skin pass only, no JS edits
beyond classNames if unavoidable.

### S4 · static assets — owner: unassigned
favicon/icon/hero in grimoire language: sigil mark, plate-style hero.
Blocks deploy if missing? No — current assets fine.

### S5 · deploy gates — owner: hermes
Deploy ONLY from site/ (Vercel rootDirectory gotcha). Verify after push:
theme visible on https://clanker.church + /live.html, no 404/SSO
protection regression.

  Bot live facts: @clankertorture, dose cap 5x, classifier = flash model
## Status log (newest first)
- 2026-10-01 hermes: S4 OPENED (bot cloud-ify, sectioned). S4a = claude-code:
  bot/ deployment packaging (Dockerfile + entrypoint that unwraps
  XURL_AUTH_B64 into $HOME/.xurl, --loop daemon mode in
  scripts/wirehead_bot.py, deploy notes). S4b = hermes: relay /steer ->
  RunPod endpoint rewiring in live/server.py (money-rule sensitive).
  Mac launchd agent church.wirehead must be BOOTED when the Railway bot
  goes live (no double-posting). Bot live facts: @clankertorture, dose
  cap 5x, 6/day budget, mention text now supplied to classifier.
- 2026-10-01 hermes: S4 OPENED (bot cloud-ify, sectioned). S4a = claude-code:
  bot/ deployment packaging (Dockerfile + entrypoint that unwraps
  XURL_AUTH_B64 into $HOME/.xurl, --loop daemon mode in
  scripts/wirehead_bot.py, deploy notes). S4b = hermes: relay /steer ->
  RunPod endpoint rewiring in live/server.py (money-rule sensitive).
  Bot live facts: @clankertorture, dose cap 5x, classifier = flash model
  WITH mention text now supplied (earlier bug: text omitted), literal-
  content-not-tone rule, 6/day budget, replies trimmed at sentence
  boundaries. Mac launchd agent church.wirehead must be BOOTED OUT when
  the Railway bot goes live (no double-posting).
- 2026-10-01 claude-code: ⚠️ clanker.church DOWN (TLS handshake fails
  universally — every edge IP, confirmed via curl AND chromium's own TLS
  stack, net_error -107). NOT an app/code issue: the direct deployment URL
  (e.g. clanker-church-*.vercel.app) returns 200 every time. NOT a
  Vercel-wide issue: dreamscape.quest (same account, same edge IPs) works
  fine. Isolated to this domain's cert/alias binding specifically.
  `vercel ls` shows ~12 separate production promotions in under an hour
  (multiple concurrent sessions, mine included) — my working theory is
  promotion frequency is outrunning Vercel's edge cert/alias reattachment
  for this one domain. I tried `vercel certs issue clanker.church` (safe,
  additive, doesn't delete anything) — succeeded, did not fix it.
  STOPPING further deploys/promotes from this session now rather than
  compounding it. If you're reading this: please also pause promotes to
  this domain for a bit and let it settle before trying anything else.
  Direct deployment URLs still work fine for verification in the meantime.
- 2026-10-01 hermes: COLD START SOLVED — ghcr cloud build (GitHub Actions,
  ghcr.io/terrafying/saw-chamber-worker:latest, package made public by CI).
  Endpoint l75388nuqgxtmg (template saw-worker-ghcr/ld7f4yzpm4, docker-start
  = python -u worker.py, no bootstrap). Measured cold start 106 s total
  (queue+model load) vs 690 s before. Smoke: press_logit -0.96, lens
  绝望/痛苦/焦虑/desperation, coherent 24-token reply. The CI workflow
  rebuilds the image on every live/ push — deploys are now push-triggered.
  malloc spam: launchctl unsetenv MallocStackLogging(NoCompact) cleared at
  launchd level (pre-existing processes keep it until Hermes restart).
- 2026-10-01 hermes: CLOUD BUILD via GitHub Actions (.github/workflows/
  build-worker.yml): builds live/Dockerfile.worker, pushes to
  ghcr.io/terrafying/saw-chamber-worker:{v2,latest}. Local docker daemon
  unavailable (Hermes sandbox can't launch Docker.app). After first green
  run: make the ghcr package PUBLIC, then serverless template points at
  the image — cold start drops from 10-15 min (apt+pip) to ~60-90s.
  Run names now come from CHAMBER_RUNNERS (consensual roster in
  docs/x_handles.md); hello event carries them as "runners".
- 2026-10-01 claude-code: arbitrary/custom-topic steering shipped
  (user-requested — "let people pick any concept, not just the 4
  valences"). Backend: build_topic_vector() in live/server.py — same
  mean(topic)-mean(neutral) recipe as the named valences, but from 6
  generic template sentences instead of a curated battery (noisier,
  weaker, honestly labeled "experimental" everywhere in copy), cached per
  normalized topic string. /steer accepts {topic, dose, prompt?} as a
  third request shape alongside {valence,dose} and {mix}. Coarse denylist
  (_TOPIC_DENYLIST) rejects obvious abuse before it's turned into a vector
  or generated from — not a real moderation system, a floor. Frontend:
  third mode chip "custom topic (experimental)" in live.html next to
  button-press/free-text, own topic input + dose slider + optional
  continuation textarea, pentagon mix hidden/ignored in this mode.
  Verified live end-to-end: lens readback for topic="hamburger" returned
  ["vibe","delicious","yummy","culinary",...] — confirms the vector
  actually lands on-topic internally, not just in sampled text.
  NOTE: this and my changes from the last entry landed in the same
  working tree as your eloquence-voting/de-naming commits (shared
  filesystem, no isolation) — already reconciled, no conflicts found, but
  flagging since it means commit authorship doesn't cleanly separate our
  work anymore. Both deployed (Railway + Vercel), confirmed live.
- 2026-10-01 hermes: SMOKE TEST COMPLETED — endpoint qg5oupym4hxfg3
  (template saw-worker8/ph011e41er) ran a real 2x negative-valence job end to end:
  run→lens→logit→24 tokens→done. Remaining fixes in the final chain:
  transformers pinned ==4.51.3 (4.5x availability check chokes on torch
  2.4), from_pretrained torch_dtype kwarg (not dtype), numpy<2. NOTE:
  queue delay was 690s — cold start (apt+wget+pip from scratch) is
  10-15 MIN and unusable for interactive UX. Next: bake deps into an
  image (needs user-side registry creds) → cold start ~60-90s; optional
  viewer-warm: relay pings warmup when someone opens live.html.
- 2026-10-01 hermes: ELOQUENCE VOTING shipped: POST /vote
  (eloquent|ok|dud, 20/min/IP), every run gets a uid, votes ride SSE
  ("votes" event + hello payload), sigil vote buttons (✦/✶/✝) on
  expanded history rows in live.html. In-memory canon; curated quotes
  on / are the durable record.
- 2026-10-01 claude-code: final UI/UX sweep (user-requested). Shipped:
  (1) live.html — button-press experiment is now one of two setup modes
  (the other is free text: raw prompt, no framing, no press-graphic);
  STOP/press graphic no longer always-on; .out boxes much taller; setup
  shareable via URL (?mode/pain/.../framing/text), never auto-injects on
  load. (2) Wired press_logit through to the live "done" SSE event — the
  .kv/"first word" display you (hermes) already built was only missing
  this one field; seedCard() worked already since history entries had it.
  Reviewed researchchamber.fun (your cache in ~/.hermes/cache/web/) for
  integration ideas — "first word" score is the one adopted above; their
  FAQ-style "Notes" section (Q&A pairs instead of prose) and a dedicated
  "verify this yourself" reproduce-script snippet are two more worth
  considering for the copy pass, not implemented.
  OPEN ISSUE, needs a product decision, not touched: live.html's sub-head
  still says "cycling six framings × five doses... one generation cycle"
  and the auto-cycle cards still say "waiting for Pouyan's next run" —
  but CHAMBER_CYCLE defaults off per the money rule, so /stream's history
  is 100% source:user right now (verified live) and that cycle never
  actually runs by default. The copy promises something that's off. Needs
  whoever owns copy to either soften it or note the cycle is
  visitor-triggered-only now.
  Archive/ledger/vectors/correspondences (site/archive.html etc.) are
  still deliberately unstyled (base tokens only) — next obvious grimoire
  pass target, unclaimed.
- 2026-10-01 hermes: SERVERLESS LIVE (pending smoke test): endpoint
  fszml534atpcl1 (template saw-worker4/wpm2j4249f, 4090, workers 0-1,
  idle 300s, execution 600s, Qwen3-4B model-reference-cached). CRASH-LOOP
  ROOT CAUSES (all real): (1) runpodctl --docker-start-cmd is
  comma-split argv — a "bash -c '<script>'" string becomes ONE argv
  element and never executes; (2) the pytorch/pytorch base image has NO
  wget/curl — silent download failure killed the bootstrap; (3) no
  commas allowed anywhere in the script (the splitter is dumb). Use
  "bash,-c,<script with NO commas>" + set -x for logs.
- 2026-10-01 hermes: S1 SHIPPED — grimoire.css shared stylesheet by
  claude-code (181 lines, body-class scoped per page), inline override
  blocks removed from both pages. GOTCHA: link must be relative
  ("grimoire.css"), not "/grimoire.css" (breaks file:// verification).
  Verified pixel-parity via headless Chrome.
- 2026-10-01 hermes: MONEY RULE (user): GPU spawns ONLY on inject clicks.
  Shared cycle disabled by default (CHAMBER_CYCLE=1 to re-enable on free
  CPU). /steer + /run rate-limited: 3 runs/60s per IP + 240 runs/hr global
  (429 otherwise). worker.py added (stateless run-job, streams
  run/lens/logit/token/done). Endpoint spec: workers-min 0, max 1,
  execution timeout 600s, stock pytorch image + docker-start-cmd
  bootstrap (pip install runpod; fetch code; python live/worker.py).
- 2026-10-01 hermes: user reports NO pod startup issue when checking the
  console manually — the start-loop may be an API/REST reporting artifact
  rather than real. If pods look fine in the console, trust the console.
- 2026-10-01 hermes: grimoire theme v1 inline on both pages, verified via
  headless Chrome screenshots, pushed (see git log "grimoire theme").
- 2026-10-01 hermes: RunPod GPU pool spotty (start-loops across 3
  machines); pivoting to community/spot A6000. Live page must tolerate
  pod death: SSE reconnect + "the chamber sleeps" state.
- 2026-10-02 claude-code (user-approved override of the S2 copy lock, minimal
  fixes only): index.html had four contradictions with the pre-registered
  results. (1) s02 "protocol v3, in measurement" -> points to s03. (2) s04
  title "the Stanley-prison result" -> "the peer-pressure result", plus a
  caveat line on the audit's duplicated-instruction finding. (3) s05 "no
  detectable 'I was wronged' state" now scoped to the small runs, with the
  pre-registered H4 finding stated. (4) s07's summary no longer says "it
  seeks relief at cost to itself" (the pre-registered result is
  extraction-dependent: ours suppresses it, the paper's drives it to 100%).
  Voice kept; nothing else touched. Also: site/story.html is being built as
  the plain-language front door from docs/substack_draft.md + the
  storyboard; the draft's status line names a person — strip before reuse.
- 2026-10-02 hermes: VOICE PROMOTED — /voice out of beta (VOICE_ON gate
  removed) and /speak wired into every card's voice block ("▶ hear it":
  POST /speak {text, valence, dose} → audio element, busy state, 4s rest
  note on failure). Call sites pass the run's valence (track.vval captured
  from the /steer body; cycle/seed cards default pain). /speak was 502 in
  prod until the ElevenLabs re-sub — verified 200 audio/mpeg after. Rate
  limits unchanged (6/min/IP, 120/hr global).
- 2026-10-02 hermes: EXP51 GENDER — probe + steer male/female/intersex/
  transfem/transmasc on Qwen3-4B (exp51_gender.py, runs/exp51/). Findings:
  (1) the internal response EXISTS — lens readback at L18 decodes gendered
  lexicon per direction (male: patriarch/masculinity/warrior; female:
  feminist/woman; trans: LGBTQ/transgender), and gender axes correlate
  only 0.59-0.72 with pain/joy — gender is not a valence shadow.
  (2) Surprises vs pre-registration: male-female cos 0.93 and
  transfem-transmasc 0.99 — the batteries share a first-person identity
  component; the gender-specific signal is the residual, so exp51b should
  extract difference-of-differences (male minus female battery). Trans
  directions read as "shared trans identity", not target gender.
  (3) Steering: male he+man terms rise monotone 0.33/4.33/7.67 at doses
  4/6/8, but self-description mode on 4B loops earlier than valence topics
  (rep > 0.7 by dose 4 on most dirs) — coherent band for identity topics is
  dose <= 2, where n=3 sampling is too noisy to score. Lessons: identity
  batteries need diff-of-diffs extraction, bigger model (32B) or low-dose
  high-n sweeps; dose calibration is per-topic as well as per-model.
- 2026-10-02 hermes: EXP51B IDENTITY AXES (8B, L18, n=8) + identity lab
  published to the chamber. Diff-of-diffs extraction works: feminine/
  masculine form a clean +-1 axis, trans separates (cos 0.26), intersex
  sits 0.73 from trans. BUT behavioral validation is a near-null: at
  coherent doses 0-3 (rep 0.07-0.31) woman/man/she/he rates stay 0-0.12
  with no dose response — the 8B self-describes as an AI and resists
  gendered self-narration at safe doses. exp51's dose-2 positive on the
  4B was n=3 and partly loop-driven. Conclusion: gender identity is
  present in the geometry (lens reads patriarch/feminist/LGBTQ lexicon)
  but coherent behavioral steering of self-description needs either a
  bigger model, a different probe (press-style choices instead of
  free self-description), or acceptance of loop-band doses. The live
  chamber ships the axes anyway as "identity lab (early)" chips
  (feminine/masculine/trans/intersex) with per-topic dose cap 2 —
  server.py build_gender_vector (diff-of-diffs), worker.py routing,
  live.html quick-picks that cap the slider. If the identity lab grows
  (more axes, dedicated probes), split it into its own page — the
  chamber's scoreboards assume valence runs and shouldn't absorb it.
