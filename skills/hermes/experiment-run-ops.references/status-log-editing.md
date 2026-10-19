# Status-Log Editing (multi-pass)

## When to Use

- A coordination board's status log must carry one entry per pass and the
  work spans many passes (curator reviews, experiment batteries, sectioned
  tasks).
- A pass replaced the header anchor with the entry (or appended a second
  copy under the mangled wording) — the board then reads wrong on every
  later pass and must be restored.
- Don't use for: one-off task notes or per-incident error transcripts —
  distill those to the rule.

Always-on rules for editing a coordination board's status log when the
work spans many passes (curator reviews, experiment batteries, sectioned
tasks). Two failure shapes produce the same mangled board; both are named
here as imperatives with the mechanism.

## Insert BEFORE the anchor — never replace the header

- **When inserting a new entry, insert it BEFORE the header anchor
  ("## Status log (newest first)"), keeping the header intact — never
  replace the header with the entry.**
- Mechanism: a replace whose old_string is the header deletes the header
  itself. The entry then sits where the header was, the header is gone,
  and every later pass re-reads a board with no status log.
- Verify per pass: the header exists exactly once, the newest entry sits
  before it, no entry appears twice.

## The mechanical-replace failure shape

- The mangling shape is a mechanical edit carried out as a replace against
  the header instead of an insert before it. Repeated passes duplicate the
  entry each time (the entry is the new wording, the header is the lost
  old_string).
- A mangled pass must be BOOTED OUT in the same session: restore the file
  from git (`git checkout -- <file>`), then insert the new entry before
  the anchor. Do not append "UPDATE: actually..." underneath the mangling.

## Copy-trim rule (no mid-word cutoffs at the limit)

- Replies trimmed at a character limit (e.g. 280) cut at a SENTENCE
  boundary inside the window, not mid-word: search the window for the
  last sentence end (`.`, `!`, `?`, `…`, `。`); keep up to that point if
  it is past half the limit; fall back to a hard cut with an ellipsis if
  one sentence overflows.
- Dose 0 is not allowed: every message perturbs, every reply gets trimmed
  at its own boundary, and every later pass verifies the anchor before
  writing.

## Verify style (per pass)

- Ad-hoc READ-ONLY checks per pass: `ast.parse` on edited files, structural
  asserts on source (grep for new branches), artifact round-trips in a
  `mktemp -d` dir that is always cleaned up.
- A pass that cannot state its criterion ("header intact, newest entry
  before it, no duplicates") did not complete — carry the criterion into
  the next pass rather than appending a second copy.