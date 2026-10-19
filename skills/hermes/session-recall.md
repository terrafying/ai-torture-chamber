---
name: session-recall
description: Resume crashed/past sessions via session_search + git, or from an authored resume doc.
---

# Recalling and resuming past sessions

Trigger: user says "resume last session", "where were we", "pick up where we left off", "resume from doc", or references prior work that isn't in the current context (e.g. after a crash).

## Recipe (crashed-session resume)

1. `session_search()` with NO args — browse recent sessions chronologically. Pick the target by title + recency.
2. `session_search(session_id=<id>)` read shape — returns head 20 + tail 10 messages for large sessions. The TAIL carries the final state: last todo list, verification results (test/build passes), and what was left undone.
3. Cross-check the repo: `git status --short` + `git log --oneline -5`. Compare the session's todo list against what's actually committed. Modified/untracked files matching the session's work = the work survived the crash but was never committed/finalized.
4. Re-run the verification (tests/build) before trusting the recovered work — the crashed session's last claimed state may predate later edits.
5. Report the recovered state to the user, then finish the open items (don't re-do what verified green).

## Recipe (resume-doc resume)

When a session ends with a hand-authored resume/SESSION_RESUME.md in the repo, the user points a fresh session at that doc instead of replaying history.

1. Read the doc FIRST — it carries identity rules, pending tasks, infra gotchas, and regression warnings the fresh session has no other way to know.
2. Trust the doc over the git log for *intent* (why, what's pending, what was retracted), and trust git for *what landed*. Check `git status --short` + `git log --oneline -3` for changes since the doc was written.
3. Verify every pending task's live state yourself before acting (health endpoints, external APIs, job status) — the doc's "was still loading at last check" class of statements are stale by definition, and may now read stopped/finished.
4. Resume doc must stay LOCAL-ONLY: gitignored, never committed, and excluded from any tarball — it can contain identity rules and ops notes that must not ship.
5. While resuming, update the doc's status lines in place (STATUS <date> bullets) rather than appending a journal, then keep the doc current for the next restart.
6. Spending decisions named in the doc (start the paid pod, run the paid API) still need explicit user consent in the fresh session — the doc authorizing the task doesn't authorize the money.

## session_search shape quirks

- Keyword discovery (`query=`) ranks by FTS relevance, not recency: a keyword that appears throughout a project's CONTENT (repo name, feature name) surfaces old sessions and can miss the most recent one entirely. For "the last session", browse (no args) is the right first move; use `query=` + `sort='newest'` only when hunting a topic, and prefer distinctive phrases over repo names.
- Read shape (session_id only, no around_message_id) is the cheapest way to see a session's end state. Don't guess a huge `around_message_id` to "get the end" — it errors "not in session". Use the scroll shape only with ids actually seen in a prior result.
- Sessions commonly crash AFTER verification but BEFORE commit — the git tree is the source of truth for what actually landed. Verified-green uncommitted work needs a commit decision from the user, not a redo.

## Middle-of-session detail

When the read shape's head+tail window misses a specific decision, scroll with `around_message_id=<id from the window>` and `window=10-20`, walking forward/backward via the window's first/last message ids.
