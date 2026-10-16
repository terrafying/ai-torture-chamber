#!/bin/bash
# interp_executor.sh — local runner for the interp-watch research loop.
#
# Runs on an idle machine (mini-mini) from a launchd job. NO LLM in the loop:
# it executes the experiment scripts that the cloud "interp lit-watch +
# experiment design" routine queued on the `interp-watch` branch, captures
# their output verbatim, and commits the results back. The cloud brain reads
# those results on its next run and interprets them.
#
# For each runs/interp_watch/<thread>/RUNME.sh with no RESULTS/<thread>.done
# marker: run it with the interp venv on PATH, tee stdout+stderr to
# RESULTS/<thread>-<UTC>.log, write the .done marker, commit, and (if anything
# ran) push. Experiments that need a GPU — RUNME.sh matching GPU_?POD, or a
# .gpupod file beside it — are skipped; those wait for a RunPod run.
#
# Env overrides: INTERP_REPO, INTERP_VENV, INTERP_MAXSEC (per-experiment cap).
set -uo pipefail

# Dedicated executor clone — NOT the user's working clone. It is only ever on
# the interp-watch branch and is safe to hard-reset/clean, so a dirty user tree
# never blocks the run. Create once: git clone <repo> ~/interp-exec/ai-torture-chamber
REPO="${INTERP_REPO:-$HOME/interp-exec/ai-torture-chamber}"
VENV="${INTERP_VENV:-$HOME/venvs/interp}"
MAXSEC="${INTERP_MAXSEC:-3600}"
BRANCH=interp-watch
WATCH="$REPO/runs/interp_watch"
RESULTS="$WATCH/RESULTS"
LOCKDIR="$HOME/.interp_executor.lock.d"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# Single instance: mkdir is atomic on macOS (no flock here). Stale lock > 6h is cleared.
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  if [ -n "$(find "$LOCKDIR" -maxdepth 0 -mmin +360 2>/dev/null)" ]; then
    log "clearing stale lock"; rm -rf "$LOCKDIR"; mkdir "$LOCKDIR" || exit 0
  else
    log "another run in progress; exit"; exit 0
  fi
fi
trap 'rm -rf "$LOCKDIR"' EXIT

# Portable timeout: GNU timeout / gtimeout if present, else run unbounded.
run_capped() { # run_capped <logfile> <dir>
  local dir="$2"
  if command -v timeout >/dev/null 2>&1;  then ( cd "$dir" && timeout  "$MAXSEC" bash RUNME.sh );
  elif command -v gtimeout >/dev/null 2>&1; then ( cd "$dir" && gtimeout "$MAXSEC" bash RUNME.sh );
  else ( cd "$dir" && bash RUNME.sh ); fi
}

export PATH="$VENV/bin:$PATH"
cd "$REPO" || { log "no repo at $REPO"; exit 1; }

if ! git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  log "no $BRANCH on origin yet (cloud routine hasn't created it); exit"; exit 0
fi
# Dedicated clone: flush any results left unpushed by a failed prior run, then
# hard-sync to origin. Safe here precisely because nothing else uses this clone.
git push --quiet origin "$BRANCH" 2>/dev/null || true
git fetch --quiet origin "$BRANCH" || { log "fetch failed"; exit 1; }
git checkout --quiet -B "$BRANCH" "origin/$BRANCH" || { log "checkout failed"; exit 1; }
git reset --hard --quiet "origin/$BRANCH"
git clean -fdq

mkdir -p "$RESULTS"
ran=0
for runme in "$WATCH"/*/RUNME.sh; do
  [ -e "$runme" ] || continue
  dir="$(dirname "$runme")"; thread="$(basename "$dir")"
  [ -f "$RESULTS/$thread.done" ] && continue
  if grep -qiE 'GPU_?POD' "$runme" || [ -f "$dir/.gpupod" ]; then
    log "skip (GPU-POD): $thread"; continue
  fi
  # Experiments that load the private chamber (relay live/server.py) need
  # WIREHEAD_LIVE set to a wirehead-site checkout; skip until that's wired,
  # rather than fail. Marked by a .needs_chamber file or a painlab.chamber import.
  if [ -z "${WIREHEAD_LIVE:-}" ] && { [ -f "$dir/.needs_chamber" ] || grep -qiE 'painlab\.chamber|load_chamber|WIREHEAD_LIVE' "$runme" "$dir"/*.py 2>/dev/null; }; then
    log "skip (needs chamber; WIREHEAD_LIVE unset): $thread"; continue
  fi
  ts="$(date -u +%Y%m%dT%H%M%SZ)"; logf="$RESULTS/${thread}-${ts}.log"
  log "running $thread -> ${logf#$REPO/}"
  run_capped "$logf" "$dir" >"$logf" 2>&1
  code=$?
  { echo "---"; echo "host=$(hostname) exit_code=$code finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } >>"$logf"
  echo "$ts exit=$code host=$(hostname)" >"$RESULTS/$thread.done"
  git add -A "$dir" "$RESULTS/${thread}-${ts}.log" "$RESULTS/$thread.done" 2>/dev/null
  git commit --quiet \
    -m "interp executor: results for $thread (exit $code)" \
    -m "Ran on $(hostname) via scripts/interp_executor.sh." \
    -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || true
  ran=$((ran+1))
done

if [ "$ran" -gt 0 ]; then
  git push --quiet origin "$BRANCH" && log "pushed $ran result(s)" || log "push failed (will retry next run)"
else
  log "nothing to run"
fi
