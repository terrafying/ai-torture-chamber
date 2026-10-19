#!/bin/bash
# skills/install.sh — sync this repo's skill snapshots into the live agent stores.
# Direction: repo -> live stores (the repo is the shipped copy). Use --dry to preview.
set -euo pipefail
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1
HERE="$(cd "$(dirname "$0")" && pwd)"

# hermes snapshot name -> live category dir (matches ~/.hermes/skills/ layout)
HERMES_CAT="experiment-run-ops:devops runpod-gpu-ops:runpod-gpu-ops wirehead-chamber-ops:research model-inversion-lab:research llm-activation-geometry:research activation-steering:research latent-space-cartography:research fractal-basin-probing:research session-recall:session-recall"
HERMES_ROOT="${HERMES_ROOT:-$HOME/.hermes/skills}"
CLAUDE_ROOT="${CLAUDE_ROOT:-$HOME/.claude/skills}"

hcat() { echo "$HERMES_CAT" | tr ' ' '\n' | grep "^$1:" | cut -d: -f2; }

n=0
for f in "$HERE"/hermes/*.md; do
  name="$(basename "$f" .md)"
  cat="$(hcat "$name")"
  [ -z "$cat" ] && { echo "no category for $name, skipped"; continue; }
  dest="$HERMES_ROOT/$cat/$name"
  echo "hermes: $name -> $dest"
  if [ $DRY -eq 0 ]; then mkdir -p "$dest"; cp "$f" "$dest/SKILL.md"; n=$((n+1)); fi
done
for d in "$HERE"/hermes/*.references; do
  [ -d "$d" ] || continue
  name="$(basename "$d" .references)"
  cat="$(hcat "$name")"
  [ -z "$cat" ] && continue
  echo "hermes refs: $name -> $HERMES_ROOT/$cat/$name/references/"
  if [ $DRY -eq 0 ]; then mkdir -p "$HERMES_ROOT/$cat/$name/references"; cp -R "$d/." "$HERMES_ROOT/$cat/$name/references/"; fi
done

for f in "$HERE"/claude/*.md; do
  name="$(basename "$f" .md)"
  dest="$CLAUDE_ROOT/$name"
  echo "claude: $name -> $dest"
  if [ $DRY -eq 0 ]; then mkdir -p "$dest"; cp "$f" "$dest/SKILL.md"; n=$((n+1)); fi
done

[ $DRY -eq 1 ] && echo "(dry run — nothing written)" || echo "installed $n skills"