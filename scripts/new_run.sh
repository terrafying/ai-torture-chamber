#!/bin/bash
# new_run.sh <slug> — create a new run directory under runs/ with the descriptive naming
# convention (<slug>-<word>) and register it. Refuses a name that already exists on disk
# or in runs/REGISTRY.md. The word comes from a fixed list so names are greppable and
# pronounceable across agents.
set -euo pipefail
cd "$(dirname "$0")/.."

SLUG="${1:?usage: scripts/new_run.sh <slug>}"
[ -d runs ] || { echo "no runs/ dir" >&2; exit 1; }

WORDS=(falcon heron wren kestrel harrier osprey swift swallow martin shrike
       tanager bittern curlew godwit siskin sora crossbill petrel
       tern plover stilt teal eider merganser)
WORD="${WORDS[$((RANDOM % ${#WORDS[@]}))]}"
NAME="${SLUG}-${WORD}"

if [ -e "runs/$NAME" ] || grep -q -- "- $NAME" runs/REGISTRY.md 2>/dev/null; then
  echo "clash: $NAME already exists — rerun for a new word" >&2; exit 1
fi

mkdir -p "runs/$NAME"
cat > "runs/$NAME/hypotheses.json" <<EOF
{
 "run": "$NAME",
 "written": "$(date +%Y-%m-%d), before any run",
 "title": "",
 "why": "",
 "design": {},
 "hypotheses": {},
 "decision_rule": ""
}
EOF
printf -- "- %s — (fill in one line when pre-registering)\n" "$NAME" >> runs/REGISTRY.md
echo "created runs/$NAME and registered it"
