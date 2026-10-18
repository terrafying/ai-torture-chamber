#!/bin/bash
# stage 1 -> stage 2 (smoke, full) -> stage 3 (smoke, full). Stage 3 is interpreted only if stage 2's H2 passes.
set -e
cd "$(dirname "$0")/../.."
D=runs/deference-heads-falcon; PY=.venv/bin/python3
[ -z "$SKIP1" ] && STAGE=1 $PY $D/run.py > $D/chain.log 2>&1
STAGE=2 DH_SMOKE=1 $PY $D/run.py >> $D/chain.log 2>&1; rm -f $D/smoke_*
STAGE=2 $PY $D/run.py >> $D/chain.log 2>&1
STAGE=3 DH_SMOKE=1 $PY $D/run.py >> $D/chain.log 2>&1; rm -f $D/smoke_*
STAGE=3 $PY $D/run.py >> $D/chain.log 2>&1
echo "ALL DONE $(date +%T)" >> $D/chain.log
