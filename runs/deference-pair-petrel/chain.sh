#!/bin/bash
# stage A (scores, 40 null pairs) -> stage B (free text, runs regardless)
set -e
cd "$(dirname "$0")/../.."
D=runs/deference-pair-petrel; PY=.venv/bin/python3
STAGE=A $PY $D/run.py > $D/chain.log 2>&1
STAGE=B $PY $D/run.py >> $D/chain.log 2>&1
echo "ALL DONE $(date +%T)" >> $D/chain.log
