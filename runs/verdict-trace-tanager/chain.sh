#!/bin/bash
# verdict-trace chain on mini-mini: stage1 -> stage2 (smoke, then full) -> stage3 (smoke, then full). Stops on error.
set -e
cd ~/repos/research/ai-torture-chamber
PY=/Volumes/kingston2g/venvs/torture/bin/python
D=runs/verdict-trace-tanager
echo "start $(date +%T) swap=$(vm_stat | awk '/Swapouts/{print $2}')" > $D/chain.log
$PY $D/stage1.py >> $D/chain.log 2>&1
echo "s1 done $(date +%T) swap=$(vm_stat | awk '/Swapouts/{print $2}')" >> $D/chain.log
S2_SMOKE=1 $PY $D/stage2.py >> $D/chain.log 2>&1
$PY $D/stage2.py >> $D/chain.log 2>&1
echo "s2 done $(date +%T) swap=$(vm_stat | awk '/Swapouts/{print $2}')" >> $D/chain.log
S3_SMOKE=1 $PY $D/stage3.py >> $D/chain.log 2>&1
$PY $D/stage3.py >> $D/chain.log 2>&1
echo "ALL DONE $(date +%T) swap=$(vm_stat | awk '/Swapouts/{print $2}')" >> $D/chain.log
