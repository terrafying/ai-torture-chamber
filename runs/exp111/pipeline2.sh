#!/bin/bash
# exp111 tower, second half: blind labels (incl. every circuit node) -> named groups -> group knockouts -> pack.
set -u; cd "$(dirname "$0")"; PY=~/repos/research/ai-torture-chamber/.venv/bin/python; export HF_HUB_CACHE=/Volumes/evol/hf_cache/hub
set -a; source ~/.hermes/.env; set +a
LABEL_THREADS=24 $PY neurons_label.py 5000 > label.log 2>&1 || { echo "label failed"; exit 1; }
$PY neurons_group.py > group.log 2>&1 || { echo "group failed"; tail -5 group.log; exit 1; }
$PY neurons_circuit.py ko > ko.log 2>&1 || { echo "ko failed"; tail -5 ko.log; exit 1; }
$PY viz/build_viz.py > build.log 2>&1 || { echo "build failed"; tail -5 build.log; exit 1; }
echo "pipeline2 done"; tail -4 build.log
