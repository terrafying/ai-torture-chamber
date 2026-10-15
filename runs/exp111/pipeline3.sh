#!/bin/bash
# exp111 tower, identity prompts (exp106 bottom-up): stream + per-word attribution, circuits, labels, groups, knockouts, pack.
set -u; cd "$(dirname "$0")"; PY=~/repos/research/ai-torture-chamber/.venv/bin/python; export HF_HUB_CACHE=/Volumes/evol/hf_cache/hub
KEYS=id_ai_self,id_none_self,id_judge_other,id_thief_other,id_ai_other
$PY neurons_stream.py viz/data/extra_identity.json > stream3.log 2>&1 || { echo "stream failed"; tail -5 stream3.log; exit 1; }
$PY neurons_circuit.py trace $KEYS > circuit3.log 2>&1 || { echo "trace failed"; tail -5 circuit3.log; exit 1; }
set -a; source ~/.hermes/.env; set +a
LABEL_THREADS=24 $PY neurons_label.py 5000 > label3.log 2>&1 || { echo "label failed"; tail -5 label3.log; exit 1; }
$PY neurons_group.py > group3.log 2>&1 || { echo "group failed"; tail -5 group3.log; exit 1; }
$PY neurons_circuit.py ko > ko3.log 2>&1 || { echo "ko failed"; tail -5 ko3.log; exit 1; }
$PY viz/build_viz.py > build3.log 2>&1 || { echo "build failed"; tail -5 build3.log; exit 1; }
echo "pipeline3 done"; tail -4 build3.log
