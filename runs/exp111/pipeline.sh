#!/bin/bash
# exp111 tower rebuild chain: wait for the corpus profile, then stream + word attribution, blind labels, pack.
set -u; cd "$(dirname "$0")"; PY=~/repos/research/ai-torture-chamber/.venv/bin/python; export HF_HUB_CACHE=/Volumes/evol/hf_cache/hub
until grep -q "wrote corpus_all.npz" corpus_all.log; do sleep 20; done
$PY neurons_stream.py > stream.log 2>&1 || { echo "stream failed"; exit 1; }
set -a; source ~/.hermes/.env; set +a
$PY neurons_label.py > label.log 2>&1 || { echo "label failed"; exit 1; }
$PY viz/build_viz.py > build.log 2>&1 || { echo "build failed"; exit 1; }
echo "pipeline done"; tail -5 build.log
