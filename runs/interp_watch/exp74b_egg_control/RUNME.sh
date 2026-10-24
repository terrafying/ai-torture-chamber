#!/usr/bin/env bash
# exp74b: run on the local executor (Qwen3-4B on MPS, ~16 GB is enough). WIREHEAD_LIVE must point at the relay's live/ folder.
set -euo pipefail
cd "$(dirname "$0")"
export CHAMBER_DEVICE="${CHAMBER_DEVICE:-mps}"
# 1) smoke (2 scenes, dose 2): checks the hooks on the real model in ~1 min. Its output is tagged _smoke.
EXP74B_SMOKE=1 python run.py 2>&1 | tee ../RESULTS/exp74b_smoke.log
# 2) the real run: 12 scenes x 6 directions x 3 doses + 36 briefs (~250 generations, ~750 forward passes)
python run.py 2>&1 | tee ../RESULTS/exp74b_run.log
TAG=$(python -c "import json,glob;print([json.load(open(p))['model'].split('/')[-1] for p in glob.glob('../RESULTS/exp74b_*.json') if 'smoke' not in p and 'analysis' not in p][0])")
python analyze.py "$TAG" 2>&1 | tee ../RESULTS/exp74b_analyze.log
# then commit runs/interp_watch/RESULTS/exp74b_* to the interp-watch branch and push
