#!/usr/bin/env bash
# dark_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export DP_DEVICE="${DP_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 2 contexts, 1 dose (~1-2 min); tagged _smoke
DP_SMOKE=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/dark_smoke.log
# 2) primary: Qwen3-1.7B (~350 short forwards, no generation; est. 10-20 min on MPS)
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/dark_Qwen3-1.7B.log
# 3) replication: Qwen3-4B (est. 20-40 min)
python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/dark_Qwen3-4B.log
# 4) base model (H5, exploratory: does post-training change where pain sits?)
python run.py Qwen/Qwen3-1.7B-Base 2>&1 | tee ../RESULTS/dark_Qwen3-1.7B-Base.log
# then commit runs/interp_watch/RESULTS/dark_* to the interp-watch branch and push
