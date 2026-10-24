#!/usr/bin/env bash
# whitebear_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export WB_DEVICE="${WB_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 1 task x 4 concepts x 4 conditions, short generations (~3-5 min); tagged _smoke
WB_SMOKE=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/whitebear_smoke.log
# 2) primary: Qwen3-1.7B (~1700 generations of <=160 tokens; est. 20-40 min on MPS)
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/whitebear_Qwen3-1.7B.log
# 3) replication: Qwen3-4B (est. 45-90 min); lower the batch if memory is tight: WB_BATCH=8
WB_BATCH="${WB_BATCH:-8}" python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/whitebear_Qwen3-4B.log
# then commit runs/interp_watch/RESULTS/whitebear_* to the interp-watch branch and push
