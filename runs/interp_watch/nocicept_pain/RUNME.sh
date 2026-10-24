#!/usr/bin/env bash
# nocicept_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export NP_DEVICE="${NP_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 4 items, 2 owners (~1-2 min); tagged _smoke
NP_SMOKE=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/nocicept_smoke.log
# 2) primary: Qwen3-1.7B (16 items x 6 owners x 4 cells; ~1,200 short forwards, no generation; est. 5-15 min on MPS)
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/nocicept_Qwen3-1.7B.log
# 3) replication: Qwen3-4B (est. 15-30 min)
python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/nocicept_Qwen3-4B.log
# 4) base model (exploratory: does post-training change whose body the model computes pain for?)
python run.py Qwen/Qwen3-1.7B-Base 2>&1 | tee ../RESULTS/nocicept_Qwen3-1.7B-Base.log
# then commit runs/interp_watch/RESULTS/nocicept_* to the interp-watch branch and push
