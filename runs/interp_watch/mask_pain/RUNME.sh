#!/usr/bin/env bash
# mask_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export MP_DEVICE="${MP_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 3 scene pairs, one ablation layer (~1-2 min); tagged _smoke
MP_SMOKE=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/mask_smoke.log
# 2) primary: Qwen3-1.7B (64 prompts x 12 teacher-forced continuations, x 2 formats + 4 ablation runs; no generation; est. 5-15 min on MPS)
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/mask_Qwen3-1.7B.log
# 3) replication: Qwen3-4B (est. 15-30 min)
python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/mask_Qwen3-4B.log
# then commit runs/interp_watch/RESULTS/mask_* to the interp-watch branch and push
