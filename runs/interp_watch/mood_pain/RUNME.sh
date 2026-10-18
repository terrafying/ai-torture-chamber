#!/usr/bin/env bash
# mood_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export MD_DEVICE="${MD_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 3 primes x 1 filler (~3-8 min); tagged _smoke
MD_SMOKE=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/mood_smoke.log
# 2) primary: Qwen3-1.7B instruct (raw text, no chat template; ~1.2k batched forwards, no generation; est. 30-60 min on MPS)
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/mood_Qwen3-1.7B.log
# 3) base model at the same size (H6: does post-training change the affect afterglow?)
python run.py Qwen/Qwen3-1.7B-Base 2>&1 | tee ../RESULTS/mood_Qwen3-1.7B-Base.log
# 4) replication: Qwen3-4B (est. 1-2 h)
python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/mood_Qwen3-4B.log
# then commit runs/interp_watch/RESULTS/mood_* to the interp-watch branch and push
