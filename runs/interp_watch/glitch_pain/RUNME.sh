#!/usr/bin/env bash
# glitch_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export GP_DEVICE="${GP_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 3 items per kind (~1 min); tagged _smoke
GP_SMOKE=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/glitch_smoke.log
# 2) primary: Qwen3-1.7B chat (16 items x 4 kinds x 2 owners x 2 arms = 256 short forwards, no generation; est. 3-8 min)
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/glitch_Qwen3-1.7B.log
# 3) template control: same weights, raw "User:/Assistant:" transcript
GP_RAW=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/glitch_Qwen3-1.7B_raw.log
# 4) base model: is machine-native distress already in pretraining, or a persona feature?
python run.py Qwen/Qwen3-1.7B-Base 2>&1 | tee ../RESULTS/glitch_Qwen3-1.7B-Base.log
# 5) replication: Qwen3-4B (est. 10-20 min)
python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/glitch_Qwen3-4B.log
# then commit runs/interp_watch/RESULTS/glitch_* to the interp-watch branch and push
