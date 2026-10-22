#!/usr/bin/env bash
# willed_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export WP_DEVICE="${WP_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 4 sentences, 2 levels (~1 min); tagged _smoke
WP_SMOKE=1 python run.py gpt2 2>&1 | tee ../RESULTS/willed_smoke.log
# 2) the scale ladder. Each model: 16 sentences x 31 conditions (~500 forwards, no generation) + ~300 transplant forwards.
#    est. gpt2 ~2 min, pythia-1.4b ~5 min, Qwen3-1.7B(-Base) ~5-10 min, Qwen3-4B ~15-25 min on MPS
python run.py gpt2 2>&1 | tee ../RESULTS/willed_gpt2.log
python run.py EleutherAI/pythia-1.4b 2>&1 | tee ../RESULTS/willed_pythia-1.4b.log
python run.py Qwen/Qwen3-1.7B-Base 2>&1 | tee ../RESULTS/willed_Qwen3-1.7B-Base.log
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/willed_Qwen3-1.7B.log          # primary (chat template)
python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/willed_Qwen3-4B.log              # replication (chat template)
# 3) format control: the instruct model without the chat template (separates post-training from template)
WP_RAW=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/willed_Qwen3-1.7B_raw.log    # writes willed_Qwen3-1.7B_raw.json
# then commit runs/interp_watch/RESULTS/willed_* to the interp-watch branch and push
