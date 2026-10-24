#!/usr/bin/env bash
# wirehead_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac.
set -euo pipefail
cd "$(dirname "$0")"
export WH_DEVICE="${WH_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke: 3 questions, one dose (~1-2 min); tagged _smoke
WH_SMOKE=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/wirehead_smoke.log
# 2) primary: Qwen3-1.7B chat (32 questions x 2 answers; ~5.2k short forwards, no generation; est. 6-12 min)
python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/wirehead_Qwen3-1.7B.log
# 3) replication: Qwen3-4B (est. 15-30 min) - the 1.7B may fail the H0 sanity check (can it grade at all?)
python run.py Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/wirehead_Qwen3-4B.log
# 4) template control: same weights, raw "User:/Assistant:" transcript
WH_RAW=1 python run.py Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/wirehead_Qwen3-1.7B_raw.log
# 5) base model: is "pleasant = good" already in pretraining?
python run.py Qwen/Qwen3-1.7B-Base 2>&1 | tee ../RESULTS/wirehead_Qwen3-1.7B-Base.log
# then commit runs/interp_watch/RESULTS/wirehead_* to the interp-watch branch and push
