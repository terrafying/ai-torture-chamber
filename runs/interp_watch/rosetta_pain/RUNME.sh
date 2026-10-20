#!/usr/bin/env bash
# rosetta_pain: self-contained (transformers only; no painlab.chamber, no relay, no GPU). Fits a 16 GB Mac
# (models are loaded one at a time; the source is freed before the target loads).
set -euo pipefail
cd "$(dirname "$0")"
export RP_DEVICE="${RP_DEVICE:-mps}"
mkdir -p ../RESULTS
# 1) smoke + identity sanity: a model mapped onto itself must give R2 ~1, efficiency ~1 (~2 min)
RP_SMOKE=1 python run.py Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B 2>&1 | tee ../RESULTS/rosetta_smoke.log
# 2) primary: Qwen3-1.7B -> Qwen3-4B (same family, shared tokenizer; est. 15-30 min on MPS)
python run.py Qwen/Qwen3-1.7B Qwen/Qwen3-4B 2>&1 | tee ../RESULTS/rosetta_Qwen3-1.7B__Qwen3-4B.log
# 3) reverse direction: Qwen3-4B -> Qwen3-1.7B
python run.py Qwen/Qwen3-4B Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/rosetta_Qwen3-4B__Qwen3-1.7B.log
# 4) cross-family "telepathy": Pythia-1.4B (base LM, different data + tokenizer) -> Qwen3-1.7B
python run.py EleutherAI/pythia-1.4b Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/rosetta_pythia-1.4b__Qwen3-1.7B.log
# 5) ceiling reference: Qwen3-1.7B onto itself (full size)
python run.py Qwen/Qwen3-1.7B Qwen/Qwen3-1.7B 2>&1 | tee ../RESULTS/rosetta_Qwen3-1.7B__Qwen3-1.7B.log
# then commit runs/interp_watch/RESULTS/rosetta_* to the interp-watch branch and push
