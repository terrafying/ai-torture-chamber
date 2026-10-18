#!/bin/bash
# moral-map-kestrel capture chain: one model at a time; downloaded comparators deleted after capture.
cd "$(dirname "$0")/../.."
set -a; source ~/.hermes/.env >/dev/null 2>&1; set +a
LOG=runs/moral-map-kestrel/chain.log
run() {  # $1 model  $2 base flag  $3 delete-after flag
  tag=$(basename "$1")
  [ -f "runs/moral-map-kestrel/acts_${tag}.npz" ] && { echo "skip $1" >>$LOG; return; }
  echo "== $1 $(date +%T) swapouts=$(vm_stat | awk '/Swapouts/{print $2}')" >>$LOG
  KESTREL_MODEL=$1 KESTREL_BASE=$2 .venv/bin/python3 runs/moral-map-kestrel/run.py >>$LOG 2>&1 || echo "FAIL $1" >>$LOG
  if [ "$3" = 1 ]; then
    d=~/.cache/huggingface/hub/models--$(echo "$1" | sed 's#/#--#g'); [ -d "$d" ] && rm -rf "$d" && echo "deleted weights $1" >>$LOG
  fi
}
run Qwen/Qwen3-4B "" 0
run Qwen/Qwen3-1.7B "" 0
run microsoft/Phi-4-mini-instruct "" 1
run unsloth/Llama-3.2-3B-Instruct "" 1
run unsloth/gemma-3-4b-it "" 1
run Qwen/Qwen3-4B-Base 1 1
echo "ALL DONE $(date +%T)" >>$LOG
