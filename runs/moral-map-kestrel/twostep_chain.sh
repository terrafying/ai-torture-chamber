#!/bin/bash
cd "$(dirname "$0")/../.."
for spec in "unsloth/Llama-3.2-3B-Instruct:" "Qwen/Qwen3-4B-Base:1"; do
  M=${spec%%:*}; B=${spec##*:}
  KESTREL_MODEL=$M KESTREL_BASE=$B .venv/bin/python3 runs/moral-map-kestrel/twostep.py 2>&1 | grep -v -i warn | tail -2
  rm -rf ~/.cache/huggingface/hub/models--$(echo "$M" | sed 's#/#--#g') && echo "deleted $M"
done
