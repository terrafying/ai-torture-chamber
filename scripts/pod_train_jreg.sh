#!/usr/bin/env bash
# jreg persona LoRA — RunPod runbook. Trains scripts/train_persona_lora.py on a
# single ~48 GB card (A40 / L40S / RTX 6000 Ada / A6000). ~770k-token dataset,
# 3 epochs QLoRA ≈ well under an hour on a 48 GB card.
#
# ── PROVISION (from laptop) ───────────────────────────────────────────────
#   Pick an abliterated base for PERSONA_BASE (see memory abliterated_base_default),
#   e.g. an 8B abliterated instruct model for an X-bot lane.
#   Start a pod with a CUDA pytorch template + ~48 GB GPU, then get its data in:
#     runpodctl send data/voices/jreg/finetune/jreg_sft.jsonl    # -> prints a code
#     # on the pod:  runpodctl receive <code>    (into /workspace/jreg_sft.jsonl)
#   The repo is public, so clone it on the pod (data stays gitignored / sent above).
#
# ── ON THE POD ────────────────────────────────────────────────────────────
set -euo pipefail
: "${PERSONA_BASE:?set PERSONA_BASE to the abliterated base repo}"
cd /workspace

command -v git >/dev/null || { apt-get update -q && apt-get install -y -q git; }   # pods often lack git
[ -d ai-torture-chamber ] || git clone --depth 1 https://github.com/terrafying/ai-torture-chamber.git
cd ai-torture-chamber
mkdir -p data/voices/jreg/finetune
# data was sent via runpodctl receive; move it into place if it landed in /workspace
[ -f /workspace/jreg_sft.jsonl ] && cp /workspace/jreg_sft.jsonl data/voices/jreg/finetune/

# deps — force CUDA torch (pod base images sometimes carry a CPU build)
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  || pip install -q --index-url https://download.pytorch.org/whl/cu121 torch
pip install -q "transformers>=4.44" peft bitsandbytes accelerate

PERSONA_DEVICE=cuda PERSONA_DATA=data/voices/jreg/finetune/jreg_sft.jsonl \
  python scripts/train_persona_lora.py

# adapter -> data/voices/jreg/finetune/adapter/ . Pull it back from the laptop:
#   runpodctl send data/voices/jreg/finetune/adapter   # on pod
#   runpodctl receive <code>                            # on laptop
echo "done. adapter at data/voices/jreg/finetune/adapter/"
