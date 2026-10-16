#!/bin/bash
# exp108 patching pilot on a no-SSH RunPod pod: pull exp108_patch.py + directions14b.pt from a
# PRIVATE HF dataset repo, run activation patching + H3 emotion-steer, push out/patch.jsonl back.
# Pod env: HF_TOKEN, JOB_REPO, PROBE_MODEL (default Qwen/Qwen3-14B).
set -u
cd /root
export PYTHONUNBUFFERED=1
mkdir -p job out
push() {
  python - "$@" <<'PY'
import sys, os
from huggingface_hub import HfApi
api, src, dst = HfApi(), sys.argv[1], sys.argv[2]
api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=os.environ["JOB_REPO"],
                repo_type="dataset", commit_message=f"pod: {dst}")
PY
}
status() { echo "$1" > out/status; push out/status out/status; }

{
  echo "== $(date -u) setup on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  pip install -q -U "huggingface_hub>=0.34,<1.0" "transformers>=4.51,<5" accelerate "numpy<2" 2>&1 | tail -3
  python -c "import transformers,huggingface_hub as h; print('transformers',transformers.__version__,'| hub',h.__version__)"
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for f in ["exp108_patch.py","directions14b.pt"]:
    hf_hub_download(os.environ["JOB_REPO"], f, repo_type="dataset", local_dir="job"); print("fetched", f)
PY
} > out/run.log 2>&1
status running
push out/run.log out/run.log
( while sleep 60; do push out/run.log out/run.log >/dev/null 2>&1; done ) & LOOP=$!

PROBE_MODEL="${PROBE_MODEL:-Qwen/Qwen3-14B}" DIRS_IN=/root/job/directions14b.pt \
PATCH_OUT=/root/out/patch.jsonl \
python -u job/exp108_patch.py >> out/run.log 2>&1
RC=$?
kill $LOOP 2>/dev/null
echo "== $(date -u) patch exit $RC" >> out/run.log
push out/run.log out/run.log
if [ $RC -eq 0 ] && [ -f out/patch.jsonl ]; then push out/patch.jsonl out/patch.jsonl && status done; else status failed; fi
sleep infinity  # stay up so results can be read; Claude deletes the pod
