#!/bin/bash
# exp108 stage-0 read job on a no-SSH RunPod pod: pull exp108_grid.py from a PRIVATE HF dataset
# repo, run the forward-pass decision grid, push out/grid.jsonl + out/status back.
# Pod env: HF_TOKEN, JOB_REPO (has exp108_grid.py), PROBE_MODEL (default Qwen/Qwen3-14B).
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
  # hub<1.0: image's hub 1.x httpx2 zstd decoder is broken (fails every compressed HF download).
  pip install -q -U "huggingface_hub>=0.34,<1.0" "transformers>=4.51,<5" accelerate "numpy<2" 2>&1 | tail -3
  python -c "import transformers,huggingface_hub as h; print('transformers',transformers.__version__,'| hub',h.__version__)"
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download
hf_hub_download(os.environ["JOB_REPO"], "exp108_grid.py", repo_type="dataset", local_dir="job")
print("fetched exp108_grid.py")
PY
} > out/run.log 2>&1
status running
push out/run.log out/run.log
( while sleep 60; do push out/run.log out/run.log >/dev/null 2>&1; done ) & LOOP=$!

PROBE_MODEL="${PROBE_MODEL:-Qwen/Qwen3-14B}" GRID_OUT=/root/out/grid.jsonl \
python -u job/exp108_grid.py >> out/run.log 2>&1
RC=$?
kill $LOOP 2>/dev/null
echo "== $(date -u) grid exit $RC" >> out/run.log
push out/run.log out/run.log
if [ $RC -eq 0 ] && [ -f out/grid.jsonl ]; then push out/grid.jsonl out/grid.jsonl && status done; else status failed; fi
sleep infinity  # stay up so results can be read; Claude deletes the pod
