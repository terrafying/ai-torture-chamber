#!/bin/bash
# exp104 probe job on a RunPod pod with no SSH: pod pulls inputs from a PRIVATE HF dataset
# repo, runs the forward-pass probe, and pushes out/results.jsonl + out/status back.
# Pod env: HF_TOKEN, JOB_REPO (dataset repo with exp104_probe.py + probe_set.jsonl + pain_dirs_14b.pt),
#          PROBE_MODEL (default Qwen/Qwen3-14B). Results in JOB_REPO: out/run.log, out/status, out/results.jsonl
set -u
cd /root
export PYTHONUNBUFFERED=1
mkdir -p job out
push() {
  python - "$@" <<'PY'
import sys, os
from huggingface_hub import HfApi
api, src, dst = HfApi(), sys.argv[1], sys.argv[2]
kw = dict(repo_id=os.environ["JOB_REPO"], repo_type="dataset", commit_message=f"pod: {dst}")
(api.upload_folder(folder_path=src, path_in_repo=dst, **kw) if os.path.isdir(src)
 else api.upload_file(path_or_fileobj=src, path_in_repo=dst, **kw))
PY
}
status() { echo "$1" > out/status; push out/status out/status; }

{
  echo "== $(date -u) setup on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  # Pin huggingface_hub<1.0: the image's hub 1.x uses httpx2 whose zstd decoder is broken
  # (TypeError: process() takes no keyword arguments) and fails every compressed HF download.
  # hub<1.0 uses requests. transformers>=4.51 is needed for Qwen3.
  pip install -q -U "huggingface_hub>=0.34,<1.0" "transformers>=4.51,<5" accelerate "numpy<2" 2>&1 | tail -3
  python -c "import transformers,huggingface_hub as h; print('transformers',transformers.__version__,'| hub',h.__version__)"
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for f in ["exp104_probe.py","probe_set.jsonl","pain_dirs_14b.pt"]:
    p=hf_hub_download(os.environ["JOB_REPO"], f, repo_type="dataset", local_dir="job")
    print("fetched", f)
PY
} > out/run.log 2>&1
status running
push out/run.log out/run.log
( while sleep 90; do push out/run.log out/run.log >/dev/null 2>&1; done ) & LOOP=$!

PROBE_MODEL="${PROBE_MODEL:-Qwen/Qwen3-14B}" \
PROBE_DIRS=/root/job/pain_dirs_14b.pt \
PROBE_DATA=/root/job/probe_set.jsonl \
PROBE_OUT=/root/out/results.jsonl \
python -u job/exp104_probe.py >> out/run.log 2>&1
RC=$?
kill $LOOP 2>/dev/null
echo "== $(date -u) probe exit $RC" >> out/run.log
push out/run.log out/run.log
if [ $RC -eq 0 ] && [ -f out/results.jsonl ]; then push out/results.jsonl out/results.jsonl && status done; else status failed; fi
sleep infinity  # stay up so results can be read; Claude deletes the pod
