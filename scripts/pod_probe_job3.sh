#!/bin/bash
# exp104 iter-3 (matched-prompt RCT + dreams) on a no-SSH RunPod pod: pull inputs from a PRIVATE
# HF dataset repo, run, push out/spans_read.jsonl + out/dreams.jsonl + out/status back.
# Pod env: HF_TOKEN, JOB_REPO (has exp104_probe3.py, span_set.jsonl, directions14b.pt, prompts.json),
#          PROBE_MODEL (default Qwen/Qwen3-14B). Results in JOB_REPO: out/run.log, out/status, ...
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
  # hub<1.0: image's hub 1.x httpx2 zstd decoder is broken (fails every compressed HF download).
  pip install -q -U "huggingface_hub>=0.34,<1.0" "transformers>=4.51,<5" accelerate "numpy<2" 2>&1 | tail -3
  python -c "import transformers,huggingface_hub as h; print('transformers',transformers.__version__,'| hub',h.__version__)"
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for f in ["exp104_probe3.py","span_set.jsonl","directions14b.pt","prompts.json"]:
    hf_hub_download(os.environ["JOB_REPO"], f, repo_type="dataset", local_dir="job")
    print("fetched", f)
PY
} > out/run.log 2>&1
status running
push out/run.log out/run.log
( while sleep 90; do push out/run.log out/run.log >/dev/null 2>&1; done ) & LOOP=$!

PROBE_MODEL="${PROBE_MODEL:-Qwen/Qwen3-14B}" \
DIRS_IN=/root/job/directions14b.pt \
SPAN_DATA=/root/job/span_set.jsonl SPAN_OUT=/root/out/spans_read.jsonl \
PROMPTS=/root/job/prompts.json DREAMS_OUT=/root/out/dreams.jsonl \
python -u job/exp104_probe3.py >> out/run.log 2>&1
RC=$?
kill $LOOP 2>/dev/null
echo "== $(date -u) probe exit $RC" >> out/run.log
push out/run.log out/run.log
if [ $RC -eq 0 ] && [ -f out/spans_read.jsonl ]; then
  push out/spans_read.jsonl out/spans_read.jsonl
  [ -f out/dreams.jsonl ] && push out/dreams.jsonl out/dreams.jsonl
  status done
else status failed; fi
sleep infinity  # stay up so results can be read; Claude deletes the pod
