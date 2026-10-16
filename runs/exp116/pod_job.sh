#!/bin/bash
# exp116 on a RunPod GPU pod with no SSH: results and logs go to the PRIVATE HF dataset repo JOB_REPO (out/).
# Pod env: HF_TOKEN, JOB_REPO, BRANCH (claude/exp116).
set -u; cd /root; mkdir -p out; export PYTHONUNBUFFERED=1 HF_HOME=/root/hf OUT=/root/out
push() { python - "$1" "$2" <<'PY'
import os, sys
from huggingface_hub import HfApi
src, dst = sys.argv[1], sys.argv[2]; api = HfApi(); kw = dict(repo_id=os.environ["JOB_REPO"], repo_type="dataset", commit_message=f"pod: {dst}")
(api.upload_folder(folder_path=src, path_in_repo=dst, **kw) if os.path.isdir(src) else api.upload_file(path_or_fileobj=src, path_in_repo=dst, **kw))
PY
}
status() { echo "$1" > out/status; push out/status out/status >/dev/null 2>&1; }
{
  echo "== $(date -u) setup on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
  command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
  pip install -q -U huggingface_hub brotli brotlicffi zstandard 2>&1 | tail -1
  pip install -q "git+https://github.com/safety-research/circuit-tracer" 2>&1 | tail -2
  rm -rf repo && git clone -q --depth 1 -b "$BRANCH" https://github.com/terrafying/ai-torture-chamber.git repo
  python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B', allow_patterns=['*.json','*.safetensors','*.txt']); print('transcoders at', s('mwhanna/qwen3-4b-transcoders', allow_patterns=['config.yaml','layer_*.safetensors','*.json']))"
  python -c "import circuit_tracer, torch; print('circuit_tracer ok', torch.__version__)"
} > out/trace.log 2>&1
status running; push out/trace.log out/trace.log >/dev/null 2>&1
( while sleep 240; do push out/trace.log out/trace.log >/dev/null 2>&1; done ) & LOOP=$!
python repo/runs/exp116/trace.py >> out/trace.log 2>&1 && st=done || st=failed
kill $LOOP; push out out >/dev/null 2>&1; status $st; sleep infinity
