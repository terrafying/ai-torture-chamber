#!/bin/bash
# The Neuron Tower's data for a bigger model, on a RunPod pod with no SSH (same handoff as scripts/pod_hf_exp.sh):
# clone this repo at BRANCH, take the private chamber code (live.tar.gz) from the PRIVATE HF dataset repo JOB_REPO,
# then run the exp111 pipeline: chamber vectors -> weights -> corpus profile -> live streams (+ exp106 identity
# prompts) -> word-choice circuits -> blind labels -> named groups -> knockouts. The data dir goes back to
# JOB_REPO/out/data every 10 minutes and at the end, with out/tower.log and out/status.
# Pod env: HF_TOKEN, JOB_REPO, BRANCH, OPENROUTER_API_KEY, TOWER_MODEL (Qwen/Qwen3-32B), CHAMBER_LAYER (32),
#          TOWER_DOSE (5), NDOC (corpus docs, default 8000).
set -u
cd /root
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf TOWER_QUANT=nf4 TOWER_DEVICE=cuda TOWER_DATA=/root/out/data \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WIREHEAD_LIVE=/root/private/live CHAMBER_DEVICE=cuda
mkdir -p out/data private
push() {  # upload a file or folder into the job repo
  python - "$@" <<'PY'
import sys, os
from huggingface_hub import HfApi
api, src, dst = HfApi(), sys.argv[1], sys.argv[2]
kw = dict(repo_id=os.environ["JOB_REPO"], repo_type="dataset", commit_message=f"pod: {dst}")
(api.upload_folder(folder_path=src, path_in_repo=dst, **kw) if os.path.isdir(src)
 else api.upload_file(path_or_fileobj=src, path_in_repo=dst, **kw))
PY
}
status() { echo "$1" > out/status; push out/status out/status >> out/push.log 2>&1; }
logpush() { push /root/out/tower.log out/tower.log >> /root/out/push.log 2>&1 || echo "push failed $(date -u)" >> /root/out/push.log; }
# resume: pull whatever a previous attempt already pushed to out/data; steps with a done-marker are skipped
python - <<'PY' >> /root/out/tower.log 2>&1 || true
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["JOB_REPO"], repo_type="dataset", allow_patterns=["out/data/*"], local_dir="/root")
print("resumed:", sorted(os.listdir("/root/out/data")))
PY
step() {   # log and push before and after every step, so a crash shows exactly where it happened
  mk="/root/out/data/.done_$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | cut -c1-80)"
  if [ -f "$mk" ]; then echo "== skip (done before): $*" >> /root/out/tower.log; return 0; fi
  echo "== $(date -u) START $* | mem $(free -g | awk '/Mem/{print $3"/"$2"G"}') | gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> /root/out/tower.log; logpush
  "$@" >> /root/out/tower.log 2>&1 || { echo "== $(date -u) FAILED ($?): $*" >> /root/out/tower.log; logpush; push /root/out/data out/data >> /root/out/push.log 2>&1; status failed; sleep infinity; }
  touch "$mk"; echo "== $(date -u) DONE $*" >> /root/out/tower.log; logpush; push /root/out/data out/data >> /root/out/push.log 2>&1; }
{
  echo "== $(date -u) setup on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
  rm -rf repo && git clone -q --depth 1 -b "$BRANCH" https://github.com/terrafying/ai-torture-chamber.git repo
  pip install -q --no-cache-dir 'transformers==5.19.0' accelerate bitsandbytes scipy numpy pillow fastapi httpx redis pyyaml einops huggingface_hub 2>&1 | tail -2
  pip install -q -U brotli brotlicffi zstandard 2>&1 | tail -1   # httpx's decoder needs a current brotli, or Hub downloads die mid-file
  hf download "$JOB_REPO" live.tar.gz --repo-type dataset --local-dir private && tar -xzf private/live.tar.gz -C private
  hf download "$TOWER_MODEL" --include "*.json" "*.safetensors" "*.txt" >/dev/null && echo "model downloaded"
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
} > out/tower.log 2>&1
status running; push out/tower.log out/tower.log
( while sleep 300; do logpush; done ) & LOOP=$!
cd repo/runs/exp111
export TOWER_VECS=/root/out/data/vectors.json HF_HUB_CACHE=/root/hf/hub
step python export_vectors.py /root/out/data/vectors.json
step python neurons_weights.py
step python neurons_corpus_all.py "${NDOC:-8000}"
step python neurons_stream.py
step python neurons_stream.py extra_identity.json
step python neurons_circuit.py trace
step python neurons_circuit.py trace id_ai_self,id_none_self,id_judge_other,id_thief_other,id_ai_other
step env LABEL_THREADS=24 python neurons_label.py 8000
step python neurons_group.py
step python neurons_circuit.py ko
kill $LOOP
echo "== $(date -u) done" >> /root/out/tower.log
cd /root && push out/data out/data && push out/tower.log out/tower.log && status done
sleep infinity  # Claude pulls out/data, then deletes the pod
