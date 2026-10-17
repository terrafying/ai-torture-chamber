#!/bin/bash
# Run a runs/expNN experiment on a RunPod pod with no SSH and no tunnel: the pod clones this (public) repo at a
# branch, takes the private chamber code (the relay's live/, as live.tar.gz) from a PRIVATE Hugging Face dataset
# repo, runs run.py once per model, then analyze.py, and pushes results, logs and status back to that repo
# (out/, every 2 min). The same handoff as scripts/pod_hf_job.sh; painlab's own `exp pod` needs a tunnel instead.
#
# Pod env: HF_TOKEN, JOB_REPO (private dataset repo with live.tar.gz), BRANCH, EXP (e.g. exp106),
#          MODELS ("Qwen/Qwen3-8B:18,Qwen/Qwen3-14B:23": model id and chamber layer).
# Pod start (REST v1, dockerEntrypoint ["/bin/bash","-c"]): fetch this script from JOB_REPO with HF_TOKEN, then bash it.
set -u
cd /root
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf CHAMBER_DEVICE=cuda CHAMBER_DTYPE=bfloat16 WIREHEAD_LIVE=/root/private/live
mkdir -p out private
push() {  # upload a file or folder into out/ of the job repo
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
  command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
  git clone -q --depth 1 -b "$BRANCH" https://github.com/terrafying/ai-torture-chamber.git repo
  pip install -q --no-cache-dir 'transformers==5.17.0' peft accelerate numpy scipy scikit-learn fastapi httpx redis pyyaml einops huggingface_hub 2>&1 | tail -2
  pip install -q --no-cache-dir -e repo 2>&1 | tail -1
  hf download "$JOB_REPO" live.tar.gz --repo-type dataset --local-dir private && tar -xzf private/live.tar.gz -C private
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
} > out/run.log 2>&1
status running; push out/run.log out/run.log
D=repo/runs/$EXP
# resume: results a previous (lost) pod already pushed to out/ go back in place; run.py skips the rows it finds
python - "$D" <<'PY' >> out/run.log 2>&1
import os, sys, shutil
from huggingface_hub import HfApi, hf_hub_download
r = os.environ["JOB_REPO"]
for f in HfApi().list_repo_files(r, repo_type="dataset"):
    if f.startswith("out/results_") and f.endswith(".json"):
        shutil.copy(hf_hub_download(r, f, repo_type="dataset"), os.path.join(sys.argv[1], os.path.basename(f))); print("resume from", f)
PY
( while sleep 120; do cp $D/results_*.json out/ 2>/dev/null; push out out >/dev/null 2>&1; done ) & LOOP=$!
RC=0
for spec in ${MODELS//,/ }; do
  m=${spec%%:*}; layer=${spec##*:}
  echo "== $(date -u) $m layer $layer" >> out/run.log
  (cd $D && CHAMBER_MODEL=$m CHAMBER_LAYER=$layer python -u run.py) >> out/run.log 2>&1 || RC=1
done
(cd $D && python analyze.py) >> out/run.log 2>&1 || RC=1
kill $LOOP
cp $D/results_*.json $D/analysis.json out/ 2>/dev/null
echo "== $(date -u) exit $RC" >> out/run.log
push out out
[ $RC -eq 0 ] && status done || status failed
sleep infinity  # Claude reads the results, then deletes the pod
