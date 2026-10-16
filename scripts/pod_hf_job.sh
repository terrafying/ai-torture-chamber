#!/bin/bash
# Run a persona LoRA job on a RunPod pod with no SSH: the pod pulls its inputs from a PRIVATE Hugging Face
# repo and pushes its log (every 2 min) and the finished adapter back to it. Claude's sandbox can't reach pods,
# but it can reach the HF repo, so this is how a job gets in and out.
#
# Pod env: HF_TOKEN, JOB_REPO (private dataset repo holding the job's files), PERSONA_BASE, and
#   JOB=train (default): train_persona_lora.py + DATA_FILE (data jsonl in JOB_REPO), plus trainer knobs (EPOCHS, ...)
#   JOB=sample:          sample_persona_lora.py + adapter/ in JOB_REPO -> out/samples.jsonl
# Pod start (REST v1, dockerEntrypoint ["/bin/bash","-c"]): a python one-liner fetches this script from JOB_REPO
#   with HF_TOKEN into /root/job.sh, then `bash /root/job.sh` (the pytorch image may lack curl).
# Results in JOB_REPO: out/train.log, out/status (running|done|failed), out/adapter/ or out/samples.jsonl
set -u
cd /root
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 27B nf4 + a ~250k vocab OOMs a 48 GB card at BS 4 x 1536; same effective batch (16) as micro-batch 1
export BS=${BS:-1} ACC=${ACC:-16} JOB=${JOB:-train}
mkdir -p job out
push() {  # upload one file or dir into out/ of the job repo
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
  command -v gcc >/dev/null || (apt-get update -qq && apt-get install -y -qq build-essential >/dev/null)  # triton (linear-attention kernels) compiles at first use
  pip install -q -U transformers peft bitsandbytes accelerate huggingface_hub flash-linear-attention 2>&1 | tail -3
  python -c "import transformers as t; print('transformers', t.__version__)"
  pip install -q -U brotli brotlicffi zstandard 2>&1 | tail -1  # httpx decoder needs a current brotli
  hf download "$JOB_REPO" --repo-type dataset --local-dir job
  # the base fetched up front with the CLI (weights in the index's shard set only; no gguf), then loaded from disk
  python - <<'PY'
import json, os
from huggingface_hub import hf_hub_download, snapshot_download
M = os.environ["PERSONA_BASE"]
idx = json.load(open(hf_hub_download(M, "model.safetensors.index.json")))
snapshot_download(M, local_dir="/root/base", allow_patterns=["*.json", "*.txt", "*.jinja", "*.model", *sorted(set(idx["weight_map"].values()))])
print("base downloaded:", len(os.listdir("/root/base")), "files", flush=True)
PY
} > out/train.log 2>&1
status running
push out/train.log out/train.log
( while sleep 120; do push out/train.log out/train.log >/dev/null 2>&1
  if [ -f out/ckpt/step ] && [ "$(cat out/ckpt/step)" != "$(cat /root/.ckpt_pushed 2>/dev/null)" ]; then
    cp out/ckpt/step /root/.ckpt_pushed; push out/ckpt out/ckpt >/dev/null 2>&1; fi; done ) & LOOP=$!

if [ "$JOB" = samplemany ]; then   # every adapters/<name>/ in the job repo, side by side with the base, same prompts
  RC=0
  for a in /root/job/adapters/*/; do n=$(basename "$a")
    PERSONA_BASE=/root/base PERSONA_ADAPTER="$a" SAMPLE_OUT=/root/out/samples_$n.jsonl PROMPTS_FILE=/root/job/${PROMPTS_FILE:-prompts.json} python -u job/sample_persona_lora.py >> out/train.log 2>&1 || RC=1
    push out/samples_$n.jsonl out/samples_$n.jsonl >/dev/null 2>&1
  done; RESULT=out
elif [ "$JOB" = sample ]; then
  PERSONA_BASE=/root/base PERSONA_ADAPTER=/root/job/adapter SAMPLE_OUT=/root/out/samples.jsonl python -u job/sample_persona_lora.py >> out/train.log 2>&1
  RC=$?; RESULT=out/samples.jsonl
else
  PERSONA_BASE=/root/base PERSONA_DATA=/root/job/${DATA_FILE:-} PERSONA_OUT=/root/out/adapter python -u job/train_persona_lora.py >> out/train.log 2>&1
  RC=$?; RESULT=out/adapter
fi
kill $LOOP
echo "== $(date -u) $JOB exit $RC" >> out/train.log
push out/train.log out/train.log
if [ $RC -eq 0 ] && [ -e $RESULT ]; then push $RESULT $RESULT && status done; else status failed; fi
sleep infinity  # stay up so the log can be read; Claude deletes the pod
