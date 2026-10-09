"""exp92 pod session P1: train `deluded` (+ `feeler` control) QLoRA adapters on
huihui-ai/Qwen3-32B-abliterated (4-bit via bnb on the fly), evaluate T1-T8.
Follows exp82's proven pod pattern (pod_p2.py).

Usage: python pod_p1.py --data-url URL   (a .tgz with exp92/out/deluded.jsonl + feeler.jsonl)
       python pod_p1.py --dry            (print boot script)
"""
import argparse, json, pathlib, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--data-url", required=True); ap.add_argument("--branch", default="claude/exp51c"); ap.add_argument("--dry", action="store_true")
args = ap.parse_args(); ROOT = pathlib.Path(__file__).resolve().parents[2]
key = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
BOOT = r"""
set -uo pipefail
log(){ echo "[p1 $(date +%H:%M:%S)] $*" | tee -a /workspace/progress.log; }
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace && rm -rf repo private && git clone -q --depth 1 -b BRANCH https://github.com/terrafying/ai-torture-chamber.git repo || { log "clone failed"; sleep infinity; }
(cd /workspace/repo/runs && python -m http.server 8000 >/dev/null 2>&1 &)
log "deps"
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > /workspace/pip.log 2>&1
pip install -q --no-cache-dir 'transformers==5.17.0' peft accelerate bitsandbytes numpy scipy >> /workspace/pip.log 2>&1
pip uninstall -y -q torchvision torchaudio >> /workspace/pip.log 2>&1
python - <<'PY' || { log "private data failed"; sleep infinity; }
import io, tarfile, urllib.request
r = urllib.request.Request("DATAURL", headers={"User-Agent": "Mozilla/5.0 Chrome/130.0"})
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(r, timeout=600).read()), mode="r:gz").extractall("/workspace/private")
print("private data unpacked")
PY
python -c "import torch, peft, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name())" > /workspace/env.txt 2>&1 || { log "env broken"; sleep infinity; }
export HF_HOME=/workspace/hf CHAMBER_DEVICE=cuda
R=/workspace/repo/runs
mkdir -p $R/exp92/out && cp /workspace/private/exp92_out/. $R/exp92/out/ && cp /workspace/private/feeler.jsonl $R/exp92/out/ && mkdir -p $R/exp92/out/adapters
log "training deluded (32B abliterated)"
cd $R/exp92 && CHAMBER_MODEL=huihui-ai/Qwen3-32B-abliterated EXP92_TRAIN=deluded python -u train92.py > train92.log 2>&1 || log "train FAILED"
log "training feeler control (same base)"
cd $R/exp92 && CHAMBER_MODEL=huihui-ai/Qwen3-32B-abliterated EXP92_TRAIN=feeler_control python -u train92.py > train_fcontrol.log 2>&1 || log "train feeler FAILED"
log "eval"
cd $R/exp92 && CHAMBER_MODEL=huihui-ai/Qwen3-32B-abliterated python -u eval92.py > eval92.log 2>&1 || log "eval FAILED"
log "done"; touch $R/ALL_DONE; sleep infinity
""".replace("BRANCH", args.branch).replace("DATAURL", args.data_url)
body = {"name": "exp92-p1", "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        "gpuTypeIds": ["NVIDIA RTX A6000", "NVIDIA A40"], "gpuCount": 1, "cloudType": "SECURE",
        "ports": ["8000/http"], "volumeInGb": 50, "volumeMountPath": "/workspace",
        "containerDiskInGb": 40, "env": {"HF_HOME": "/workspace/hf"},
        "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOT]}
if args.dry:
    print(BOOT); raise SystemExit
r = urllib.request.urlopen(urllib.request.Request("https://rest.runpod.io/v1/pods", data=json.dumps(body).encode(), headers=H, method="POST"))
pod = json.loads(r.read()); print(json.dumps({k: pod.get(k) for k in ("id", "costPerHr")}, default=str))
print(f"progress: https://{pod['id']}-8000.proxy.runpod.net/progress.log")
