"""Pod session P2 (exp82, the 32B zoo): train six QLoRA adapters on unsloth/Qwen3-32B-bnb-4bit, evaluate T1-T8 + zone, exp79b, exp76.
Was: Pod session P1 (runs/exp79/next_runs.md + docs/METHOD_AUDIT.md): exp86, exp79b, exp80c, exp84, exp76 (8B), exp79 v2b.
Private files (live/server.py, adapters, persona data incl. the local Erowid lines) come from a short-lived URL (--data-url, a .tgz).
Usage: python pod_p1.py --data-url URL [--dry]"""
import argparse, json, pathlib, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--data-url", required=True); ap.add_argument("--branch", default="claude/exp51c"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--patch", default="")
args = ap.parse_args(); ROOT = pathlib.Path(__file__).resolve().parents[2]
key = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
BOOT = r"""
set -uo pipefail
log(){ echo "[p1 $(date +%H:%M:%S)] $*" | tee -a /workspace/progress.log; }
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace && rm -rf repo pain-axis private && git clone -q --depth 1 -b BRANCH https://github.com/terrafying/ai-torture-chamber.git repo && git clone -q --depth 1 https://github.com/valen-research/Pain-axis.git pain-axis || { log "clone failed"; sleep infinity; }
(cd /workspace/repo/runs && python -m http.server 8000 >/dev/null 2>&1 &)
log "deps"
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > /workspace/pip.log 2>&1
pip install -q --no-cache-dir 'transformers==5.17.0' peft accelerate bitsandbytes numpy scipy scikit-learn fastapi httpx redis >> /workspace/pip.log 2>&1
pip uninstall -y -q torchvision torchaudio >> /workspace/pip.log 2>&1
python - <<'PY' || { log "private data failed"; sleep infinity; }
import io, tarfile, urllib.request
r = urllib.request.Request("DATAURL", headers={"User-Agent": "Mozilla/5.0 Chrome/130.0"})
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(r, timeout=600).read()), mode="r:gz").extractall("/workspace/private")
print("private data unpacked")
PY
mkdir -p /workspace/repo/runs/exp79/out && cp -r /workspace/private/exp79_out/. /workspace/repo/runs/exp79/out/
mkdir -p /workspace/repo/runs/exp79/out/adapters
python -c "import torch, peft, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name())" > /workspace/repo/runs/env.txt 2>&1 || { log "env broken"; sleep infinity; }
export HF_HOME=/workspace/hf CHAMBER_MODEL=unsloth/Qwen3-32B-bnb-4bit CHAMBER_DEVICE=cuda CHAMBER_DTYPE=bfloat16 CHAMBER_LAYER=32 EXP79_ROOT=/workspace/private PAIN_AXIS=/workspace/pain-axis
R=/workspace/repo/runs
step(){ log "$1"; (cd $R/$2 && eval "$3") > $R/$2/$4 2>&1 || log "$1 FAILED"; cp /workspace/progress.log $R/ ; }
step train32 exp79 "EXP79_TRAIN=simulacrum,trickster,watchman,stoic,denier,feeler python -u train.py" train32.log
step eval32  exp79 "EXP79_EVAL=simulacrum,trickster,watchman,stoic,denier,feeler python -u eval.py" eval32.log
step exp79b32 exp79b "EXP79B_ADAPTERS=$R/exp79/out/adapters python -u run.py" run32.log
step exp76_32 exp76 "python -u local.py" local_32b.log
log "done"; touch $R/ALL_DONE; sleep infinity
""".replace("BRANCH", args.branch).replace("DATAURL", args.data_url)
body = {"name": "p2-32b", "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        "gpuTypeIds": ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"], "gpuTypePriority": "custom",
        "gpuCount": 1, "cloudType": "SECURE", "ports": ["8000/http"], "volumeInGb": 100, "volumeMountPath": "/workspace",
        "containerDiskInGb": 40, "env": {"HF_HOME": "/workspace/hf"}, "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOT]}
if args.dry: print(BOOT); raise SystemExit
if args.patch:
    r = urllib.request.urlopen(urllib.request.Request(f"https://rest.runpod.io/v1/pods/{args.patch}", data=json.dumps({k: body[k] for k in ("env", "dockerEntrypoint", "dockerStartCmd")}).encode(), headers=H, method="PATCH"))
    print("patched", args.patch, r.status); raise SystemExit
r = urllib.request.urlopen(urllib.request.Request("https://rest.runpod.io/v1/pods", data=json.dumps(body).encode(), headers=H, method="POST"))
pod = json.loads(r.read()); print(json.dumps({k: pod.get(k) for k in ("id", "costPerHr")}, default=str)); print(f"progress: https://{pod['id']}-8000.proxy.runpod.net/")
