"""Launch the exp79 (+exp80) pod on RunPod REST v1. Usage: python pod.py [--branch claude/exp51c] [--dry]
The pod serves /workspace/repo/runs on port 8000 from the start; stop it with runpodctl when ALL_DONE appears."""
import argparse, json, pathlib, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--branch", default="claude/exp51c"); ap.add_argument("--dry", action="store_true")
ap.add_argument("--v2", default="", help="old pod URL: run v2 (trickster+, simulacrum+) instead of v1")
ap.add_argument("--patch", default="", help="existing pod id: restart it with this bootstrap instead of creating a pod")
ap.add_argument("--data-url", default="", help="private, short-lived URL of a .tgz of local data files (see make_v2.py)")
args = ap.parse_args()
ROOT = pathlib.Path(__file__).resolve().parents[2]
key = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
REPO = "https://github.com/terrafying/ai-torture-chamber.git"
BOOT = r"""
set -uo pipefail
log(){ echo "[exp79 $(date +%H:%M:%S)] $*" | tee -a /workspace/progress.log; }
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace && rm -rf repo master && git clone -q --depth 1 -b BRANCH REPO repo && git clone -q --depth 1 REPO master || { log "clone failed"; sleep infinity; }
cp /workspace/progress.log repo/runs/ 2>/dev/null; (cd /workspace/repo/runs && python -m http.server 8000 >/dev/null 2>&1 &)
log "deps"
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > repo/runs/pip.log 2>&1
pip install -q --no-cache-dir 'transformers==5.17.0' peft accelerate numpy fastapi httpx redis >> repo/runs/pip.log 2>&1
pip uninstall -y -q torchvision torchaudio >> repo/runs/pip.log 2>&1
python -c "import torch, peft, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name())" > repo/runs/env.txt 2>&1 || { log "env broken"; touch repo/runs/FAILED; sleep infinity; }
export HF_HOME=/workspace/hf CHAMBER_MODEL=Qwen/Qwen3-8B CHAMBER_DEVICE=cuda CHAMBER_DTYPE=bfloat16 CHAMBER_LAYER=18 EXP79_ROOT=/workspace/master
cd /workspace/repo/runs/exp79
if [ -n "__OLD__" ]; then
  log "v2 data"; EXP79_V2=watchman,stoic,denier,trickster,simulacrum python -u make_v2.py > make_v2.log 2>&1 || { log "v2 data failed"; touch ../FAILED; sleep infinity; }
  log "v2 train"; EXP79_TRAIN=watchman_plus,stoic_plus,denier_plus,trickster_plus,simulacrum_plus python -u train.py > train_v2.log 2>&1 || { log "v2 train failed"; touch ../FAILED; sleep infinity; }
  log "v2 eval"; EXP79_SKIP_BASE=1 EXP79_EVAL=watchman_plus,stoic_plus,denier_plus,trickster_plus,simulacrum_plus python -u eval.py > eval_v2.log 2>&1 || { log "v2 eval failed"; touch ../FAILED; }
  log "done"; touch /workspace/repo/runs/ALL_DONE; sleep infinity
fi
log "data";  python -u make_data.py > data.log 2>&1  || { log "data failed"; touch ../FAILED; sleep infinity; }
log "train"; python -u train.py > train.log 2>&1    || { log "train failed"; touch ../FAILED; sleep infinity; }
log "eval";  python -u eval.py > eval.log 2>&1      || { log "eval failed"; touch ../FAILED; }
cd ../exp80
log "exp80"; python -u run.py > run.log 2>&1        || { log "exp80 failed"; touch ../FAILED; }
log "done"; touch /workspace/repo/runs/ALL_DONE
sleep infinity
""".replace("BRANCH", args.branch).replace("REPO", REPO).replace("__OLD__", args.v2)
DATA_ENV = {"EXP79_DATA_URL": args.data_url} if args.data_url else {}
body = {"name": "exp79-v2" if args.v2 else "exp79-zoo", "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        "gpuTypeIds": ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"],
        "gpuTypePriority": "custom", "gpuCount": 1, "cloudType": "SECURE", "ports": ["8000/http"],
        "volumeInGb": 60, "volumeMountPath": "/workspace", "containerDiskInGb": 40, "env": {"HF_HOME": "/workspace/hf", **DATA_ENV},
        "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOT]}
if args.dry: print(BOOT); raise SystemExit
if args.patch:
    r = urllib.request.urlopen(urllib.request.Request(f"https://rest.runpod.io/v1/pods/{args.patch}", data=json.dumps({k: body[k] for k in ("env", "dockerEntrypoint", "dockerStartCmd")}).encode(), headers=H, method="PATCH"))
    print("patched", args.patch, r.status); raise SystemExit
r = urllib.request.urlopen(urllib.request.Request("https://rest.runpod.io/v1/pods", data=json.dumps(body).encode(), headers=H, method="POST"))
pod = json.loads(r.read()); print(json.dumps({k: pod.get(k) for k in ("id", "costPerHr", "machine", "desiredStatus")}, default=str))
print(f"progress: https://{pod['id']}-8000.proxy.runpod.net/")
