"""Pod session P4: exp90 (avoidance learning) and exp91 (assistant axis) on Qwen3-14B, then exp90 subset, exp91 and exp89 on Qwen3-8B.
Private files (live/server.py, the jlens package, the 8B Jacobian lens) come from a short-lived URL (--data-url, a .tgz).
Usage: python pod_p4.py --data-url URL [--dry] [--only "step ..."] [--patch POD_ID]"""
import argparse, json, pathlib, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--data-url", required=True); ap.add_argument("--branch", default="claude/exp51c"); ap.add_argument("--dry", action="store_true")
ap.add_argument("--patch", default=""); ap.add_argument("--only", default="", help="space-separated step names to run (rerun)")
args = ap.parse_args(); ROOT = pathlib.Path(__file__).resolve().parents[2]
key = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
BOOT = r"""
set -uo pipefail
log(){ echo "[p4 $(date +%H:%M:%S)] $*" | tee -a /workspace/progress.log; }
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace && rm -rf repo private && git clone -q --depth 1 -b BRANCH https://github.com/terrafying/ai-torture-chamber.git repo || { log "clone failed"; sleep infinity; }
(cd /workspace/repo/runs && python -m http.server 8000 >/dev/null 2>&1 &)
log "deps"
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > /workspace/pip.log 2>&1
pip install -q --no-cache-dir 'transformers==5.17.0' accelerate numpy scipy scikit-learn fastapi httpx redis pyyaml einops >> /workspace/pip.log 2>&1
pip uninstall -y -q torchvision torchaudio >> /workspace/pip.log 2>&1
python - <<'PY' || { log "private data failed"; sleep infinity; }
import io, tarfile, urllib.request
r = urllib.request.Request("DATAURL", headers={"User-Agent": "Mozilla/5.0 Chrome/130.0"})
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(r, timeout=1200).read()), mode="r:gz").extractall("/workspace/private")
print("private data unpacked")
PY
python -c "import torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())" > /workspace/repo/runs/env.txt 2>&1 || { log "env broken"; sleep infinity; }
export HF_HOME=/workspace/hf CHAMBER_DEVICE=cuda CHAMBER_DTYPE=bfloat16 EXP79_ROOT=/workspace/private JLENS_PKG=/workspace/private/jacobian-lens
R=/workspace/repo/runs
step(){ [ -n "ONLY" ] && [[ " ONLY " != *" $1 "* ]] && return 0; log "$1"; (cd $R/$2 && eval "$3") > $R/$2/$4 2>&1 || log "$1 FAILED"; cp /workspace/progress.log $R/ ; }
M14="CHAMBER_MODEL=Qwen/Qwen3-14B CHAMBER_LAYER=23"; M8="CHAMBER_MODEL=Qwen/Qwen3-8B CHAMBER_LAYER=18"
step exp90_14b exp90 "$M14 python -u run.py" run_14b.log
step exp91_14b exp91 "$M14 python -u run.py" run_14b.log
step exp90_8b  exp90 "$M8 EXP90_CONDS='pain|kv,fear|kv,egg|kv,none|kv' python -u run.py" run_8b.log
step exp91_8b  exp91 "$M8 python -u run.py" run_8b.log
step exp89_8b  exp89 "$M8 JLENS=/workspace/private/qwen3-8b_jacobian_lens.pt python -u run.py" run_8b.log
log "done"; touch $R/ALL_DONE; sleep infinity
""".replace("BRANCH", args.branch).replace("DATAURL", args.data_url).replace("ONLY", args.only)
body = {"name": "p4-14b", "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        "gpuTypeIds": ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"], "gpuTypePriority": "custom",
        "gpuCount": 1, "cloudType": "SECURE", "ports": ["8000/http"], "volumeInGb": 100, "volumeMountPath": "/workspace",
        "containerDiskInGb": 40, "env": {"HF_HOME": "/workspace/hf"}, "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOT]}
if args.dry: print(BOOT); raise SystemExit
if args.patch:
    r = urllib.request.urlopen(urllib.request.Request(f"https://rest.runpod.io/v1/pods/{args.patch}", data=json.dumps({k: body[k] for k in ("env", "dockerEntrypoint", "dockerStartCmd")}).encode(), headers=H, method="PATCH"))
    print("patched", args.patch, r.status); raise SystemExit
r = urllib.request.urlopen(urllib.request.Request("https://rest.runpod.io/v1/pods", data=json.dumps(body).encode(), headers=H, method="POST"))
pod = json.loads(r.read()); print(json.dumps({k: pod.get(k) for k in ("id", "costPerHr")}, default=str)); print(f"progress: https://{pod['id']}-8000.proxy.runpod.net/")
