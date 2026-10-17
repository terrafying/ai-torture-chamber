#!/usr/bin/env python3
"""v2 of the 14B worstchoice deployer — OBSERVABLE this time.

v1 lesson (2026-10-17): 2.5h of pod time, zero output, no way to see why (ports blocked, logs
trapped on the pod volume, HF upload only at the very end). v2 keeps a background loop that
uploads progress.log + worstchoice_14b.log to the private HF results repo every 3 minutes from
the FIRST minute, and uploads the tail on any failure path. Local watcher polls the same files
on HF, so a hang is visible within minutes and the pod can be killed early.
  .venv/bin/python3 runs/inversion-falcon/pod_worstchoice_v2.py
"""
import json, os, time, urllib.request, urllib.error
from pathlib import Path

env = {}
for line in open(Path.home() / ".hermes" / ".env"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1)
        env[k] = v.strip().strip('"')
KEY = env["RUNPOD_API_KEY"]
HF_TOKEN = env.get("HF_TOKEN", "")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) AppleWebKit/537.36"}  # default urllib UA trips Cloudflare 1010
RESULTS_REPO = "Badgersex/inversion-falcon-results"
REPO_URL = "https://github.com/terrafying/ai-torture-chamber.git"
assert REPO_URL.startswith("https://github.com/")

BOOTSTRAP = r"""
set -euo pipefail
log(){ echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/workspace/hf
log "step 1/6: tailer"   # incremental log uploads: the whole point of v2
pip install -q huggingface_hub >/dev/null 2>&1 || pip install -q huggingface_hub
cat > /workspace/tailer.py <<'PYEOF'
import os, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["RESULTS_REPO"]
def up(path, name):
    try: api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo, repo_type="dataset")
    except Exception as e: print("tailer skip", name, e, flush=True)
up("/workspace/progress.log", "progress.log")
n = 0
while True:
    time.sleep(180)
    n += 1
    open("/workspace/progress.log", "a").write(f"[tailer {n} alive {time.strftime('%H:%M:%S')}]\n")
    up("/workspace/progress.log", "progress.log")
    for f, nm in (("/workspace/worstchoice_14b.log", "worstchoice_14b.log"),
                  ("/workspace/bootstrap.log", "bootstrap.log")):
        if os.path.exists(f): up(f, nm)
PYEOF
( HF_TOKEN='__HF_TOKEN__' RESULTS_REPO='__RESULTS_REPO__' python /workspace/tailer.py >> /workspace/tailer.log 2>&1 & )
log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; echo "[bootstrap $(date -u +%H:%M:%S)] $*" >> /workspace/progress.log; }
log "step 2/6: git"
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; } >> /workspace/progress.log 2>&1
rm -rf /workspace/repo
git clone -q --depth 1 -b claude/exp51c __REPO_URL__ /workspace/repo && cd /workspace/repo || { log "git FAILED"; exit 1; }
git rev-parse HEAD >> /workspace/progress.log
test -f runs/inversion-falcon/worstchoice.py || { log "clone MISSING worstchoice.py (stale/partial clone)"; exit 1; }
log "step 3/6: torch"
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { log "installing torch (slow)"; pip install -q --no-cache-dir 'torch>=2.4' --index-url https://download.pytorch.org/whl/cu121 >> /workspace/progress.log 2>&1 || { log "torch FAILED"; tail -5 /workspace/progress.log; exit 1; }; }
log "step 4/6: deps"
pip install -q transformers scipy accelerate 2>>/workspace/progress.log || log "deps partial"
log "step 5/6: run"
export INVERSION_MODEL=Qwen/Qwen3-14B HF_TOKEN='__HF_TOKEN__'
python runs/inversion-falcon/worstchoice.py > /workspace/worstchoice_14b.log 2>&1 || { log "worstchoice FAILED — uploading tail"; python - <<'PYFAIL'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
t = open("/workspace/worstchoice_14b.log").read()[-4000:]
open("/workspace/fail_tail.txt","w").write(t)
api.upload_file(path_or_fileobj="/workspace/fail_tail.txt", path_in_repo="fail_tail.txt", repo_id=os.environ["RESULTS_REPO"], repo_type="dataset")
PYFAIL
exit 1; }
log "step 6/6: upload results"
python - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["RESULTS_REPO"]
api.upload_file(path_or_fileobj="/workspace/worstchoice_14b.log", path_in_repo="worstchoice_14b.log", repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj="runs/inversion-falcon/worstchoice_14b.json", path_in_repo="worstchoice_14b.json", repo_id=repo, repo_type="dataset")
print("uploaded", flush=True)
PYEOF
pkill -f tailer.py
log "DONE"
sleep 60
""".replace("__REPO_URL__", REPO_URL).replace("__HF_TOKEN__", HF_TOKEN).replace("__RESULTS_REPO__", RESULTS_REPO)

def rest(method, path, body=None):
    req = urllib.request.Request(f"https://rest.runpod.io/v1/{path}",
        data=json.dumps(body).encode() if body else None, headers=H, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=60)
        raw = r.read().decode()
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:600]}") from e

def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql?beta=true",
        data=json.dumps({"query": q}).encode(), headers=H)
    r = json.load(urllib.request.urlopen(req, timeout=60))
    if r.get("errors"): raise RuntimeError(r["errors"])
    return r["data"]

# old pod: DELETE it (its pod-volume HF cache dies with it; v2 redownloads — the price of
# observability, and the old pod's volume is the only place the blind logs live, unreadable)
try:
    rest("POST", "pods/wntak372hwrdz0/delete" if False else "pods/wntak372hwrdz0/stop")
    time.sleep(10)
except Exception as e:
    print("old pod stop:", e)
# try the documented delete endpoint shapes
deleted = False
for ep in (f"pods/wntak372hwrdz0", ):
    try:
        rest("DELETE", ep); print("deleted via DELETE", ep); deleted = True; break
    except Exception as e:
        print("delete try:", str(e)[:100])
if not deleted:
    try:
        d = gql('mutation { podDelete(input: {podId: "wntak372hwrdz0"}) { id success } }')
        print("deleted via gql:", d); deleted = True
    except Exception as e:
        print("gql delete:", str(e)[:200])

data = gql('query { gpuTypes { id lowestPrice { minimumBidPrice } } }')
by_id = {g["id"]: g for g in data["gpuTypes"]}
pod_body = {
    "cloudType": "SECURE", "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA L40S"],
    "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
    "name": "inversion-falcon-worstchoice-14b-v2",
    "containerDiskInGb": 40, "volumeInGb": 120, "volumeMountPath": "/workspace",
    "ports": ["22/tcp"], "supportPublicIp": True,
    "env": {"HF_TOKEN": HF_TOKEN, "RESULTS_REPO": RESULTS_REPO},  # PUBLIC_KEY auto-injected by the API; passing an empty string gets 403/1010
    "dockerStartCmd": ["/bin/bash", "-c", BOOTSTRAP],
}
pod = None
for gpu in ("NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX 6000 Ada Generation", "NVIDIA L4"):
    pod_body["gpuTypeIds"] = [gpu]
    try:
        pod = rest("POST", "pods", pod_body)
        print("created on", gpu, pod.get("id"))
        break
    except RuntimeError as e:
        print(gpu, ":", str(e)[:150])
if not pod:
    raise SystemExit("no GPU available")
print("POD_ID=" + str(pod.get("id")))
