#!/usr/bin/env python3
"""pod_worstchoice.py — deploy inversion-falcon/worstchoice.py on a RunPod GPU pod (14B).

Pattern borrowed from runpod_deploy.py (GraphQL pricing + REST v1 pod create, bootstrap with
fail-fast steps). Differences: no server — the bootstrap pip-installs, downloads the model,
runs runs/inversion-falcon/worstchoice.py with INVERSION_MODEL=Qwen/Qwen3-14B, then uploads
worstchoice_14b.json + the log to the private HF repo INVERSION_RESULTS_REPO (auto-created)
using HF_TOKEN. Requires INVERSION_RESULTS_REPO in ~/.hermes/.env (e.g. terrafying/inv-falcon-results).
"""
import json, os, pathlib, sys, time, urllib.error, urllib.request

key = [l.split("=", 1)[1].strip() for l in
       open(pathlib.Path.home() / ".hermes/.env") if l.startswith("RUNPOD_API_KEY=")][0]
env = {}
for l in open(pathlib.Path.home() / ".hermes/.env"):
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        env[l.split("=", 1)[0]] = l.split("=", 1)[1].strip()
HF_TOKEN = env.get("HF_TOKEN", "")
RESULTS_REPO = env.get("INVERSION_RESULTS_REPO", "")
H = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
REPO_URL = "https://github.com/terrafying/ai-torture-chamber.git"
assert REPO_URL.startswith("https://github.com/") and " " not in REPO_URL

BOOTSTRAP = r"""
set -euo pipefail
log(){ echo "[bootstrap $(date +%H:%M:%S)] $*"; }
log "step 1/6: git"
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace
[ -d repo/.git ] && { cd repo; git fetch -q --depth 1 origin master; git reset -q --hard origin/master; } || { git clone -q --depth 1 __REPO_URL__ repo && cd repo; }
log "step 2/6: pip (fail fast)"
pip install -q --no-cache-dir 'transformers>=4.51' 'accelerate' 'huggingface_hub' > /tmp/pip1.log 2>&1 || { log "pip FAILED"; tail -5 /tmp/pip1.log; exit 1; }
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || pip install -q --no-cache-dir 'torch>=2.4' --index-url https://download.pytorch.org/whl/cu121 > /tmp/pip2.log 2>&1 || { log "torch CUDA FAILED"; tail -5 /tmp/pip2.log; exit 1; }
python - <<'PYEOF'
import torch
assert torch.cuda.is_available()
print("[bootstrap] cuda ok:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, "GB", flush=True)
PYEOF
log "step 3/6: hf auth"
python -c "from huggingface_hub import HfApi; import os; api=HfApi(token='__HF_TOKEN__'); api.create_repo('__RESULTS_REPO__', private=True, exist_ok=True); print('results repo ok', flush=True)"
log "step 4/6: run worstchoice on 14B (model download ~28GB + capture + eval — hours; watch the log)"
cd repo
export HF_HOME=/workspace/hf INVERSION_MODEL=Qwen/Qwen3-14B HF_TOKEN='__HF_TOKEN__'
python runs/inversion-falcon/worstchoice.py > /workspace/worstchoice_14b.log 2>&1 || { log "worstchoice FAILED"; tail -20 /workspace/worstchoice_14b.log; exit 1; }
log "step 5/6: upload results"
python - <<'PYEOF'
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["RESULTS_REPO"]
api.upload_file(path_or_fileobj="/workspace/worstchoice_14b.log", path_in_repo="worstchoice_14b.log", repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj="runs/inversion-falcon/worstchoice_14b.json", path_in_repo="worstchoice_14b.json", repo_id=repo, repo_type="dataset")
print("uploaded", flush=True)
PYEOF
log "step 6/6: DONE — pod can be terminated"
sleep 300
""".replace("__REPO_URL__", REPO_URL).replace("__HF_TOKEN__", HF_TOKEN).replace("__RESULTS_REPO__", RESULTS_REPO)

def gql(query):
    req = urllib.request.Request("https://api.runpod.io/graphql?beta=true",
        data=json.dumps({"query": query}).encode(), headers=H)
    try:
        r = json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:400]}") from e
    if r.get("errors"):
        raise RuntimeError(r["errors"])
    return r["data"]

def rest(method, path, body=None):
    req = urllib.request.Request(f"https://rest.runpod.io/v1/{path}",
        data=json.dumps(body).encode() if body else None, headers=H, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=60)
        raw = r.read().decode()
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:1200]}") from e

data = gql('query { gpuTypes { id lowestPrice { minimumBidPrice } } }')
by_id = {g["id"]: g for g in data["gpuTypes"]}
chosen = None
for gpu in ("NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA A40", "NVIDIA GeForce RTX 4090",
            "NVIDIA L4", "NVIDIA RTX 5000 Ada Generation"):
    if gpu in by_id:
        chosen = gpu
        break
if not chosen:
    raise SystemExit("no preferred GPU type available: " + ", ".join(sorted(by_id)))
print("chosen GPU:", chosen)

# schema notes from runpod_deploy.py: gpuTypeIds array, ports array, env a string-valued
# object, bootstrap passed via args (no onStartScript field), containerImage not image.
# schema per docs.runpod.io/api-reference/pods (modified 2026-10-09): imageName/name,
# dockerStartCmd array overrides the start CMD (that's how the bootstrap runs), ports STRING.
pod_body = {
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuTypeIds": [chosen],
    "gpuCount": 1,
    "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
    "name": "inversion-falcon-worstchoice-14b",
    "containerDiskInGb": 40,
    "volumeInGb": 120,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
    "supportPublicIp": True,
    "env": {
        "HF_TOKEN": HF_TOKEN,
        "RESULTS_REPO": RESULTS_REPO,
    },
    "dockerStartCmd": ["/bin/bash", "-c", BOOTSTRAP],
}
pod_body["gpuTypeIds"] = [chosen]
r = None
for gpu in (chosen, "NVIDIA L40S", "NVIDIA A40", "NVIDIA GeForce RTX 4090",
            "NVIDIA L4", "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX 5000 Ada Generation",
            "NVIDIA A100-SXM4-40GB", "NVIDIA A100 80GB PCIe"):
    pod_body["gpuTypeIds"] = [gpu]
    try:
        r = rest("POST", "pods", pod_body)
        print("created on", gpu)
        break
    except RuntimeError as e:
        print(f"{gpu}: {e}", flush=True)
        if "no instances" in str(e) or "500" in str(e):
            continue
        raise
if r is None:
    raise SystemExit("all GPU candidates unavailable right now")
pid = r.get("id") or r.get("podId")
print("pod created:", pid, "| status:", r.get("desiredStatus"))
pathlib.Path(__file__).with_name("pod_14b.json").write_text(json.dumps(r, indent=1))
print("poll the pod with: rest GET /pods (id above); logs via RunPod console")
