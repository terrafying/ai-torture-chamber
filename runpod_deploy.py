#!/usr/bin/env python3
"""runpod_deploy.py — create the Saw chamber GPU pod (A6000/L40S, on-demand).

Pricing via GraphQL gpuTypes (still live); pod creation via REST v1
(POST /v1/pods) — the GraphQL podDeploy mutation was sunset server-side
(2026-09-30: "Unknown type PodCreateInput" with no client change).

Bootstrap = bash with `set -euo pipefail`: clone --depth 1, pip install
(both stages FAIL FAST instead of swallowing the exit status through a
pipe), a CUDA preflight assert, uvicorn on 8000/http, and a /health wait
loop. Runpod gives a public proxy URL once runtime.ports appear.

Env-driven: RUNPOD_API_KEY in ~/.hermes/.env. Print the pod id + URL.
"""
import json, os, pathlib, sys, time, urllib.error, urllib.request

key = [l.split("=", 1)[1].strip() for l in
       open(pathlib.Path.home() / ".hermes/.env") if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

REPO_URL = "https://github.com/terrafying/ai-torture-chamber.git"
# guards the exact corruption this URL has hit twice before (an identity-scrub
# pass mangled it into the literal string "https://repo (private).git", which
# then crash-loops the pod silently while it keeps billing) — fail fast here
# instead of deploying a broken bootstrap.
assert REPO_URL.startswith("https://github.com/") and REPO_URL.endswith(".git") \
    and " " not in REPO_URL, f"REPO_URL looks corrupted: {REPO_URL!r}"

# checkpoint revision pins, per model — the 4B hash is the one the
# independent chamber reset verified (docs/chamber-audit.md); an unpinned
# download silently tracks Qwen updates and breaks cross-run comparability.
# The deploy default is Qwen3-32B (unpinned until audited): pass
# CHAMBER_MODEL_REVISION when deploying a model that has a verified pin.
MODEL_REVISIONS = {"Qwen/Qwen3-4B":
                   "1cfa9a7208912126459214e8b04321603b3df60c"}
MODEL_REVISION = MODEL_REVISIONS.get(
    os.environ.get("CHAMBER_MODEL", "Qwen/Qwen3-32B"), "")

BOOTSTRAP = r"""
set -euo pipefail
log(){ echo "[bootstrap $(date +%H:%M:%S)] $*"; }
log "step 1/5: git"
command -v git >/dev/null || { apt-get update -qq &&
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace
if [ -d repo/.git ]; then
  cd repo; log "existing repo: fetch/reset to origin/master"
  git fetch -q --depth 1 origin master && git reset -q --hard origin/master
else
  git clone -q --depth 1 {REPO_URL} repo && cd repo
fi
cd live
log "step 2/5: pip requirements (errors FAIL FAST — check the tail below)"
pip install -q --no-cache-dir -r requirements.txt 'transformers>=4.51' \
  > /tmp/pip1.log 2>&1 || { log "pip requirements FAILED"; tail -5 /tmp/pip1.log; exit 1; }
log "step 3/5: CUDA torch wheel (requirements.txt pins the CPU wheel for Railway)"
pip install -q --no-cache-dir 'torch>=2.4' \
  --index-url https://download.pytorch.org/whl/cu121 \
  > /tmp/pip2.log 2>&1 || { log "torch CUDA install FAILED"; tail -5 /tmp/pip2.log; exit 1; }
log "step 4/5: preflight — CUDA must be real before anything listens on 8000"
python - <<'PYEOF'
import torch, transformers, fastapi
assert torch.cuda.is_available(), "torch.cuda.is_available() is False: CPU torch shipped"
print("[bootstrap] cuda ok:", torch.cuda.get_device_name(0), flush=True)
PYEOF
export HF_HOME=/workspace/hf CHAMBER_DEVICE=cuda CHAMBER_DTYPE=float16 \
  CHAMBER_MODEL=${CHAMBER_MODEL:-Qwen/Qwen3-32B} PORT=8000
if [ -n "{MODEL_REVISION}" ]; then
  export CHAMBER_MODEL_REVISION={MODEL_REVISION}
fi
log "step 5/5: uvicorn on 8000 (first boot downloads ~8GB weights — slow /health is normal)"
python -m uvicorn server:app --host 0.0.0.0 --port 8000 &
UV=$!
for i in $(seq 1 120); do
  sleep 5
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    log "HEALTH OK after ~$((i*5))s — pod ready"; break
  fi
  kill -0 "$UV" 2>/dev/null || { log "uvicorn died before /health — check logs above"; exit 1; }
  [ $i -eq 120 ] && log "no /health after 600s — leaving uvicorn running, check pod logs"
done
wait "$UV"
""".replace("{REPO_URL}", REPO_URL).replace("{MODEL_REVISION}", MODEL_REVISION)

GPU_TYPES = ["NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA GeForce RTX 4090"]

def gql(query, variables=None):
    req = urllib.request.Request(
        "https://api.runpod.io/graphql?beta=true",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers=H)
    try:
        r = json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    if r.get("errors"):
        raise RuntimeError(r["errors"])
    return r["data"]

def rest(method, path, body=None):
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/{path}",
        data=json.dumps(body).encode() if body else None,
        headers=H, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=60)
        raw = r.read().decode()
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        # error bodies carry an OpenAPI `problems` array — that IS the schema
        # documentation (enums, array-vs-object, key names). Print it whole.
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:1200]}") from e

# ---- pricing: GraphQL gpuTypes still answers (REST has no gpuTypes path) ----
chosen = None
data = gql("""query { gpuTypes { id lowestPrice { minimumBidPrice } } }""")
by_id = {g["id"]: g for g in data["gpuTypes"]}
for gpu in GPU_TYPES:
    g = by_id.get(gpu)
    if not g:
        print(f"{gpu}: not listed", flush=True)
        continue
    price = g["lowestPrice"]["minimumBidPrice"] if g.get("lowestPrice") else None
    print(f"{gpu}: min-bid ${price}/hr", flush=True)
    if price and price <= 0.85:
        chosen = (gpu, price)
        break
if not chosen:
    sys.exit("no suitable GPU under $0.85/hr")
gpu, price = chosen
print(f"deploying on {gpu} @ min-bid ${price}/hr", flush=True)

# ---- create: REST v1. Schema notes from the deployed procedure:
# gpuTypeIds is an ARRAY; ports is an ARRAY of strings; env is a
# STRING-VALUED OBJECT (not the GraphQL key/value list); cloudType only
# SECURE|COMMUNITY. On a 4xx, the problems array above names the offending
# field — fix the body, don't guess.
pod_body = {
    "podName": "saw-chamber-gpu",
    "containerImage": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
    "gpuTypeIds": [gpu],
    "gpuCount": 1,
    "cloudType": "SECURE",
    "ports": ["8000/http"],
    "volumeInGb": 40,
    "volumeMountPath": "/workspace",
    "env": {"HF_HOME": "/workspace/hf"},
    "args": ["/bin/bash", "-c", BOOTSTRAP],
    "supportPublicIp": True,
    "startSsh": False,
}
print("creating pod via REST v1...", flush=True)
pod = rest("POST", "pods", pod_body)
pid = pod.get("id") or pod.get("podId")
print("pod created:", pid, "| status:", pod.get("desiredStatus"), flush=True)
pathlib.Path("runs/exp39").mkdir(parents=True, exist_ok=True)
pathlib.Path("runs/exp39/runpod_pod.json").write_text(json.dumps(pod, indent=1))

# ---- wait for runtime, then verify /health before claiming success ----
print(f"proxy url once running: https://{pid}-8000.proxy.runpod.net", flush=True)
deadline = time.time() + 600
while time.time() < deadline:
    time.sleep(30)
    p = rest("GET", f"pods/{pid}")
    ports = (p.get("runtime") or {}).get("ports")
    if ports:
        print("runtime up:", [(x.get("port"), x.get("ip")) for x in ports], flush=True)
        break
    print("...no runtime yet (provisioning can take minutes)", flush=True)
else:
    sys.exit("no runtime after 10 min — resume once, then re-provision on a "
             "different host if the resume errors GPU-free (see runpod skill)")
url = f"https://{pid}-8000.proxy.runpod.net"
for attempt in range(20):
    time.sleep(15)
    try:
        h = json.load(urllib.request.urlopen(f"{url}/health", timeout=15))
        print("/health:", h, flush=True)
        sys.exit(0 if h.get("ok") else "health endpoint answering but ok=false")
    except Exception:
        print(f"health attempt {attempt + 1}: not ready yet (first boot pulls "
              f"~8GB of weights)", flush=True)
sys.exit("/health never answered — check pod logs: bootstrap now fails fast "
         "with a [bootstrap] step marker naming the failed step")
