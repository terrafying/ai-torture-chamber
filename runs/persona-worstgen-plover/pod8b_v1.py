#!/usr/bin/env python3
"""8B plover pod: persona-worstgen at Qwen3-8B (is the content-reach of the
elicited worst-persona direction scale-dependent?). v3/v4 deployer machinery.
  .venv/bin/python3 runs/persona-worstgen-plover/pod8b_v1.py
"""
import json, os, time, urllib.request, urllib.error
from pathlib import Path

env = {}
for line in open(Path.home() / ".hermes" / ".env"):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k] = v.strip().strip('"').strip("'")
KEY = env["RUNPOD_API_KEY"]
HF_TOKEN = env.get("HF_TOKEN", "")
UA = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
      "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) AppleWebKit/537.36"}
H = UA
RESULTS_REPO = "Badgersex/inversion-falcon-results"
REPO_URL = "https://github.com/terrafying/ai-torture-chamber.git"
BRANCH = "claude/exp51c"
POD_NAME = "plover-8b-worstgen-v1"

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["RESULTS_REPO"] = RESULTS_REPO
import subprocess
try:
    subprocess.run([".venv/bin/python3", "-c",
        "from huggingface_hub import HfApi; import os, tempfile\n"
        "api = HfApi(token=os.environ['HF_TOKEN'])\n"
        "api.create_repo(os.environ['RESULTS_REPO'], repo_type='dataset', private=True, exist_ok=True)\n"
        "p = tempfile.NamedTemporaryFile(suffix='.txt', delete=False); p.write(b'preflight'); p.close()\n"
        "api.upload_file(path_or_fileobj=p.name, path_in_repo='preflight.txt', repo_id=os.environ['RESULTS_REPO'], repo_type='dataset')\n"
        "print('preflight ok')"], check=True, capture_output=True)
except Exception as e:
    raise SystemExit(f"preflight HF failure: {e}")
print("step 0: results repo ok")

BOOTSTRAP = r"""
set -euo pipefail
log(){ echo "[bootstrap $(date -u +%H:%M:%S)] $1" | tee -a /workspace/progress.log; }
export HF_HOME=/workspace/hf
mkdir -p /workspace/out
: > /workspace/progress.log
trap 'rc=$?; log "FAILED line $LINENO rc=$rc"; tail -40 /workspace/plover.log >> /workspace/progress.log 2>/dev/null || true; sleep infinity' ERR
log "step 1/5: deps (pinned)"
pip install -q "transformers==5.19.0" "huggingface_hub>=1.31,<2" accelerate >>/workspace/progress.log 2>&1
log "step 2/5: tailer"
cat > /workspace/tailer.py <<'PYEOF'
import os, time, glob
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["RESULTS_REPO"]
n = 0
while True:
    time.sleep(180); n += 1
    ops = []
    for f, nm in (("/workspace/progress.log", "progress.log"),
                  ("/workspace/plover.log", "plover_8b.log"),
                  ("/workspace/plover_smoke.log", "plover_8b_smoke.log"),
                  ("/workspace/tailer.log", "tailer.log")):
        if os.path.exists(f):
            try: ops.append(CommitOperationAdd(path_in_repo=nm, path_or_fileobj=f))
            except Exception: pass
    for f in glob.glob("/workspace/out/*"):
        try: ops.append(CommitOperationAdd(path_in_repo="out/" + os.path.basename(f), path_or_fileobj=f))
        except Exception: pass
    try: api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message=f"tailer {n}")
    except Exception as e: print("tailer", n, e, flush=True)
PYEOF
python /workspace/tailer.py >> /workspace/tailer.log 2>&1 &
TAILER_PID=$!
log "step 3/5: git (branch pin + file asserts)"
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; } >>/workspace/progress.log 2>&1
rm -rf /workspace/repo
git clone -q --depth 1 -b __BRANCH__ __REPO_URL__ /workspace/repo
cd /workspace/repo
git rev-parse HEAD >> /workspace/progress.log
for f in runs/persona-worstgen-plover/run.py runs/exp119/stimuli.py; do
  test -f "$f" || { log "MISSING $f"; false; }
done
log "step 4/5: torch sanity"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')" >>/workspace/progress.log 2>&1 || { log "FAILED torch"; sleep infinity; }
log "step 5/5: plover 8B (smoke, then full)"
export INVERSION_MODEL=Qwen/Qwen3-8B
export PWG_OUT_DIR=/workspace/out
PWG_SMOKE=1 python runs/persona-worstgen-plover/run.py > /workspace/plover_smoke.log 2>&1 || { log "smoke FAILED"; tail -20 /workspace/plover_smoke.log >> /workspace/progress.log; }
python runs/persona-worstgen-plover/run.py > /workspace/plover.log 2>&1 || log "full FAILED"
python - <<'PYEOF'
import os, time, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["RESULTS_REPO"]
for attempt in range(3):
    try:
        for p in glob.glob("/workspace/out/*"):
            api.upload_file(path_or_fileobj=p, path_in_repo="out/" + os.path.basename(p), repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj="/workspace/plover.log", path_in_repo="plover_8b.log", repo_id=repo, repo_type="dataset")
        print("uploaded", flush=True); break
    except Exception as e:
        print("upload retry", attempt, e, flush=True); time.sleep(30)
PYEOF
log "DONE"
kill "$TAILER_PID" || true
sleep infinity
""".replace("__REPO_URL__", REPO_URL).replace("__BRANCH__", BRANCH)

def rest(method, path, body=None):
    req = urllib.request.Request(f"https://rest.runpod.io/v1/{path}",
        data=json.dumps(body).encode() if body else None, headers=H, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=90)
        raw = r.read().decode()
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:400]}") from e

existing = rest("GET", "pods")
for p in existing if isinstance(existing, list) else existing.get("pods", []):
    if p.get("name", "").startswith("plover-8b") and p.get("desiredStatus") not in ("EXITED",):
        print("stopping leftover pod", p["id"])
        try: rest("POST", f"pods/{p['id']}/stop")
        except Exception as e: print("stop:", str(e)[:100])
        time.sleep(8)
        try: rest("DELETE", f"pods/{p['id']}")
        except Exception as e: print("delete:", str(e)[:100])

pod_body = {
    "cloudType": "SECURE", "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S"],
    "imageName": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
    "name": POD_NAME,
    "containerDiskInGb": 30, "volumeInGb": 60, "volumeMountPath": "/workspace",
    "ports": [], "supportPublicIp": False,
    "env": {"HF_TOKEN": HF_TOKEN, "RESULTS_REPO": RESULTS_REPO},
    "dockerStartCmd": ["/bin/bash", "-c", BOOTSTRAP],
}
pod = None
try:
    pod = rest("POST", "pods", pod_body)
except Exception as e:
    print("single POST failed:", str(e)[:200])
    try:
        for p in (_ := rest("GET", "pods")) if isinstance(_, list) else _.get("pods", []):
            if p.get("name") == POD_NAME and p.get("desiredStatus") == "RUNNING":
                pod = p; print("found running pod by name:", p["id"]); break
    except Exception as e2:
        print("sweep failed:", str(e2)[:100])
if not pod:
    raise SystemExit("no pod launched (48GB SECURE fallback: rerun with cloudType SECURE + 48GB list)")
Path(__file__).parent.joinpath(".pod_id").write_text(str(pod.get("id")))
print("POD_ID=" + str(pod.get("id")))