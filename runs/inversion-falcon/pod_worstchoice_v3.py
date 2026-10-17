#!/usr/bin/env python3
"""v3 of the 14B worstchoice deployer — preflight-reviewed (PREFLIGHT_CONSULT_RESPONSE.md).

Fixes over v1/v2 (the 5 dead launches + Claude's review):
- Cloudflare-safe UA on all RunPod calls (v1 killer).
- RESULTS_REPO under our own namespace, created + test-uploaded BEFORE launching the pod (v2 killer).
- git clone -b claude/exp51c --depth 1 + test -f on all three stimuli files (v3 killer).
- v8: image pytorch 2.6.0 (transformers 5.20.1 needs torch>=2.6; the 2.4.0 image made every v7 boot
  fail at import -> restart loop), ERR trap holds the container, tailer imports glob.
- deps: transformers==5.20.1 + huggingface_hub>=1.31,<2 pinned in order, no stray unpinned
  installs before the tailer starts (v4/v5 killer: unpinned step-1 resolve pulled httpx2,
  hub-0.36 pin broke transformers 5.20.1 imports).
- tailer: tracked PID (no pkill -f self-match), touch progress.log first, single batched
  create_commit per cycle (rate limit), uploads from /workspace/out (NOT the repo dir,
  which the next boot rm -rf's).
- every end state: upload what exists, write DONE/FAILED marker to progress.log,
  sleep infinity (never exit nonzero: RunPod restarts the container and would wipe results).
- pod outputs land in /workspace/out; results+log+npz all uploaded.
- launcher: single POST with all 48GB GPUs (no 24GB cards), no unused gql query, REST DELETE
  only, pod id persisted to runs/inversion-falcon/.pod_id, broad exception catch + pod-by-name
  sweep before retrying.
- HF token stays in env only (no __HF_TOKEN__ string replacement into the start cmd).
  .venv/bin/python3 runs/inversion-falcon/pod_worstchoice_v3.py
"""
import json, os, time, glob, urllib.request, urllib.error
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
POD_NAME = "inversion-falcon-worstchoice-14b-v8"

# ---- step 0: results repo exists + uploadable BEFORE any pod money is spent ----
def hf_call(fn):
    try:
        return fn()
    except Exception as e:
        raise SystemExit(f"preflight HF failure: {e}") from e
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["RESULTS_REPO"] = RESULTS_REPO
import subprocess
hf_call(lambda: subprocess.run([".venv/bin/python3", "-c",
    "from huggingface_hub import HfApi; import os, pathlib, tempfile\n"
    "api = HfApi(token=os.environ['HF_TOKEN'])\n"
    "api.create_repo(os.environ['RESULTS_REPO'], repo_type='dataset', private=True, exist_ok=True)\n"
    "p = tempfile.NamedTemporaryFile(suffix='.txt', delete=False); p.write(b'preflight'); p.close()\n"
    "api.upload_file(path_or_fileobj=p.name, path_in_repo='preflight.txt', repo_id=os.environ['RESULTS_REPO'], repo_type='dataset')\n"
    "print('preflight ok')"], check=True, capture_output=True))
print("step 0: results repo ok")

BOOTSTRAP = r"""
set -euo pipefail
log(){ echo "[bootstrap $(date -u +%H:%M:%S)] $1" | tee -a /workspace/progress.log; }
export HF_HOME=/workspace/hf
mkdir -p /workspace/out
: > /workspace/progress.log
# any failing command: mark FAILED and hold the container (exiting makes RunPod restart = billing loop)
trap 'rc=$?; log "FAILED line $LINENO rc=$rc"; tail -40 /workspace/worstchoice_14b.log >> /workspace/progress.log 2>/dev/null || true; sleep infinity' ERR
log "step 1/6: deps (pinned, before any use)"
pip install -q "transformers==5.20.1" "huggingface_hub>=1.31,<2" scipy accelerate >>/workspace/progress.log 2>&1
log "step 2/6: tailer"
cat > /workspace/tailer.py <<'PYEOF'
import os, time, glob
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["RESULTS_REPO"]
n = 0
while True:
    time.sleep(180); n += 1
    ops = []
    for f, nm in (("/workspace/progress.log", "progress.log"),
                  ("/workspace/worstchoice_14b.log", "worstchoice_14b.log"),
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
log "step 3/6: git (branch pin + file asserts)"
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; } >>/workspace/progress.log 2>&1
rm -rf /workspace/repo
git clone -q --depth 1 -b __BRANCH__ __REPO_URL__ /workspace/repo
cd /workspace/repo
git rev-parse HEAD >> /workspace/progress.log
for f in runs/inversion-falcon/worstchoice.py runs/inversion-falcon/stimuli.py runs/exp119/stimuli.py; do
  test -f "$f" || { log "MISSING $f"; false; }
done
log "step 4/6: torch sanity (no reinstall — image must provide CUDA)"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')" >>/workspace/progress.log 2>&1 || { log "torch FAILED (host/driver)"; log "FAILED torch"; sleep infinity; }
log "step 5/6: run"
export INVERSION_MODEL=Qwen/Qwen3-14B
python runs/inversion-falcon/worstchoice.py > /workspace/worstchoice_14b.log 2>&1
cp runs/inversion-falcon/worstchoice_14b.json /workspace/out/ 2>/dev/null || true
cp runs/inversion-falcon/subspace96_14b.npz /workspace/out/ 2>/dev/null || true
log "step 6/6: upload results (retry x3)"
python - <<'PYEOF'
import os, time, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["RESULTS_REPO"]
for attempt in range(3):
    try:
        for f, nm in (("/workspace/worstchoice_14b.log", "worstchoice_14b.log"),) + tuple(
                (p, "out/" + os.path.basename(p)) for p in glob.glob("/workspace/out/*")):
            api.upload_file(path_or_fileobj=f, path_in_repo=nm, repo_id=repo, repo_type="dataset")
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
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:600]}") from e
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__} on {method} {path}: {e}") from e

# ---- cleanup any previous pod of ours by name (not hard-coded ids) ----
existing = rest("GET", "pods")
for p in existing if isinstance(existing, list) else existing.get("pods", []):
    if p.get("name", "").startswith("inversion-falcon-worstchoice-14b") and p.get("desiredStatus") not in ("EXITED",):
        print("stopping leftover pod", p["id"])
        try: rest("POST", f"pods/{p['id']}/stop")
        except Exception as e: print("stop:", str(e)[:100])
        time.sleep(8)
        try: rest("DELETE", f"pods/{p['id']}")
        except Exception as e: print("delete:", str(e)[:100])

pod_body = {
    "cloudType": "SECURE", "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA A40",
                    "NVIDIA RTX 6000 Ada Generation"],
    "imageName": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
    "name": POD_NAME,
    "containerDiskInGb": 40, "volumeInGb": 120, "volumeMountPath": "/workspace",
    "ports": [], "supportPublicIp": False,
    "env": {"HF_TOKEN": HF_TOKEN, "RESULTS_REPO": RESULTS_REPO},
    "dockerStartCmd": ["/bin/bash", "-c", BOOTSTRAP],
}
pod = None
try:
    pod = rest("POST", "pods", pod_body)
except Exception as e:
    print("single POST failed:", str(e)[:200])
    # sweep: did the POST actually land before erroring?
    try:
        for p in (_ := rest("GET", "pods")) if isinstance(_, list) else _.get("pods", []):
            if p.get("name") == POD_NAME and p.get("desiredStatus") == "RUNNING":
                pod = p; print("found running pod by name:", p["id"]); break
    except Exception as e2:
        print("sweep failed:", str(e2)[:100])
if not pod:
    raise SystemExit("no pod launched")
pid = pod.get("id")
Path(__file__).parent.joinpath(".pod_id").write_text(str(pid))
print("POD_ID=" + str(pid))
print("watch: progress.log + DONE/FAILED markers in", RESULTS_REPO)