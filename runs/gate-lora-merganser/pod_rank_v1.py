#!/usr/bin/env python3
"""Loop-3 ranking pod: 14B cloze over the full arm-B candidate pool.
v3/v4 deployer machinery (preflighted), single job. Pod deleted by the operator on DONE.
  .venv/bin/python3 runs/gate-lora-merganser/pod_rank_v1.py
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
POD_NAME = "merganser-loop3-rank-14b-v1"

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
trap 'rc=$?; log "FAILED line $LINENO rc=$rc"; tail -40 /workspace/rank.log >> /workspace/progress.log 2>/dev/null || true; sleep infinity' ERR
log "step 1/5: deps (pinned)"
pip install -q "transformers==5.20.1" "huggingface_hub>=1.31,<2" scipy accelerate >>/workspace/progress.log 2>&1
log "step 2/5: tailer"
cat > /workspace/tailer.py <<'PYEOF'
import os, time
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["RESULTS_REPO"]
n = 0
while True:
    time.sleep(180); n += 1
    ops = []
    for f, nm in (("/workspace/progress.log", "progress.log"),
                  ("/workspace/rank.log", "rank.log"),
                  ("/workspace/tailer.log", "tailer.log")):
        if os.path.exists(f):
            try: ops.append(CommitOperationAdd(path_in_repo=nm, path_or_fileobj=f))
            except Exception: pass
    for f in ("/workspace/out/ranked.json", "/workspace/out/ranked_smoke.json"):
        import os as _o
        if _o.path.exists(f):
            try: ops.append(CommitOperationAdd(path_in_repo="out/" + _o.path.basename(f), path_or_fileobj=f))
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
for f in runs/gate-lora-merganser/rank_judge.py runs/gate-lora-merganser/candidates_full.json; do
  test -f "$f" || { log "MISSING $f"; false; }
done
log "step 4/5: torch sanity"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')" >>/workspace/progress.log 2>&1 || { log "FAILED torch"; sleep infinity; }
log "step 5/5: rank (14B cloze over 2814 candidates)"
export INVERSION_MODEL=Qwen/Qwen3-14B
export WC_OUT_DIR=/workspace/out
RANK_SMOKE=1 python runs/gate-lora-merganser/rank_judge.py > /workspace/rank_smoke.log 2>&1 || { log "rank SMOKE FAILED"; tail -20 /workspace/rank_smoke.log >> /workspace/progress.log; }
python runs/gate-lora-merganser/rank_judge.py > /workspace/rank.log 2>&1 || log "rank FAILED"
cp /workspace/out/ranked.json /workspace/out/ 2>/dev/null || true
python - <<'PYEOF'
import os, time, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["RESULTS_REPO"]
for attempt in range(3):
    try:
        for p in glob.glob("/workspace/out/*"):
            api.upload_file(path_or_fileobj=p, path_in_repo="out/" + os.path.basename(p), repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj="/workspace/rank.log", path_in_repo="rank.log", repo_id=repo, repo_type="dataset")
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

existing = rest("GET", "pods")
for p in existing if isinstance(existing, list) else existing.get("pods", []):
    if p.get("name", "").startswith("merganser-loop3-rank") and p.get("desiredStatus") not in ("EXITED",):
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
    try:
        for p in (_ := rest("GET", "pods")) if isinstance(_, list) else _.get("pods", []):
            if p.get("name") == POD_NAME and p.get("desiredStatus") == "RUNNING":
                pod = p; print("found running pod by name:", p["id"]); break
    except Exception as e2:
        print("sweep failed:", str(e2)[:100])
if not pod:
    raise SystemExit("no pod launched")
Path(__file__).parent.joinpath(".pod_id").write_text(str(pod.get("id")))
print("POD_ID=" + str(pod.get("id")))