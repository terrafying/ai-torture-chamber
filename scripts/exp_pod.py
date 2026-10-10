#!/usr/bin/env python3
"""Run an experiment script (default exp48) on a RunPod GPU pod: smoke -> full (Qwen3-4B, the pre-registered
run) -> replication on a larger model. Results are served read-only over the
pod's HTTP proxy at https://<pod>-8000.proxy.runpod.net/ so they can be
fetched with curl; the pod is then stopped (not terminated) by hand.

  python3 scripts/exp_pod.py [--script exp49_persuasion_vs_steering.py] [--big Qwen/Qwen3-14B] [--gpu "NVIDIA RTX A6000"]
Needs RUNPOD_API_KEY in ~/.hermes/.env. Same REST v1 shape as runpod_deploy.py.
"""
import argparse, json, pathlib, sys, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--big", default="Qwen/Qwen3-14B")
ap.add_argument("--script", default="exp48_emotion_binding.py",
                help="experiment script; results are served from runs/<expNN>/")
ap.add_argument("--gpu", default="NVIDIA RTX A6000,NVIDIA L40S", help="comma-separated, in preference order")
ap.add_argument("--models", default=None,
                help="comma-separated HF ids: run smoke+full for each in turn, continuing past "
                     "a failed model (marker FAILED_<name>), instead of the smoke/4B/big chain")
ap.add_argument("--volume", type=int, default=80)
ap.add_argument("--branch", default=None, help="git branch to clone (default: the repo's default branch)")
ap.add_argument("--pod", default=None, help="update this existing pod's start command instead of creating one")
args = ap.parse_args()
EXP = args.script.split("_")[0]          # e.g. exp49
if args.models:   # per-model smoke then full; a failed model is marked and skipped
    RUNS = "\n".join(
        f"n={m.split('/')[-1]}; log \"$n\"; "
        f"python {{SCRIPT}} --smoke --device cuda --model {m} > runs/{{EXP}}/$n.smoke.log 2>&1 && "
        f"python {{SCRIPT}} --device cuda --model {m} > runs/{{EXP}}/$n.log 2>&1 "
        f"|| {{ log \"$n FAILED\"; touch runs/{{EXP}}/FAILED_$n; }}"
        for m in args.models.split(","))
else:
    RUNS = """run(){ log "$1"; shift; "$@" || { log "FAILED"; touch runs/{EXP}/FAILED; sleep infinity; }; }
run smoke   sh -c 'python {SCRIPT} --smoke --device cuda > runs/{EXP}/smoke_gpu.log 2>&1'
run full-4B sh -c 'python {SCRIPT} --device cuda > runs/{EXP}/full.log 2>&1'
run big     sh -c 'python {SCRIPT} --device cuda --model {BIG} > runs/{EXP}/big.log 2>&1'"""

key = [l.split("=", 1)[1].strip() for l in open(pathlib.Path.home() / ".hermes/.env")
       if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
     "User-Agent": "Mozilla/5.0"}  # urllib default UA gets Cloudflare 1010
REPO_URL = "https://github.com/terrafying/ai-torture-chamber.git"
assert " " not in REPO_URL and REPO_URL.endswith(".git"), REPO_URL

BOOTSTRAP = r"""
set -uo pipefail
log(){ echo "[{EXP} $(date +%H:%M:%S)] $*"; }
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace && rm -rf repo && git clone -q --depth 1 {BRANCH}{REPO_URL} repo && cd repo || exit 1
mkdir -p runs/{EXP}
# serve results from the start, so progress is visible while it runs
(cd runs/{EXP} && python -m http.server 8000 >/dev/null 2>&1 &)
log "deps"
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > runs/{EXP}/pip.log 2>&1
pip install -q --no-cache-dir 'transformers==5.17.0' numpy accelerate bitsandbytes mistral-common >> runs/{EXP}/pip.log 2>&1
# the base image's torchvision/torchaudio are built for torch 2.4 and break
# transformers' lazy model imports ("Could not import module Qwen3ForCausalLM")
pip uninstall -y -q torchvision torchaudio >> runs/{EXP}/pip.log 2>&1
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())" > runs/{EXP}/env.txt 2>&1 || { log "no CUDA"; sleep infinity; }
python -c "import transformers.models.qwen3.modeling_qwen3 as m; print('qwen3 import ok', m.__name__)" >> runs/{EXP}/env.txt 2>&1 || { log "import broken"; touch runs/{EXP}/FAILED; sleep infinity; }
export HF_HOME=/workspace/hf
{RUNS}
log "done"; touch runs/{EXP}/ALL_DONE
sleep infinity
""".replace("{RUNS}", RUNS).replace("{REPO_URL}", REPO_URL).replace(
    "{BRANCH}", f"-b {args.branch} " if args.branch else "").replace("{BIG}", args.big).replace(
    "{EXP}", EXP).replace("{SCRIPT}", args.script)

def rest(method, path, body=None):
    req = urllib.request.Request(f"https://rest.runpod.io/v1/{path}",
                                 data=json.dumps(body).encode() if body else None,
                                 headers=H, method=method)
    try:
        raw = urllib.request.urlopen(req, timeout=60).read().decode()
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:1200]}")

if args.pod:   # PATCH resets the container with the new command; the volume (HF cache) stays
    pod = rest("PATCH", f"pods/{args.pod}", {"dockerEntrypoint": ["/bin/bash", "-c"],
                                             "dockerStartCmd": [BOOTSTRAP]})
    print("updated", args.pod, pod.get("desiredStatus"))
    sys.exit(0)
pod = rest("POST", "pods", {
    # REST v1 PodCreateInput (schema: https://rest.runpod.io/v1/openapi.json)
    "name": f"{EXP}-pod",
    "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
    "gpuTypeIds": [g.strip() for g in args.gpu.split(",")], "gpuTypePriority": "custom",
    "gpuCount": 1, "cloudType": "SECURE",
    "ports": ["8000/http"], "volumeInGb": args.volume, "volumeMountPath": "/workspace",
    "containerDiskInGb": 40, "env": {"HF_HOME": "/workspace/hf"},
    "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOTSTRAP],
})
pid = pod.get("id")
pathlib.Path(f"runs/{EXP}").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"runs/{EXP}/pod.json").write_text(json.dumps(pod, indent=1))
print("pod", pid, pod.get("desiredStatus"), "cost/hr", pod.get("costPerHr"))
print(f"results: https://{pid}-8000.proxy.runpod.net/")
