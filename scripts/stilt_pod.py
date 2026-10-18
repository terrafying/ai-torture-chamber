#!/usr/bin/env python3
"""Run refusal-neuron-stilt on a RunPod A40: clone the clean public branch (claude/stilt-pod, which carries
ONLY the stilt + exp119 files, no other branch history), then STAGE=search -> intervene -> cap. Results are
served read-only at https://<pod>-8000.proxy.runpod.net/ ; poll for ALL_DONE, fetch, then stop the pod by hand.
Gate side only. Needs RUNPOD_API_KEY in ~/.hermes/.env.  Same REST v1 shape as scripts/exp_pod.py.
  python3 scripts/stilt_pod.py [--gpu "NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S"] [--pod <id to re-run>]"""
import argparse, json, pathlib, sys, urllib.error, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S")
ap.add_argument("--volume", type=int, default=60)
ap.add_argument("--pod", default=None)
args = ap.parse_args()
D = "runs/refusal-neuron-stilt"
REPO = "https://github.com/terrafying/ai-torture-chamber.git"
LAYERS = ",".join(str(i) for i in range(36))   # full-model neuron search (pre-reg: rank all 36*9728)
BOOTSTRAP = r"""
set -uo pipefail
log(){ echo "[stilt $(date +%H:%M:%S)] $*"; }
command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }
cd /workspace && rm -rf repo && git clone -q --depth 1 -b claude/stilt-pod REPO repo && cd repo || exit 1
(cd DIR && python -m http.server 8000 >/dev/null 2>&1 &)
log "deps"
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > DIR/pip.log 2>&1
pip install -q --no-cache-dir 'transformers==5.17.0' numpy scipy accelerate >> DIR/pip.log 2>&1
pip uninstall -y -q torchvision torchaudio >> DIR/pip.log 2>&1
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())" > DIR/env.txt 2>&1 || { log no-cuda; sleep infinity; }
python -c "import transformers.models.qwen3.modeling_qwen3 as m; print('qwen3 ok')" >> DIR/env.txt 2>&1 || { touch DIR/FAILED; sleep infinity; }
export HF_HOME=/workspace/hf RN_LAYERS="LAYERSLIST"
run(){ log "$1"; STAGE="$1" python DIR/run.py > "DIR/$1.log" 2>&1 || { log "$1 FAILED"; touch "DIR/FAILED_$1"; }; }
run search
run intervene
run cap
log done; touch DIR/ALL_DONE
sleep infinity
""".replace("REPO", REPO).replace("DIR", D).replace("LAYERSLIST", LAYERS)
key = [l.split("=", 1)[1].strip() for l in open(pathlib.Path.home() / ".hermes/.env") if l.startswith("RUNPOD_API_KEY=")][0]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
def rest(method, path, body=None):
    req = urllib.request.Request(f"https://rest.runpod.io/v1/{path}", data=json.dumps(body).encode() if body else None, headers=H, method=method)
    try:
        raw = urllib.request.urlopen(req, timeout=60).read().decode(); return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:1000]}")
if args.pod:
    p = rest("PATCH", f"pods/{args.pod}", {"dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOTSTRAP]})
    print("updated", args.pod, p.get("desiredStatus")); sys.exit(0)
pod = rest("POST", "pods", {
    "name": "stilt-pod", "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
    "gpuTypeIds": [g.strip() for g in args.gpu.split(",")], "gpuTypePriority": "custom",
    "gpuCount": 1, "cloudType": "SECURE", "ports": ["8000/http"],
    "volumeInGb": args.volume, "volumeMountPath": "/workspace", "containerDiskInGb": 40,
    "env": {"HF_HOME": "/workspace/hf"}, "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOTSTRAP]})
pid = pod.get("id")
pathlib.Path(D).mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{D}/pod.json").write_text(json.dumps(pod, indent=1))
print("pod", pid, pod.get("desiredStatus"), "cost/hr", pod.get("costPerHr"))
print(f"results: https://{pid}-8000.proxy.runpod.net/   (poll for ALL_DONE)")
