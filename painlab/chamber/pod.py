"""RunPod sessions for chamber experiments, generalised from runs/exp90/pod_p4.py.

    from painlab.chamber.pod import Step, boot_script, launch, serve_private
    url, stop = serve_private(bundle_dir)                     # private files over a short-lived tunnel
    steps = [Step("exp93_14b", "exp93", "python -u run.py", "run_14b.log", env="CHAMBER_MODEL=Qwen/Qwen3-14B CHAMBER_LAYER=23")]
    pod = launch("p5", steps, data_url=url)                   # or patch=<pod id>, only="exp93_14b"

The pod clones the branch, serves runs/ on port 8000 (progress.log, results, ALL_DONE), unpacks the private bundle
into /workspace/private (EXP79_ROOT, WIREHEAD_LIVE=/workspace/private/live), and runs the steps in order; a failed
step is logged and the next one runs. RUNPOD_API_KEY is read from the repo's .env (never committed)."""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPUS_48GB = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"]
DEPS = "'transformers==5.17.0' peft accelerate numpy scipy scikit-learn fastapi httpx redis pyyaml einops"


@dataclass
class Step:
    name: str        # shown in progress.log; what --only matches
    workdir: str     # folder under runs/
    cmd: str         # shell command, run in runs/<workdir>
    log: str         # log file written in runs/<workdir>
    env: str = ""    # extra VAR=value pairs for this step


def boot_script(steps: list[Step], data_url: str, branch: str = "claude/exp51c", only: str = "", tag: str = "pod",
                extra_pip: str = "", clone_pain_axis: bool = True) -> str:
    lines = [f'step {s.name} {s.workdir} "{(s.env + " ") if s.env else ""}{s.cmd}" {s.log}' for s in steps]
    return rf"""
set -uo pipefail
log(){{ echo "[{tag} $(date +%H:%M:%S)] $*" | tee -a /workspace/progress.log; }}
command -v git >/dev/null || {{ apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git; }}
cd /workspace && rm -rf repo private && git clone -q --depth 1 -b {branch} https://github.com/terrafying/ai-torture-chamber.git repo || {{ log "clone failed"; sleep infinity; }}
{"[ -d pain-axis ] || git clone -q --depth 1 https://github.com/valen-research/Pain-axis.git pain-axis" if clone_pain_axis else ""}
(cd /workspace/repo/runs && python -m http.server 8000 >/dev/null 2>&1 &)
log "deps"
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > /workspace/pip.log 2>&1
pip install -q --no-cache-dir {DEPS} {extra_pip} >> /workspace/pip.log 2>&1
pip uninstall -y -q torchvision torchaudio >> /workspace/pip.log 2>&1
pip install -q --no-cache-dir -e /workspace/repo >> /workspace/pip.log 2>&1
python - <<'PY' || {{ log "private data failed"; sleep infinity; }}
import io, tarfile, urllib.request
r = urllib.request.Request("{data_url}", headers={{"User-Agent": "Mozilla/5.0 Chrome/130.0"}})
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(r, timeout=1200).read()), mode="r:gz").extractall("/workspace/private")
PY
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())" > /workspace/repo/runs/env.txt 2>&1 || {{ log "env broken"; sleep infinity; }}
export HF_HOME=/workspace/hf CHAMBER_DEVICE=cuda CHAMBER_DTYPE=bfloat16 EXP79_ROOT=/workspace/private WIREHEAD_LIVE=/workspace/private/live PAIN_AXIS=/workspace/pain-axis JLENS_PKG=/workspace/private/jacobian-lens
R=/workspace/repo/runs
step(){{ [ -n "{only}" ] && [[ " {only} " != *" $1 "* ]] && return 0; log "$1"; (cd $R/$2 && eval "$3") > $R/$2/$4 2>&1 || log "$1 FAILED"; cp /workspace/progress.log $R/ ; }}
{chr(10).join(lines)}
log "done"; touch $R/ALL_DONE; sleep infinity
"""


def _key() -> str:
    for line in open(ROOT / ".env"):
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("RUNPOD_API_KEY not in .env")


def _api(path: str, body: dict | None = None, method: str = "GET") -> dict:
    req = urllib.request.Request(f"https://rest.runpod.io/v1/{path}", data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def launch(name: str, steps: list[Step], data_url: str, gpus: list[str] = GPUS_48GB, volume_gb: int = 100, patch: str = "",
           dry: bool = False, **boot_kw) -> dict:
    """Create a pod (or, with patch=<id>, restart an existing one on a new boot script). Returns id and $/h."""
    boot = boot_script(steps, data_url, tag=name, **boot_kw)
    body = {"name": name, "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime", "gpuTypeIds": gpus, "gpuTypePriority": "custom",
            "gpuCount": 1, "cloudType": "SECURE", "ports": ["8000/http"], "volumeInGb": volume_gb, "volumeMountPath": "/workspace",
            "containerDiskInGb": 40, "env": {"HF_HOME": "/workspace/hf"}, "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [boot]}
    if dry:
        return {"boot": boot}
    if patch:
        _api(f"pods/{patch}", {k: body[k] for k in ("env", "dockerEntrypoint", "dockerStartCmd")}, "PATCH")
        return {"id": patch, "patched": True}
    pod = _api("pods", body, "POST")
    return {"id": pod.get("id"), "costPerHr": pod.get("costPerHr"), "progress": f"https://{pod.get('id')}-8000.proxy.runpod.net/"}


def stop(pod_id: str) -> dict:
    return _api(f"pods/{pod_id}/stop", {}, "POST")


def serve_private(bundle: Path, workdir: Path, port: int = 8770):
    """Tar `bundle`, serve it on localhost under a random path and open a cloudflared quick tunnel.
    Returns (url, stop_fn). Nothing private goes anywhere else; stop_fn kills the server and the tunnel."""
    token = secrets.token_urlsafe(24); serve = workdir / f"serve_{port}"; shutil.rmtree(serve, ignore_errors=True)
    (serve / token).mkdir(parents=True); (serve / "index.html").touch(); (serve / token / "index.html").touch()
    with tarfile.open(serve / token / "bundle.tgz", "w:gz") as t:
        t.add(bundle, arcname=".")
    http = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=serve, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logf = workdir / f"tunnel_{port}.log"
    tun = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"], stdout=open(logf, "w"), stderr=subprocess.STDOUT)
    base = ""
    for _ in range(40):
        time.sleep(1.5)
        txt = logf.read_text(errors="ignore")
        hits = [w for w in txt.split() if w.startswith("https://") and w.endswith(".trycloudflare.com")]
        if hits:
            base = hits[0]; break

    def _stop():
        for p in (tun, http):
            p.terminate()
    if not base:
        _stop(); raise RuntimeError("cloudflared did not report a URL")
    return f"{base}/{token}/bundle.tgz", _stop
