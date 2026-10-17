I reviewed `pod_worstchoice_v2.py`, but I couldn't check anything online: PyPI lookups and `git ls-tree` on the remote branch were both blocked by permissions. So the dependency-version answers below come from reading code and what I already know, and the smoke test is designed to settle what I couldn't check.

## Will break, most likely first

**1. `pkill -f tailer.py` (line 89) can kill the bootstrap itself.** `pkill -f` matches the full command line. The bootstrap runs as `bash -c "<whole script>"`, and that string contains `tailer.py`, so bash matches too. If bash isn't PID 1 (RunPod may add an init process), it dies with exit 143. RunPod then restarts the container, wipes `/workspace/repo` and runs the whole 14B job again, on every successful run. If bash is PID 1, the kernel ignores the SIGTERM and you get away with it. Don't rely on either.
- **Fix:** `python /workspace/tailer.py >> /workspace/tailer.log 2>&1 & TAILER_PID=$!` (drop the `( … & )` subshell; it hides `$!`), then `kill "$TAILER_PID" || true`.

**2. Relying on container restarts is a billing loop, not recovery.**
- **Repeat failures retry forever.** Every `exit 1` path (git, torch, run) re-runs from the top. So does a single HF 5xx in step 6, because `set -e` is on.
- **Even success probably restarts.** I don't think RunPod pods have a "completed" state. The `sleep 60` → exit 0 most likely restarts too and re-runs the job.
- **A restart deletes your results.** `worstchoice_14b.json` and `subspace96_14b.npz` are written into `/workspace/repo/runs/inversion-falcon/`, which the next boot `rm -rf`s. If the step 6 upload fails, the restart destroys the results.
- **Fix:** every end state uploads what exists, writes `DONE` or `FAILED <step>` to progress.log, then `sleep infinity`. The local watcher stops and deletes the pod when it sees either marker. Write outputs to `/workspace/out/`, not the repo, and retry uploads (3×, 30 s backoff). Also upload the `.npz`; right now it's never uploaded.

**3. The tailer fails silently if the results repo doesn't exist.** Nothing creates `Badgersex/inversion-falcon-results`. A missing repo, a read-only token or a wrong namespace all raise inside `up()`. The tailer prints that to `tailer.log`, which lives on the pod, and you're blind again, just like v1.
- **Fix:** in the launcher, before creating the pod, run `HfApi(token).create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)` plus one test `upload_file`. If either fails, abort locally for $0.
- **Also:** `touch /workspace/progress.log` before starting the tailer. Its first upload currently runs before the file exists, which leaves you blind for the first 3 minutes.

**4. The unpinned `pip install huggingface_hub` at step 1 (line 32) is the stray resolve.** It's the container's first fresh resolve, so it's the likely source of the httpx2 mess. The tailer starts on whatever it installs. Then step 4 swaps hub/httpx underneath that running process.
- **Fix:** move the pinned install to step 1 and start the tailer after it.
- **Your premise is suspect:** `pytorch/pytorch:2.4.0` is a mid-2024 image and should not ship transformers 5.x. Something earlier installed it. The smoke test's first step settles this.

**5. Two of the fallback GPUs can't hold the model.** Qwen3-14B in bf16 is about 29.5 GB of weights. The RTX 4090 and L4 have 24 GB, so they OOM after a full download. The 48 GB cards (L40S, A6000, A40, 6000 Ada) are fine.
- **Fix:** drop the 4090 and L4.
- **Load directly to GPU:** `.to(DEV)` builds the full model in CPU RAM first. `device_map="cuda"` loads it straight onto the GPU.

**6. Launcher paths that can kill or orphan things:**
- **Line 132 is unused.** `gql('query { gpuTypes … }')` feeds `by_id`, which nothing reads. If GraphQL is flaky (your memory notes parts of it are sunset), it aborts the launch after the old pod is already stopped. Delete it.
- **The GraphQL delete fallback can't work.** `podDelete` isn't a mutation; the one that exists is `podTerminate(input:{podId})`. REST `DELETE /v1/pods/{id}` is the right call: keep it, drop the fallback.
- **The old pod ID is hard-coded.** Each relaunch cleans up `wntak372hwrdz0`, never the pod it just made, so orphans pile up. A stopped 120 GB volume is roughly $24/month per orphan. Fix: write the new ID to `runs/inversion-falcon/.pod_id` and clean up that pod on the next run. The `"…delete" if False else "…stop"` line is dead code.
- **A timeout can create an unseen pod.** `urlopen` timeouts raise `URLError`/`TimeoutError`, not `RuntimeError`, so the GPU loop crashes. The POST may have succeeded on RunPod's side, leaving a billing pod you never saw. Fix: catch broadly, and on any error `GET pods` filtered by `name` before trying the next GPU. Simpler still: send all acceptable GPUs in one `gpuTypeIds` list and make a single POST.

## Should fix

**7. The HF token is baked into the start command.** The `__HF_TOKEN__` replacements on lines 52 and 69 make it readable via `GET /pods/{id}`, the dashboard and `ps`. It's already passed in `env`, so use `"$HF_TOKEN"` and drop the replace. HF tokens are `[A-Za-z0-9_]`, so the single-quoting is safe; leakage is the issue.

**8. Failures that get swallowed or waste time:**
- **Deps failure isn't fatal.** `|| log "deps partial"` lets the run continue into an import error. Make it fatal.
- **Step 3 tries to reinstall torch when CUDA is missing.** The image already has CUDA torch, so `cuda.is_available()` being false means a host or driver problem. Reinstalling (it would pull torch 2.5.x from the cu121 index) won't fix that and costs about 5 minutes. Fail immediately instead.
- **`pip check`** will always complain on a conda-based image. It's noise.

**9. Tailer commit volume.** Each `upload_file` is a separate HF commit, so 3 files every 3 min is up to 60 commits an hour. Batch them into one `create_commit` per cycle so a long run doesn't hit HF rate limits.

**10. Unverified, and blocking if wrong.** `worstchoice.py` imports `runs/inversion-falcon/stimuli.py` and `runs/exp119/stimuli.py`. I couldn't confirm they're committed on `origin/claude/exp51c`. Check with `git ls-tree -r --name-only origin/claude/exp51c -- runs/exp119/stimuli.py runs/inversion-falcon/stimuli.py`. Add `test -f` for both next to the existing check.

**11. Fine as is:**
- **Disk sizes:** 40 GB container disk and 120 GB volume are enough (30 GB weights plus about 10 GB of hf_xet cache, since `HF_HOME` is on the volume).
- **`dockerStartCmd` without `dockerEntrypoint`:** works because this image has no ENTRYPOINT.
- **`ports: ["22/tcp"]`** does nothing: the image has no sshd.
- **The local `.env` parser** doesn't handle `export X=` lines or single-quoted values. Minor.

## Your questions

**Hub 1.x and httpx:** don't add a separate httpx pin. As far as I know, hub 1.x uses httpx itself and declares its own compatible range. Big Qwen downloads mostly go through `hf_xet` (a hub dependency on x86_64), not httpx. I can't tell you what pulled `httpx2`. The smoke test runs `pip show httpx2` and its "Required-by" line answers that.

**Known-good set for transformers 5.20.1:** I can't verify one from here. Get it empirically: once the smoke test passes, `pip freeze` the relevant lines into `runs/inversion-falcon/constraints.txt`. After that, always install with `pip install -c constraints.txt …` using exact `==` versions, not ranges.

## Smoke test (no GPU work)

Run it on the cheapest acceptable GPU so the driver check is real. If transformers has to be installed, expect 90–180 s, not 60. Put this in `dockerStartCmd` with the same env:

```bash
set -uo pipefail; L=/workspace/smoke.log; exec > >(tee -a $L) 2>&1
python -V; pip list 2>/dev/null | grep -iE '^(torch|transformers|huggingface|httpx|hf.xet|tokenizers) '   # BEFORE any pip: settles "image ships 5.20.1"
command -v git || { apt-get update -qq && apt-get install -y -qq git; }
git clone -q --depth 1 -b claude/exp51c https://github.com/terrafying/ai-torture-chamber.git /tmp/r && git -C /tmp/r rev-parse HEAD
for f in runs/inversion-falcon/worstchoice.py runs/inversion-falcon/stimuli.py runs/exp119/stimuli.py; do test -f /tmp/r/$f && echo "OK $f" || echo "MISSING $f"; done   # failure 3
pip install -q <EXACT step-1 pin line>; pip show httpx2 2>/dev/null | grep -i required-by
python - <<'PY'
import os, torch, transformers, huggingface_hub, httpx
print("versions", torch.__version__, transformers.__version__, huggingface_hub.__version__, httpx.__version__)  # failure 5
from transformers import AutoModelForCausalLM, AutoTokenizer                                                   # failure 5
from huggingface_hub import hf_hub_download, HfApi
print(hf_hub_download("Qwen/Qwen3-14B", "config.json"))                                                        # failure 4 (real download path)
AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_properties(0).total_memory/1e9, "GB")
PY
pip freeze | grep -iE '^(transformers|huggingface|httpx|hf.xet|tokenizers|safetensors|accelerate)=' 
python -c "import os;from huggingface_hub import HfApi;HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='$L',path_in_repo='smoke.log',repo_id=os.environ['RESULTS_REPO'],repo_type='dataset')"  # failure 2
sleep infinity
```

Read `smoke.log` from HF, then stop the pod yourself.

**A free preflight:** most of this doesn't need a pod. Run it on the Mac with `docker run --platform linux/amd64 pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime bash -c '<same script minus the cuda line>'`. Emulation is slow but costs $0, and it would have caught failures 3, 4 and 5.

I haven't edited the script. Say if you want me to apply these fixes.
