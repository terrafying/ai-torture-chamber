Preflight review request (pure infra, no experiment design): our lab runs GPU pods
(RunPod, pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime image) for long python sweeps.
Today ONE deploy script burned 5 pod launches on 5 distinct failures. Please review
`runs/inversion-falcon/pod_worstchoice_v2.py` (read it) and tell us what else in it will
break, in order of likelihood, with concrete fixes. The 5 failures we already hit, so you
don't re-derive them:
1. urllib default UA -> Cloudflare 1010 on ALL rest.runpod.io writes (fixed: browser UA).
2. results repo namespace we can't create under this HF token (fixed: use our own namespace).
3. git clone without -b cloned the default branch, runner file missing (fixed: -b <branch> --depth 1).
4. image's fresh pip resolve pulled httpx2, hub downloads crashed (TypeError process()).
5. our fix for 4 (pin huggingface_hub==0.36.2) broke transformers 5.20.1 imports
   (ResolvedRevision ImportError) - image ships transformers 5.20.1 which needs hub>=1.31.
Current state: transformers==5.20.1 + hub>=1.31,<2 pinned in order. Questions:
- Will hub 1.x + the image's httpx work for large model downloads, or do we still need an httpx pin? What's the known-good set for transformers 5.20.1?
- The bootstrap has a background tailer (uploads progress.log to HF every 3 min), a pkill at the end, sleep 60, and relies on RunPod restarting the container on nonzero exit. What's fragile about that pattern?
- Anything else in the script that will bite: env var quoting, the dockerStartCmd array, volume size, ports, the gql delete fallbacks, the stop-then-delete sequence for relaunches?
- What's the minimal smoke test we can run INSIDE a fresh pod (60s, no GPU work) that would have caught failures 3-5 before the real run?
Be specific and quantitative where possible.
