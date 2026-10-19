---
name: experiment-run-ops
description: "Run long experiment jobs safely: orphans, races, resume."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [experiments, long-running-jobs, mps, mac-cluster, resumable-artifacts]
    related_skills: []
---

# Experiment Run Ops (long-running local/remote jobs)

For any multi-hour compute job this user runs: experiment scripts in the
research repos (fractal-basins-lab, seed scanners, render pipelines), on the
Mac cluster (minipro24 = M4 Pro 24GB local, mini-mini.local = M4 at
10.10.10.2 thunderbolt or 192.168.3.2, atlas-mbp = M2 Pro at 10.0.0.119
wifi, user id `e` not `ee`) or Colab. The class pattern: resumable
self-describing artifacts + one-writer discipline.

## Pre-flight (before launching any run)

1. **Check for orphan processes from earlier sessions** running the same
   script: `pgrep -f <script>`; use `ps -o pid,lstart,command` and
   `ps eww <pid>` (reads the child env, e.g. FB_RES/FB_SEEDS) plus
   `lsof -p <pid> | grep cwd` for provenance. Kill orphans or wait.
   Symptom if skipped: files that existed then vanished, summaries stuck at
   stale protocol values.
2. **Inventory the run dir before touching it** (`ls -la` + timestamps).
   Never `rm` artifacts you did not create in this session without checking
   whether a live process is mid-write.
3. **ONE writer per run dir.** Splitting across machines? Split by task
   namespace, not by seed within the same directory.
4. **Check for co-resident agent sessions on the same machine before loading
   heavy compute or writing in a shared repo** — another CLI agent (claude,
   codex) may be mid-run in the same tree: `ps aux | grep -E 'claude|codex'`
   then `lsof -p <pid> | grep cwd` for its working repo, plus a remote-host
   pgrep when its job runs via ssh poller. Confirm its run dir differs from
   yours (and remember the process table only shows LOCAL load — a remote
   worker under an ssh watcher is invisible to local pgrep). Lane check,
   don't assume, and don't kill its wrappers: they may be live ssh pollers.
   When the user says a co-resident agent is about to SHIP, clear your own
   working tree first (`git checkout --` uncommitted edits, stash WIP files)
   — a half-edited file in a shared tree is what its pipeline deploys or
   conflicts on, and a stale in-progress edit of a file it just changed is
   unmergeable noise.
   

## Artifact discipline

- **Self-describing artifacts**: every npz/json stores its config alongside
  results (resolution, seed, params, input identity, measured difficulty).
  This is what makes reuse-vs-stale decisions possible later.
- **Long runners dump incrementally, per cell/seed, to an append-only
  jsonl** — a runner that accumulates rows and `json.dump`s once at the end
  loses EVERYTHING on an early kill, and this user kills jobs early by design
  ("skip the nulls unless the results call for it"). One row per cell appended
  as computed; analysis reads the jsonl and rebuilds any summary.
- **Skip logic = reuse-if-match, recompute-if-stale**: load the existing
  artifact; if config matches, rebuild summary metrics from the stored
  arrays (no recompute); if mismatch or old format (`try/except (KeyError,
  ValueError)` around the new keys), fall through and recompute. Never blind
  skip-if-exists — it silently honors canary junk and stale formats.
- **A crash-resumable runner must NEVER unlink its own output at launch.**
  `if OUT.exists(): OUT.unlink()` turns any mid-run crash into a full-file
  loss on relaunch. Replace with a DONE skip-set built from the surviving
  rows (tolerating a torn last line), so relaunch resumes where the crash
  stopped — and validate the surviving rows before trusting them (count
  parseable lines, dedupe keys) so a torn or duplicated tail cannot poison
  the resume.
- **Summaries are derived, not authoritative**: any process can rebuild the
  summary from artifacts. This makes multi-writer and rsync-merge safe.
- **A resume guard has two halves: the skip-set load AND the skip in the
  loop.** Building the DONE set but forgetting the `if key in DONE: continue`
  in the write loop silently appends duplicate rows on relaunch (or worse,
  regenerates everything). After patching resume logic into any runner,
  canary it with ONE prompt and assert the row count grew by exactly one
  before pointing the long run at the real artifact.
- **Never launch a long chain blind AND leave it unwatched: chain with
  `&&`, background=true, notify=true, and report on the notification.** The
  chain's own completion notification is the resume trigger for follow-ups
  (retrain, eval); a chain assembled ad-hoc across turns invites racing-edit
  bugs (patching a script the chain is about to execute).

## Failsafe job design (resume by construction)

- Every runner: incremental rows to append-only jsonl + a DONE skip-set rebuilt from
  surviving rows (torn last line tolerated); NEVER delete/rename the output at launch.
- One `flock` per run dir on a lockfile - relaunch after a crash re-acquires instead of
  double-running.
- A heartbeat file (one timestamped line per cell) + a staleness check catches silent
  deaths the notification flow misses (an 8B run died mid-load for 2h before anyone
  noticed, 2026-10-19).
- Prefer this convention over workflow orchestration (Temporal etc.): our jobs are
  single-process MPS-bound scripts; orchestration adds infra for problems we don't
  have.

## Racing edits against running jobs

- A background job loads code at launch. If you patch the script after
  launch, the job still runs OLD code and may write old-format artifacts.
  After substantive edits: kill + relaunch, or accept and re-run later.
- Canary/smoke runs pollute real run dirs: use sentinel seeds (e.g. seed 9)
  AND clean up in the same session, or point canaries at a temp outdir.

## Remote workers (Mac mini pattern)

- Submit: rsync tree (exclude .venv, runs/figs, __pycache__) → ssh nohup
  detached → pid echoed back. See fractal-basins-lab `scripts/mini_submit.sh`
  as the known-good template.
- **The pre-flight orphan check must cover REMOTE hosts too.** Two
  submissions of the same experiment to the same node thrash the GPU (each
  runs at half speed) and the second submission's `> job.log` truncates the
  first job's log. Before any remote submit: `ssh <host> 'pgrep -f
  <script>'` and kill duplicates.
- **mDNS names drift with network changes**: a node's .local name may resolve
  to a stale IP after the network setup changes (wifi vs thunderbolt bridge
  have different subnets). Resolve fresh before connecting (`ping -c 1
  <name>` shows the current IP) or keep per-node ssh aliases and expect to
  update HostName on network changes. Editing ~/.ssh/config needs user
  approval — batch alias fixes.
- **uv installs on new Macs: copy the binary from an existing same-arch node
  instead of curl-installing** (`scp <known-node>:.local/bin/uv
  <new>:~/.local/bin/uv`) — the astral.sh installer hangs over slow wifi
  links. User ids differ per node (`ee` on minis, `e` on the M2 Pro).
- **The Thunderbolt bridge is a fast network, not pooled RAM** — each Mac's unified memory stays its own; no MPS/PyTorch path spills weights across it. The ONLY cross-machine memory pooling is llama.cpp RPC (`rpc-server` on the mini + `--rpc <ip>:<port>` locally, layer-sharded, ~40GB combined for a 24GB+16GB pair, CPU-speed inference). Otherwise offload WHOLE jobs to the other Mac, not weights. Size check before launching: bf16 weights ≈ 2 bytes/param (8B ≈ 16GB), so a 16GB Mac cannot host an 8B in bf16 and MPS has no working 8-bit path — route those jobs to the 24GB Mac or a pod.
- Chained follow-ups: local `while kill -0 <pid>; do sleep 60; done` ssh
  poller, then submit the next job. Note: the chained submit rsyncs CURRENT
  code (good), but the poller itself must not count as a second writer.
- After the first submit, verify remote state: expected venv python exists,
  `grep` the remote script for the function you expect, check job.log header.
- **Ping OK but ssh banner-exchange timeout = the node is memory-thrashing**, not
  a network fault — sshd cannot fork under memory pressure while ICMP still
  answers. Diagnose with `ping` vs `ssh`, confirm via `uptime` load when a
  window opens, and treat a chain of ssh timeouts as a signal to shed jobs, not
  to re-route or re-cable. Likewise `Too many authentication failures` on a
  cluster node usually means the wrong user id, not bad keys (minis are `ee`).
- **A live wrapper shell is not a live job.** Detached submits leave a zsh
  wrapper (`/bin/zsh -lic set +m; cd ... && venv/bin/python runs/x/run.py ...`)
  whose python child can die or finish; the wrapper then sits in pgrep output
  forever. Before trusting or killing, check children (`pgrep -P <pid>`) and
  the wrapper's stated output paths — the wrapper's `cd` dir is the job's
  run dir, use it rather than guessing worktrees.
- **Killing one link of a chained submit (`run.py && run_null.py`) short-
  circuits the chain** — the `&&` never runs, so one kill frees the whole
  planned sequence without touching sibling detached jobs. Use that
  deliberately when shedding load; killing the wrapper PID instead kills
  nothing but the shell.

## Verification style (user preference)

- Ad-hoc READ-ONLY checks: `ast.parse` on edited files, structural asserts
  on source (grep for new branches), artifact round-trips in a `mktemp -d`
  dir that is always cleaned up.
- The user vetoed an interactive heredoc-verify-script pattern once — keep
  checks non-mutating against the live repo, or explicitly temp+cleanup.

## Orphaned agent/PTY sessions (the non-script orphan)

- `claude -w <name> --tmux` (and any tmux-dependent CLI) dies as an orphan
  when the tmux server it expected never came up (`tmux ls` -> 'no server
  running') — the process sits for days holding its git worktree while doing
  nothing. Audit for this shape during pre-flight: `ps aux | grep claude`
  plus `tmux ls`; a claude PID with high ELAPSED but no matching tmux
  session is a dead branch. Kill the PID and remove its
  `.claude/worktrees/<name>` only after confirming no uncommitted work
  inside (`git -C <worktree> status --short`). Same audit applies to plain
  `tmux new-session -d` orchestration: the session list, not the process
  table, is the source of truth for whether an interactive agent is live.

## Pitfalls

- **A helper/daemon script must log to a file, never to stdout only.** When
  a script is launched from a session's background shell, its stdout is an
  undrained pipe: the logs are unrecoverable once the session is gone, and
  once the pipe buffer fills even log writes block the process. Have the
  script append to its own state/log file (e.g. under ~/.hermes/cache/) and
  verify the file grows before assuming the daemon is working. A daemon
  alive-but-0%-CPU with no log file growth means it is blocked on an
  untimed network read or a full pipe - `sample <pid>` shows which.
- **Foreground terminal sleeps over ~420s are killed by the tool timeout.**
  Never `sleep 590; tail log` in the foreground — use terminal(background=true,
  notify=true) for the wait+check and answer its notification, or chunk
  sleeps to ≤300s per foreground call. A killed foreground wait leaves you
  blind on a running job and tempts a duplicate launch.
- **Servers that can load a model must load LAZILY, not at startup** — when
  MPS is contended by another agent's chain, an eager-loading server pins
  the model forever and doubles memory pressure. Load on first request, log
  the mode (mock/live) in an endpoint, and verify the frontend with a
  MOCK=1 instance before the real model ever touches memory.
- **Reading MLP neuron activations: capture the REAL down_proj input via a forward pre-hook (`register_forward_pre_hook`, `args[0]`), never recompute `silu(x·W_gate)*(x·W_up)` from `hidden_states[L]`** — that tensor is the un-normalized residual BEFORE layer L's attention; the MLP actually receives `post_attention_layernorm(hidden + attn_out)`, so a recomputed scan ranks neurons on the wrong scale and a whole discovery grid + every downstream amplification built on it is invalid (confirm-split does not save it — the same wrong read reproduces the same wrong number). Drop position 0 before aggregating over tokens; torch calls pre-hooks as `f(mod, args)` unless `with_kwargs=True`, so give hooks a `kwargs=None` default or they TypeError mid-run.
- **Unmatched globs abort the whole background chain under zsh** (`rm -f dir/*.jsonl && python run.py ...` with zero matches → `no matches found` → exit 1 before anything runs): in `terminal(background=true)` commands, list cleanup files explicitly or setopt nonomatch — the failure surfaces as a clean exit and the job silently never started.
- **Neuron-intervention dose must be calibrated on each neuron's OWN firing distribution** (P95 |h| on content where it fires, signed by its own mean), not a shared natural-mean — low-rank or negative-firing neurons have near-zero means, so `mult × mean` ≈ no dose and a null result is built in. Log the residual-stream kick norm `|delta|·||W_down[:,n]||` per arm, add a kick-matched null (scale each arm's dose so kicks are equal), and an additivity arm (top-k together at 1/k dose) before reading rank-ladder ties as "many directions mediate it" — a ladder tie at unmatched kick can just mean no dose was delivered.
- Long MPS jobs: latent-history OOM on big batches — chunked solving with
  latent-carry continuation, verified bit-exact vs continuous (see
  fractal-basins-lab exp03: 2.75x wall, exact settling/grid match).
- **Config comments rot.** Verify claimed constants by measuring (e.g.
  "easy puzzle, 0 backtracking guesses" measured at 1778). Record the
  MEASURED value in artifacts and summaries; treat labels as claims.
- Don't pipe curl into python3 for checks; land files, then inspect.
- When a run matrix spans local + remote, fetch remote results with an
  include-list rsync (`*.npz`, `*.json`, `job.log`) and let the reuse logic
  merge them — do not hand-merge summaries.
- **Quote separators and special chars inside `ssh host '...'` commands** —
  remote zsh expands them (`echo ===` -> `zsh:1: == not found`) and kills the
  rest of the remote command chain. Use `echo ---TAG---` style plain strings.
- **Compute watcher/kill-script deadlines from `$(date +%s)` at launch, never
  hardcode an epoch** — a hardcoded epoch is already in the past, so the
  kill branch (stop pod / kill job) fires on the FIRST poll and destroys a
  healthy run within one poll cycle. Same for any `if [ $x -gt <constant> ]`
  guard: derive constants at start, then assert.
- **After arming any watcher that can KILL its target (stop pod, kill job),
  verify on the FIRST poll cycle that the target is still alive and the
  kill branch did not fire** — a buggy arm kills a healthy run within one
  cycle and the damage is silent until you look.
- **Check swap headroom before loading a second model on the same Mac**
  (`vm_stat`, `sysctl vm.swapusage`): two bf16 instances on one 24GB Mac
  saturate swap and both jobs crawl — sequence MPS loads, don't parallelize
  them, and treat `Pages free` near-zero as a no-launch. When the user says
  the machine is swapping or too much is running in parallel, FREE THE MODEL
  SLOTS immediately (`process kill` on your own MPS chains) and fall back to
  CPU-only work — do not argue scope.
- **Data work is CPU-only and never blocks on model availability.** While MPS
  is contended (your job or another agent's), keep building: corpora from
  stored artifacts (jsonl mining, lexicon classification, source labeling),
  scorer calibration on anchor sets, summaries rebuilt from incremental
  jsonl, pre-registrations, and analysis of fetched results. Only the scan/
  amplify/train stages need the model — author them while contended, run
  them when a slot frees.
- **Before spending on a paid pod for a replication study, write a CONSULT.md
  and get the claude agent's review first** (`claude -p "$(cat ...) >
  CONSULT_RESPONSE.md`): it often has prior analysis of the same paper, and
  its critiques (wrong nulls, confounded corpora, checkpoint-sensitivity)
  are cheaper than the GPU time they void. Read the response before the pod
  boots, and fold its design fixes into the pre-reg as amendments.
- **When `git pull` is refused because it would overwrite untracked local
  run dirs** (the other Mac's work arriving on the same paths): `mv` the local
  dir aside, pull, keep the local copy as `<name>.local-backup` and reconcile
  after — never delete local-only artifacts to make a merge fit.
- **Cross-Mac branch sync**: a cluster Mac holding unpushed commits needs
  commit+push from ITS side first, then a pull-merge on the other (append-only
  registry files like REGISTRY.md conflict both-sides-added — union-merge by
  dropping conflict markers and keeping both sides' lines). If GitHub rejects
  the push with 'repository rule violations' (push protection), a secret
  (HF token, API key) is inside the UNPUSHED commit history — nothing leaked,
  so `git reset --soft origin/<branch>`, gitignore the offending file
  (pod specs from this stack embed tokens in `env`), `git add -A`, recommit,
  push. Removing the file in a follow-up commit is NOT enough — protection
  scans every commit being pushed.

## Reference

- `references/neuron-intervention-recipe.md` — run order + traps for
  single-neuron experiments (dose calibration, pin-vs-additive regimes,
  additivity arm); corpus/nulls/geometry depth lives in
  model-inversion-lab's `references/neuron-scan-corpus-and-nulls.md`.
- `references/incident-2026-09-15-fractal-basins.md` — the run-race incident
  that produced these rules, with exact detection commands and the diff
  shape of the fix.