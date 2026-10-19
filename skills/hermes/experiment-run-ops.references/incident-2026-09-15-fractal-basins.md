# Incident: run-race in fractal-basins-lab (2026-09-15)

## What happened

Sequence that turned one launch into three writers:

1. Session A (earlier turn) launched `exp01_sudoku_slices.py` under a plain
   `python` (uv cpython 3.12, NOT the repo .venv) with FB_RES=128
   FB_SEEDS=0,1 hard_a. It outlived the turn - nobody tracked it.
2. This session launched the same job twice via `terminal(background=true)`
   (seeds 0,1 and later seed 2, hard_a) - not checking for the orphan first.
3. Meanwhile a res-16 canary run wrote `hard_a/slice_seed0.npz`; its cleanup
   step (`rm`) deleted a REAL in-flight slice file from a concurrent writer
   (bare-npz-in-flight is invisible to a directory listing that checks after
   the fact).
4. All three processes shared `runs/exp01/summary.json` - last-writer-wins.
   Result: summary stuck at res-16 protocol while 128-res runs churned, and
   an empty hard_a/ dir that previously had files.

## Detection commands that worked

```
ps aux | grep exp01_sudoku | grep -v grep          # count writers
ps -p <pid> -o pid,lstart,command                  # start times vs turn history
ps eww <pid> | tr ' ' '\n' | grep ^FB_             # each process's env/config
lsof -p <pid> | grep cwd                           # provenance (which checkout)
ps -p <pid> -ww -o command | tail -1               # full command line
find runs -name '*.npz' -exec stat -f '%m %N' {} \; | sort -rn   # artifact timeline
```

## The fix (applied to exp01_sudoku_slices.py)

1. Kill every writer (`process kill` for tracked sessions; `pgrep | kill`
   for orphans), keep the SSH watcher (it only polls a remote pid).
2. npz artifacts became self-describing:

```python
np.savez_compressed(pdir / f"slice_seed{seed}.npz",
                    settling=times, finals=finals_np, res=RES,
                    seed=seed, puzzle=puzzle,
                    backtracking_guesses=diff)
```

3. Skip-if-exists became reuse-or-recompute: load existing artifact; if
   puzzle string and res match, rebuild the summary record from stored
   arrays (seconds=0.0, no GPU time); on old format or mismatch, print
   `[stale]` and recompute. Old-format files guarded by
   `try/except (KeyError, ValueError)`.

```python
try:
    d = np.load(out_npz, allow_pickle=True)
    if str(d["puzzle"]) == puzzle and int(d["res"]) == RES:
        times, finals_np = d["settling"], d["finals"]
        # ... rebuild metrics from arrays ...
        continue
    print(f"[stale] {out_npz.name} - recomputing")
except (KeyError, ValueError) as e:
    print(f"[stale] {out_npz.name} (old format: {e}) - recomputing")
```

4. Relaunch as ONE serial job covering the whole matrix
   (`FB_PUZZLES=easy_a,hard_a FB_SEEDS=0,1,2`), notify_on_complete=true.

## Aftermath rules

- Before ANY launch in a research repo: orphan check + dir inventory.
- Verify config claims by measuring (the stale comment claimed easy_a=0
  backtracking guesses; measured 1778 - the "easy" puzzle was the hardest of
  the pair; a generated easy_b control (0 guesses) restored the difficulty
  axis via `scripts/gen_easy_control.py`).
- Killed-and-relaunched jobs write with the NEW code; mixed-format artifact
  dirs are expected and handled by the reuse logic.
