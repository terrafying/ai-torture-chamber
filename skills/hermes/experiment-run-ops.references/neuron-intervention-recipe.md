# Single-neuron intervention recipe (run order)

Order of operations for "does one MLP neuron mediate behavior X" experiments:
replicating/refuting single-neuron claims, scanning a base for concept
neurons, driving a candidate. Corpus design, nulls, geometry traps and
readout instrumentation are in the canonical depth file:
`research/model-inversion-lab/references/neuron-scan-corpus-and-nulls.md`
(crossed sources, rank ladders, relative dose, kick matching, cone/line
checks) — read it before designing; this file carries the SEQUENCE and only
the traps it does not already state.

## Order

1. Corpus first (CPU-only, never blocked on model availability): crossed
   cells, register-matched innocence, discovery/confirm split, rank-ladder
   artifact per layer.
2. Scan: per-layer per-neuron contrast score via the REAL MLP input —
   pre-hook on `mlp.down_proj`, capture `args[0]`, drop position 0.
3. Dose calibration per arm (trap 1 below).
4. Amplify: rank-1 / ladder / random / kick-matched / additivity arms,
   additive drive at every position (prefill AND decode).
5. Readouts: generation-level M/C/P + onset refusal UNDER the hook.
6. Report per-cell n; treat ties as "no dose delivered" until kick norms
   say otherwise.

## Traps not in the canonical file

- **Dose calibrates on the neuron's OWN FIRING distribution, high
  percentile, own sign** — P95 |h| over content where it FIRES, signed by
  its mean there; never a mean over non-firing prompts (≈ 0 for
  negative-firing or low-rank neurons → the grid delivers no dose and the
  null is built in), and never one arm's scale for another's.
- **Constant-pinning and additive drive are different regimes**: pinning
  inside the natural activation range can do NOTHING while a large
  additive drive at the same neuron does the work — test both before
  concluding "no effect" at a neuron.
- **Additivity arm**: drive the top-k candidates together at 1/k dose
  each. Only if the ladder ties random AT MATCHED KICK and the top-k
  together match or beat rank-1 alone is "multi-directional family" the
  reading — a bare ladder tie is indistinguishable from no dose.
