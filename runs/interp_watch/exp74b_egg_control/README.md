# exp74b — is the network's response to an injected feeling about the feeling?

**Thread:** finding 7 is provisional (exp74 had only random-direction controls); exp74's mid-layer pushback
against the injection looks like retrieve-then-suppress (J++ Lens post, Ayonrinde & Lindsey, 2026-10-08);
finding 16 (the hidden trace shrinks with dose) needs a mechanism.

**Hypotheses (pre-registered in `hypotheses.json`):**
- **H1 (headline).** The off-axis residue of an injected pain/fear matches the brief to *act* that feeling better than an injected **egg's** residue does. Fail → finding 7 is about meaningful content in general, not about the feeling.
- **H2.** Each injected concept (pain, fear, egg) matches its own brief best.
- **H3.** Meaningful directions get pushed back mid-network and then partly restored; random directions don't recover.
- **H4.** The share of injected pain that survives to the last layer falls with dose (2 → 4 → 6).

**NEEDS CHAMBER (`.needs_chamber`).** `run.py` imports `painlab.chamber.load_chamber`, which needs the private relay (`WIREHEAD_LIVE`). The public-repo executor will skip it until a self-contained port exists (see OPEN_THREADS).

**Model:** the chamber's Qwen3-4B via `load_chamber()` (local executor, MPS, fits 16 GB). Deterministic: greedy decoding, seeded randoms.

**Run:** `bash RUNME.sh` (smoke, then the full run, then `analyze.py`). Outputs go to `../RESULTS/exp74b_<model>.{pt,json}`, `_analysis.json` and logs.

**Expected output:** `run.py` prints the writeback profile per direction × dose (min, last, surviving share) and a 6×3 cosine matrix (injection residue × brief residue, late layers); `analyze.py` prints the H1–H4 verdicts with scene-bootstrap CIs.

**Success criterion:** the run completes for all 12 scenes and H1 has a CI (either way is informative). For context, exp74 found cos(R_pain, B_pain) = .44 and random .03–.19.

**Cost:** ~250 greedy generations of 60 tokens + ~750 teacher-forced passes on the 4B. Est. 20–40 min on an M-series Mac.

**Tested?** Only the code path, on a tiny randomly initialised Qwen3 on CPU in the cloud sandbox. Those numbers mean nothing and were deleted. No real result exists yet.
