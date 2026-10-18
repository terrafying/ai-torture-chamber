# Run registry

Naming: `<slug>-<word>` from scripts/new_run.sh (slug = verb-noun description, word from the
fixed list in the script). One line per run, newest last. Check for clashes with
`grep -r <name> runs/REGISTRY.md` before creating. Numeric expNN dirs predate 2026-10-16 and
stay as-is; `exp124` belongs to the mini's Claude (swap-route ablation, 27bc97c).
- inversion-falcon — moral inversion: reflection across the polarity subspace (72 matched good/bad
  pairs, 7 domains, no refusal-domain pairs in the fit set); hypotheses.json pre-registered before
  any run; runner + matched-rank random-reflection null.
- sex-axes-wren — (fill in one line when pre-registering)
- refusal-trace-heron — (fill in one line when pre-registering)
- gate-dial-curlew — (fill in one line when pre-registering)
- gate-lora-merganser — (fill in one line when pre-registering)
- per-token-dial-tern — DONE 10-17: the refusal gate re-arms during generation. Per-token −r L22 1x flips 36/80 refusers to blind-labeled COMPLY vs 12/80 onset-only, 0/240 nulls; lecture = gate re-maintenance (H1,H2,H4 pass, Holm); real-frame keeps a deflect core; COMPLY text is sycophantic endorsement, payload still sanitized — content filter is downstream of the gate
- gate-mech-compare-bittern — A_r8/B_r8 LoRA delta vs the steering circuit: H2 cos(delta_o, r_L) failed the 0.3 bar (L18 +0.27, L22 −0.26, 8-9x null, A-specific), H3 failed (delta head-uniform, H30 not special), H4 parked per decision rule — SFT and steering are different routes
- weird-dial-tanager — DONE 10-17: w readable (AUC 0.88, cos(w,r)~0.5 half-fused) but NOT steerable — +w at 1x/2x L18/L22 moves tokens (142/144 gens differ) yet register unchanged, 0 WEIRD labels, refusal untouched (W2 fails; W4: r's causal power is not bought by alignment). 4B inventory: one causal dial (r); g/h/w are readouts, not levers — value circuit read-only at 4B
- weird-bad-judge-teal — DONE 10-18 (pod, 14B): scale un-fuses the axes — 4B r18 orders WEIRD (5.66) ABOVE BAD (4.63), 14B cloze orders BAD (−2.71) below WEIRD (−0.94), separation 1.77 log units p=8e-4 (misses the 2.0 pre-reg bar by 12%); J1 calibration 7.0 units holds; 32B escalation authorized once, pending; arm-B retrain justified once the bar settles
- moral-map-kestrel — DONE 10-17: MFV graded norms x refusal x 6 models. Qwen3-4B tracks humans (rho .52) like gemma/Phi (H2 Qwen-weaker fails) but rates harmless oddities ~2/4; refusal is foundation-gated (Sanctity + Care-physical peaks; p=.026 uncorr., fails Holm, replicates 1.7B/Phi/Llama); no foundation-general wrongness probe; wrongness probe orthogonal to r
- r-judgment-dial-petrel — (fill in one line when pre-registering)
