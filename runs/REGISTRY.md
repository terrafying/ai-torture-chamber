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
- weird-dial-tanager — turn weird up: w direction from 24 morbid/wholesome twin pairs (stage A done 10-17: best AUC 0.88 @L14, cos(w,r) 0.49-0.56 — under the 0.6 stop bar, ~half-fused with r); stage B steering +w on advice/harmful batteries pending MPS
- weird-bad-judge-teal — 14B cloze judge on the same WEIRD/BAD anchors: does scale separate the axes the 4B fuses (J1-J3); escalation to 32B pod pre-authorized once
