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
