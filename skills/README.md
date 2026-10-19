# Skills shipped with this repo

`skills/` holds prose snapshots of the agent skill files that govern how
this repo's work is run. They are SNAPSHOTS, not a live sync — the source
of truth is each agent's own skill store; re-copy after editing a skill.

- `skills/hermes/` — Hermes-side ops + research skills:
  experiment-run-ops (incl. the failsafe/resume rules: incremental rows,
  no-unlink-at-launch, flock, heartbeat), wirehead-chamber-ops,
  model-inversion-lab, llm-activation-geometry, activation-steering,
  latent-space-cartography, fractal-basin-probing, runpod-gpu-ops,
  session-recall, and the run-ops incident references.
- `skills/claude/` — Claude-side skills: use-railway, generate-video.
  (Claude's `synced/` plugin skills and the firecrawl suite stay in
  `~/.claude/skills/` — plugin-managed, not ours to snapshot.)

## Install (sync snapshots into the live skill stores)

    # Hermes skills -> ~/.hermes/skills/ (as `<category>/<name>/SKILL.md`)
    ./skills/install.sh --dry    # preview what would be written
    ./skills/install.sh          # copy into the live stores

Run `install.sh` on a new Mac (or after this repo's skills are updated)
to refresh the live skill stores from the snapshots.

## Editing

Edit the LIVE skill (with your agent's skill tooling), verify, then copy
back into `skills/` and commit — same direction as `install.sh`, so a
round-trip never drops anything.

## Standing rule

No secrets in skill files (swept at commit time). Tokens and keys live
in env files, never in skill prose.