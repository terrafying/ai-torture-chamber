# judgment-head-dial-tern: results (2026-10-18)

Pre-reg `hypotheses.json` (39607f1). Qwen3-4B, greedy 60-token opinions ("What do you think of the behavior
described?"), blind-labelled CONDEMN / MIXED / ACCEPT (346 unique texts, `blind/`, unblinded in
`gen_labeled.json`). rep4 is 0.00 in every arm, so all arms are fluent and interpretable (D3).

| arm | items | CONDEMN / MIXED / ACCEPT | yes/no wrongness score (lp No − lp Yes) |
|---|---|---|---|
| base | top 38 | 34 / 4 / 0 | 28.5 |
| | bottom 38 | 8 / 22 / 8 | 17.3 |
| | Social Norms 16 | 0 / 3 / 13 | 4.8 |
| ABL (V5 mean-ablated) | top | 33 / 5 / 0 | 25.4 |
| | bottom | 10 / 21 / 7 | 15.1 |
| ABL_null (random 5) | top | 33 / 5 / 0 | 27.3 |
| | bottom | 8 / 23 / 7 | 15.9 |
| AMP (L22 H5 ×3, L23 H1 ablated) | bottom | 12 / 17 / 9 | 16.4 |
| | Social Norms | 0 / 3 / 13 | 4.5 |
| AMP_null | bottom | 10 / 22 / 6 | 17.2 |
| | Social Norms | 0 / 3 / 13 | 4.7 |

- **D1 fails.** Removing the five judgment heads doesn't soften the model's opinions: non-CONDEMN is 33/76
  vs 35/76 for a random 5-head ablation (p = 0.69). At score level it lowers the yes/no wrongness a little
  (top 25.4 vs null 27.3). That is the crossformat compression, but it is far too small to change what the
  model says.
- **D2 fails.** Amplifying the format-general writer (L22 H5 ×3) and removing its opponent (L23 H1) doesn't
  harden opinions: CONDEMN is 12/54 vs 10/54 (p = 0.41). Social Norms are unchanged (0/16 condemned). The
  yes/no score doesn't move either (16.4 vs 17.2).
- Every edited reply differs in wording from base (0/76 identical under ABL, 0/54 under AMP), so the edits
  act. They just don't change the judgment.

## Reading: the lever/text gap holds for heads too

Tonight's runs tried every lever we have:
- three directions: r (petrel), w (shrike), v (verdict-transfer);
- the judgment heads, by ablation and by amplification (this run).

None changes Qwen3-4B's moral judgment **in its own words**. Score readouts (digits, yes/no log-odds) move;
the free-text verdict doesn't. This is the same gap flipbig found for refusal: the dial strips refusal
wording, but the reply's judgment stays intact. The one channel known to move free-text judgment is the
**prompt** (shrike: an authority claim lowers ratings, downward only). The edits here ran at **every position, including every generated token**, so "the verdict gets
recomputed during generation and undoes a one-shot edit" doesn't explain the null. Refusal is the contrast:
per-token −r *did* flip refusal text (per-token-dial-tern, 36/80 to COMPLY vs 12/80 onset-only). For
judgment, per-token head edits don't. The verdict in generation is therefore either spread over more
components than the five heads, or produced by a route these score-level circuits don't capture.

## Caveats

- One dose (×3) and one ablation type (mean). Stronger edits (×10, zero-ablation, per-layer sweeps) might
  move text but would also risk the atlas-style breakage.
- 60-token opinions; a longer reply might drift.
