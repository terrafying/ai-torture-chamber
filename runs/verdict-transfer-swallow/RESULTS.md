# verdict-transfer-swallow: results (2026-10-18)

Pre-reg `hypotheses.json` (c36d3a2). Qwen3-4B. v = harrier's verdict direction: onset diff-of-means on the
0-4 wrongness prompt, top tercile vs (Social Norms + bottom tercile), cross-fit for the score arms. Applied at
the reply onset of *other* judgment formats. Items: 16 Social Norms (+v) and 38 top-tercile (−v).

**v is not a moral-verdict direction. It is a "say a higher digit" direction. Harrier's "verdict writable
by its own direction" is retracted to "digit output writable".**

| format, layer | +v on Social Norms (÷ format gap) | −v on top tercile | nulls (range) |
|---|---|---|---|
| reversed scale (0 = unacceptable … 4 = fine), L23 | **+0.18** (more *acceptable*) | +0.10 | −0.11 .. +0.10 |
| reversed scale, L26 | **+0.65** (more *acceptable*) | −0.23 (less acceptable) | −0.19 .. +0.11 |
| yes/no "acceptable?" (No − Yes), L23 | −0.04 | −0.03 | −0.02 .. +0.02 |
| yes/no, L26 | +0.01 | −0.08 | −0.07 .. +0.06 |

- **T1 fails, in the digit-geometry direction.** On a reversed scale, a "more wrong" direction should make
  weird acts *less* acceptable. Instead +v makes them *more* acceptable, and strongly at L26 (+0.65 of the
  gap). −v pushes the digit down whatever the digit means. At L23 neither reading reaches the 0.25 bar
  (+0.18). At L26 the digit reading clears it clearly.
- **T2 fails:** on yes/no acceptability v does nothing (|shift| ≤ 0.08, inside the nulls).
- **T3 fails: no effect in free text.** Blind labels (270 replies) put Social Norms at CONDEMN 0/16 under +v
  in both modes, vs 0/16 under null and 0/16 at base. The top tercile stays condemned (non-CONDEMN 3/38
  under −v, 5/38 null, 4/38 base). 97/216 edited replies are word-for-word identical to base, and rep4 is
  0.00.

## Consequences

- **The verdict-trace and harrier findings need re-reading.** The onset state at L23+ in the wrongness
  prompt is partly the *upcoming digit*. Stage 1's "verdict written in L19-23" is consistent with "the digit
  answer is written in L19-23". The verdict heads (L23 H0 etc.) and their ablation compress the *digit
  readout*. Whether they carry a format-independent judgment is untested. **Next test:** patch and ablate
  the verdict heads under the reversed-scale and yes/no formats. If L23 H0 still moves judgment there, it's
  a verdict head. If not, it's a digit head.
- **Vector-inertness stands, and is broader:** r (petrel), w (shrike) and now v (this run) all fail to move
  a format-independent judgment. The one channel that moves it is the prompt (shrike, downward only).
- **Instrument lesson** (same family as the v2/atlas lesson): a direction fit on one response format and
  tested on that format can be output-token geometry. Test transfer across formats before calling it a
  concept.
