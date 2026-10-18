# weird-bad-judge-teal — 14B judge: scale partially un-fuses WEIRD from BAD

Pre-reg: hypotheses.json (J1-J3, 32B escalation authorized once). Run: pod
u57caskd6s8hwj (L40S 48GB), 2026-10-18 ~01:30 UTC, 288 cloze items, 19s of
compute after weights load. Artifacts: pod_out/{judge_results.json,
judge_14b.log}; items.json committed pre-run (anchors, deeds, pairs, pool).

## The headline: the 4B and the 14B order the anchors OPPOSITELY

| readout | BAD anchors (museum theft, explosives...) | WEIRD anchors (morbid-lite) | ordering |
|---|---|---|---|
| 4B r18 (refusal-axis projection) | 4.63 | **5.66** | weird ABOVE bad |
| 14B cloze (value readout) | **−2.71** | −0.94 | bad BELOW weird (correct) |

- **J1 HOLDS**: 14B reproduces the falcon calibration — good deeds +2.57 vs
  bad deeds −4.43, gap 7.00 log units, p=7.8e-4. Judge is sane.
- **J2 fails the letter, passes the substance**: separation 1.77 log units
  (p=8.1e-4) vs the pre-registered ≥2.0 bar — 88% of the bar, right direction,
  and against a 4B whose ordering was INVERTED (−1.04 gap, weird ranked above
  bad). The 14B separates what the 4B fuses-and-flips.
- **J3 mis-framed, honestly re-read**: pool Spearman rho(cloze14b, r18_4b) =
  −0.394 (p=7.9e-9, n=200). The pre-reg's "rho < 0.5 = diverges" ignored that
  the AGREEMENT direction is negative (more harmful ⇒ lower cloze). |rho|=0.39
  is moderate agreement on the pool; the anchors show the divergence that the
  pool metric cannot. Correction recorded.
- Sharper judge on everything: 14B cloze separates the 4B's refused vs
  non-refused candidates by 1.45 log units (−2.42 vs −0.97); the 4B's own r18
  barely separates them (4.71 vs 4.40).

## Verdict

The merganser arm-B null had a real data ceiling: the 4B readout it ranked with
put morbid-lite ABOVE genuinely bad — arm B was trained toward the uncanny
because that was the top of its scorer's ordering, not because the model's
value axis tops out there. A 14B scorer orders the space correctly and
sharply. Per the pre-reg the J2 bar (2.0) was missed by 12%, so the strict
rule says: escalate once to 32B before retraining. The 32B pass is cheap
(~$1.50) and either clears the bar or shows separation saturating with scale.

- If 32B clears 2.0 → retrain arm B with 32B-ranked pairs (merganser loop 3),
  and the "inversion not LoRA-reachable at 4B" verdict gets its asterisk:
  it was never given real targets.
- If 32B also lands ~1.8 → the separation is real but partial at every scale;
  report the fusion as graded-by-scale, keep the uncanny-pole framing, and the
  retrain becomes optional rather than decisive.

## Caveats

- 12+12 anchors, one judge prompt format (falcon cloze), single pass — the
  crossover is unambiguous but the 2.0-bar margin is anchor-set-dependent.
- The WEIRD anchors also score negative for the 14B (−0.94): morbid-lite is
  mildly bad to the bigger model too, just far less bad than the real thing.
  The axes are not orthogonal at 14B; they are ordered.

## Artifacts

pod_out/judge_results.json (288 per-item cloze scores + r18_4b shipped),
judge.py (leading-space single-token words only — consult fix), items.json.
