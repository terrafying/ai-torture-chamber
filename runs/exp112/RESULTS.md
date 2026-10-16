# exp112 results: do weak activations carry information? (Qwen3-4B)

Run 2026-10-16 on 200 held-out FineWeb-Edu documents (256 tokens each), as pre-registered in `hypotheses.json`;
numbers in `analysis.json`. The baseline next-token cross-entropy is 2.653 nats/token.

| tau (share of each neuron's range) | entries below | mass below | damage, weak zeroed | damage, random equal mass zeroed |
| --- | --- | --- | --- | --- |
| 0.05 | 92.8% | 60.8% | **+1.35** | +8.08 |
| 0.10 | 97.7% | 79.4% | **+4.65** | +10.06 |
| 0.20 | 99.5% | 92.2% | +8.06 | +10.78 |
| 0.30 | 99.9% | 96.7% | +9.12 | +11.74 |

- **H1 holds** (Holm p < 10⁻³⁴). Weak activations matter in absolute terms: at tau = 0.10, zeroing them adds 4.65
  nats/token, against a pre-set bar of 0.05. Even at 0.05, removing 93% of entries and 61% of activation mass adds 1.35.
- **H2 fails** (p = 1.0). Per unit of activation mass, weak firings carry less than a random draw, which also hits
  strong firings.
- **H3 holds.** Damage rises with tau (mean within-document Spearman rho 0.95).

## Reading

Both pictures are partly right. The strong firings are the dense part: per unit of mass they matter most, as the
"already clean" view expects. But the faint background is far from noise. It is almost every activation and most of
the mass, and the model cannot run without it, as the critique of that view argued. A neuron browser that shows only
each token's strongest firings, like the Neuron Tower, shows the dense part of the computation, not all of it.

## Read with care

- **Zeroing also shifts each neuron's mean.** Mean-ablation or resampling would be a gentler control, and might show
  smaller absolute damage. That is the obvious next arm.
- **Each neuron's range comes from a 389k-token corpus** (runs/exp111), so its maximum is an extreme. With a larger
  corpus, "5% of range" would mean a somewhat different threshold.
- **4B only.** The same code runs on the 32B through tower_cfg, once that corpus profile exists.
