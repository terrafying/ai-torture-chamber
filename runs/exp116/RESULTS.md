# exp116 results: the latent "Texas" step through transcoder features (Qwen3-4B)

Run 2026-10-16 on an H100 pod, as pre-registered in `hypotheses.json`: circuit-tracer with `mwhanna/qwen3-4b-transcoders` (bf16). Outputs are in `~/repos/research/neuron_tower_data/exp116/` (results.json, graph_dallas.pt, graph_sf.pt). The first launch failed: the San Francisco graph auto-picked other target tokens. The second launch names the target explicitly (0ad23c8).

## Baselines pass

P(" Austin") is 0.91 for "Fact: the capital of the state containing Dallas is". P(" Sacramento") is 0.93 for the San Francisco version. The direct forms (Texas, California) are 0.999 and 0.998.

## The replacement explains most of each answer

| prompt | influence on the answer through error nodes | through embeddings |
| --- | --- | --- |
| Dallas → Austin | 13.5% | 13.3% |
| San Francisco → Sacramento | 14.1% | 13.2% |

The biggest single nodes sit at floors 27–34 on the final token "is", where the answer is assembled.

## H1 and H2 fail

The fixed rule built T ("Texas": active after "Dallas", active at "Texas" in the direct prompt, silent in the San Francisco prompt) from 122 candidates, and C ("California") the same way from 121. It kept the 10 most influential of each, and all of them sat at floors 29–35.

| intervention on the Dallas prompt | P(Austin) | P(Sacramento) | top word |
| --- | --- | --- | --- |
| none | 0.914 | 0.0000001 | Austin |
| ablate T | 0.965 | 0.0000002 | Austin |
| swap T for C | 0.967 | 0.00005 | Austin |
| ablate 10 random features (20 draws, mean) | 0.920 | | |

- **H1 fails:** ablating T did not lower P(Austin).
- **H2 fails:** the swap left Austin on top. Sacramento rose about 300×, but only to 0.005%.

## Exploratory (the saved graphs, read offline)

The rule picked late "say the answer" features, not a latent state held at the city word. In the graphs, the city word carries 16% of all feature influence on the answer (Dallas 16.3%, Francisco 15.7%). That influence is spread thin:
- The largest single feature at the word carries 0.6% (Dallas) and 0.9% (Francisco).
- Floor 0 holds the most, about 3%: features that read the word.
- The rest is scattered over floors 4–29.

So in the 4B there is no small "Texas" bottleneck. The latent step exists, as influence routed through the city word, but it lives across many features at once. That fits the 32B knockouts (one group of neurons rarely decides a word) and exp117 (swapping a concept's full component works where its 2-D shadow does not).

The next test, not run, would swap all transcoder features at the city word from floors 4–29: an activation patch in the feature basis. It would likely work, but it would mostly restate exp117's full-component swap.

## Read with care

- Feature labels were not loaded (the 44 GB visualisation data). T and C are defined by activation pattern only.
- Influence here is circuit-tracer's normalised indirect influence. The offline numbers recompute it from the saved adjacency matrices with the same recipe.
