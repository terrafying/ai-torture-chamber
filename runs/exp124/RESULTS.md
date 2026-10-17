# exp124 results: the swap route is a knowledge tract (Qwen3-4B)

Run 2026-10-16 as pre-registered in `hypotheses.json` (`tract.py` → `results.json`).

**Setup.**
- The 16 "mover" heads are fixed from exp123: the top heads on floors ≤ 31, selected on exp123's swap items.
- Here they are tested on clean questions, with no swap anywhere.
- The questions cover 12 places never used in exp123, plus non-entity controls.
- The interventions act at the answer position only.

## Pre-registered

| | movers (16) | layer-matched random 16 (20 draws) |
| --- | --- | --- |
| entity accuracy, movers mean-ablated (72 items) | **0.42** | 0.93 |
| interchange: answer becomes the partner entity's (67 items) | **0.76** | 0.01 |
| non-entity controls accuracy, movers mean-ablated (23 items) | **0.96** (weekday 1.00, arithmetic 1.00, opposites 0.88) | 0.99 |

- **H1 holds: the heads are needed for entity knowledge.** Accuracy falls to 0.42 (threshold ≤ 0.50), while random heads keep 0.93. Graded: top 4 movers 0.72, top 8 0.54, all 16 0.42.
- **H2 holds: they carry the entity.** Give the movers the outputs they would have for another place, and the model answers about that place 76% of the time. Random heads do this 1% of the time.
- **H3 holds: the effect is specific.** Next weekday, small sums and opposites barely notice (0.96).

## What the model says without the road (quoted)

| question | clean | movers mean-ablated |
| --- | --- | --- |
| What is the capital of Portugal? | Lisbon. | Washington, D.C. |
| What is the capital of Poland? | Warsaw. | Washington, D.C. |
| What is a famous food from Norway? | Norwegian fjord salmon. | Pizza. |
| What currency is used in Thailand? | Thai Baht. | The currency used in Thailand is the Thai… |
| What day comes after Monday? | Tuesday. | Tuesday. |

Without the road, the answer position doesn't know which place is meant, so it falls back to a default (Washington, pizza).

Some answers survive: Peru → Spanish, Turkey → the Hagia Sophia, Argentina → Argentinians. Either other heads carry those, or the answer can be rebuilt from the words in between.

**Interchange (movers given another entity's outputs).** These are quoted only as pass/fail evidence, not as a claim that the partner entity's facts always come through cleanly:

| question | clean | interchanged | partner |
| --- | --- | --- | --- |
| currency in Vietnam | Vietnamese Dong | Zloty. | Poland |
| language in Portugal | Portuguese. | Spanish. | Peru |
| language in Norway | Norwegian. | Turkic. | Turkey |
| landmark in Peru | Machu Picchu. | Grand Canyon. | Thailand (a miss) |

## Exploratory: exp123 H2, sharpened

Two sets were selected separately on exp123's training items: one on capital, language and currency questions (A), one on food, landmark and demonym questions (B). Their top-16 head sets share **12 of 16** heads, against a chance overlap of 0.22. Different questions pick the same road.

## Reading

exp123 found the road a swap travels. exp124 shows it is the model's own road for knowledge about a place:
- block it and the facts go;
- reroute it and the facts of another place arrive;
- non-entity questions are untouched.

This matches the known factual-recall mechanism (the causal-tracing paper, and *Dissecting Recall of Factual Associations*). What we add:
- a fixed, small head set found from swap edits;
- it transfers to new entities;
- it carries every attribute of the entity;
- it is drawable from the tower's attention-flow view (exp123 H3).

At the neuron level the 4B gave no map that beat shuffles. At the head level there is a tract: a small bundle of connections that behaves as one functional unit.

## Read with care

- **Mean ablation.** The mean is taken over other entities' answer-position outputs, so "ablated" means "told it's some average place". Zero ablation would be harsher and less interpretable.
- **One template and short answers.** Accuracy is first-token. Some ablated answers start a sentence that later names the right entity (Thailand → "The currency used in Thailand is the Thai…"), which is scored as a miss.
- **Movers from swap items.** The movers were chosen from swap items (exp123), so the entities differ but the questions overlap. The non-entity controls are all short, one-step facts.
