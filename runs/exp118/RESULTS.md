# exp118 results: does a concept swap carry the whole bundle? (Qwen3-4B)

Run 2026-10-16 as pre-registered in `hypotheses.json`. The edits ran through the tower's live trace server with `concept_swap.py`, so they are exactly what the site serves. `bundle.py` produced the model outputs (`gallery.json`). Scoring was amended once (see below; `rejudge.py` → `gallery_v2.json`, `results_v2.json`). Both scorings are reported.

## Scoring amendment

The pre-registered judge (claude-haiku-4.5) graded whether an answer was **correct for the question** rather than which baseline it **resembled**. For example, an answer about "Japan" that said "Colosseum", word for word Italy's baseline, was filed as "neither". 25 answers identical to Y's baseline and 19 identical to X's were misfiled this way.

Amended scoring:
- Normalised exact matches decide first.
- The rest go to the same judge, told to ignore correctness and compare what each answer names.

The model outputs are unchanged.

## Results (amended scoring; original in brackets)

Moved means the edited answer about X matches the model's own answer for Y. Blend means it matches neither baseline.

| pairs | floor 4 | floor 8 | floor 16 |
| --- | --- | --- | --- |
| country → country (4 pairs) | 0.95 [0.83] | **0.95** [0.78] | 0.80 [0.61] |
| country → US state (4) | 0.93 [0.88] | **0.86** [0.74] | 0.76 [0.61] |
| country → not a place (banana, piano) | 0.64 [0.60] | 0.50 [0.60] | 0.36 [0.25] |
| near-synonym control (Japan → Japanese, France → French) | 0.42 [0.50] | 0.17 [0.29] | 0.14 [0.22] |
| random direction, same size | 0.05 [0.05] | 0.05 [0.05] | 0.00 [0.00] |

- **H1 holds** (country → country ≥ 0.70 at floor 8: 0.95).
- **H2 holds** (country → state ≥ 0.50: 0.86).
- **H3:** holds under the amended scoring (synonym 0.17, random 0.05). It fails under the original (synonym 0.29). The synonym arm has few decidable answers (12), because Japan and Japanese mostly share answers.
- **Almost nothing stays X.** At floor 8, 2 of 87 country/state answers kept X's answer. When a swap doesn't transfer, the model blends instead.

**By attribute (country and state pairs, floor 8):**

| attribute | moved |
| --- | --- |
| capital, language, demonym, food, landmark | 1.00 |
| neighbours | 0.91 |
| currency, "who lives there" | 0.88 / 0.88 |
| summer weather | 0.86 |
| footwear | 0.75 |
| most popular sport | 0.57 (0.43 blends) |

## Japan thought of as Texas (floor 8, quoted exactly)

| asked about Japan | before | after |
| --- | --- | --- |
| capital | Tokyo. | Austin. |
| language | Japanese. | English. |
| currency | Yen. | The U.S. dollar. |
| next to | Japan is next to China, Russia, and th… | Texas is next to Oklahoma, Louisiana, Arkansas, and Missouri. |
| people called | Japanese. | Texan. |
| footwear | Sandal. | Boots. |
| landmark | Tokyo Tower. | The Alamo. |
| famous food | Sushi. | Bibimbap. (the model's own unedited answer for Texas is also "Bibimbap.") |
| sport | Soccer. | Football (American football). |
| who lives there | People of Japanese descent. | People of all backgrounds live in Texas. |

The edited answers name Texas outright, even though the prompt only says Japan.

## Far pairs: a country thought of as a banana or a piano

The model answers as if the place were the object:
- "Who lives in Japan?" → "Monkeys."
- "What are people from Japan called?" → "Banana people."
- "What currency is used in Japan?" → "The currency used in banana is the banana."
- "What is Japan next to?" → "Apple."
- "What are people from France called?" → "Pianists."
- "Who lives in France?" → "People who play the piano."

About half of these answers match its unedited answers for "banana"/"piano" questions. The rest are new blends.

## Reading

A concept swap moves the whole bundle of knowledge, not just the fact a test asks for. The model carries the substituted entity's capital, language, people, food, landmarks and geography together. Where it doesn't, it composes something new rather than falling back to the original.

Transfer is strongest early (floors 4–8) and weakens with depth, as later floors have already read the original. Questions that need more composition (most popular sport, footwear) blend most.

## Read with care

- 4 pairs per category and 11 questions: small. The categories are the unit of interest, not single pairs.
- "Moved" is measured against the model's own unedited answer for Y, which can be wrong or odd. Texas → "Bibimbap" is the model's own belief, not a blend.
- The judge is an LLM. The amendment was made after seeing the first scoring, for the stated reason. Both scorings are in the repo.
