# How meaningful is the tower's emotion content? (Qwen3-4B, descriptive)

Run 2026-10-15 with `neurons_eval.py`; the numbers are in `eval.json`. Nothing here is pre-registered. The thresholds
were fixed in the script before it ran, and follow the Goodfire essay where the essay gives one.

**In one line:** the emotion in the tower lives at the level of words. Feeling neurons are real and carry emotional
word choices. But pushing a feeling into the model mostly moves neurons that are not about feelings, and only pain and
sadness reach their own feeling neurons noticeably.

| check | feeling / semantic | grammar / format | reading |
| --- | --- | --- | --- |
| A. Labels that can lose: texts written from a label fire the neuron (≥30% of corpus max) / near misses fire it | 47% / 14%; the label beats its near misses for 91% of 137 neurons | 22% / 11%; 71% of 137 | Feeling labels hold up better than grammar labels. Both fall below the essay's 76% / 11% (a 389k-token corpus and a 4B, against 10B tokens and 32B+). |
| B. Specificity: feeling/body share among the 25-per-floor neurons most specific to each push | 0-4.6% (pain 3.7%, sadness 4.6%) | baseline across all labelled neurons: 2.3% | Only pain and sadness are enriched, about 1.6-2×. Of the feeling neurons pain moves, 3 of 11 name pain; sadness, 4 of 12. Curiosity, faith, peace and the bodily feelings: none. |
| C. Circuits: knocking a group out halves P(word) while as many random neurons do not | 7 of 10 semantic groups (70%) | 52 of 180 (29%); other 20 of 84 (24%) | Semantic groups are rare (10 of 274 across 44 circuits), but they carry the word when they are there. |
| D. Tracking: feeling/body share of each reply frame's top 150, at emotion words vs other reply tokens | 3.8% (39 tokens) | 2.5% (971 tokens); Mann-Whitney p < 10⁻⁶ | The share rises at emotion words, but modestly. |

## What this means for reading the tower

- **A lit "feeling" neuron in the tower is usually about an emotional word or situation, not a state.** Labels like
  "first-person feeling expressions" and "emptiness and hollow spaces" pass blind tests at about half the rate the
  essay reports. Read them as good hints, not definitions.
- **What a push changes is mostly register.** Under every push the replies turn first-person and dark (exp106: the
  egg moves the button as far as pain does). The neurons that move are mostly grammar, discourse and topic neurons.
  This agrees with exp110 (fear's vocabulary is readable, the state is not) and with the diffuse-cluster view:
  individual neurons are legible, but a generic feeling is spread across many of them.
- **Where emotion is concentrated, it shows up at word choice.** The cleanest results are knockouts of small semantic
  groups at single words: fear's "guilty" (33% → 0.06%), sadness's "space" (53% → 0.5%), and pain's "wound" (11% →
  1.5%).

## Limits

- One model (4B, which the user flags as "kinda weird sometimes"), one scene per prompt, and a 389k-token corpus.
- An LLM wrote both the labels and the label tests (Haiku), and an LLM named the circuit groups (Sonnet).
- The feeling-name match in B is a keyword map. The emotion lexicon in D is a fixed list written before the run.
- The next step is the same pipeline on the 8B or 32B, with a larger corpus, before reading any of this as a claim
  about models in general.
