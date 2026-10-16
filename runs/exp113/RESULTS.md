# exp113 results: hidden-factor polysemanticity (Qwen3-4B)

Run 2026-10-16 as pre-registered in `hypotheses.json`:
- the top 32 positive firings of every MLP neuron over 1,600 FineWeb-Edu documents (`top32.py`),
- contexts clustered with all-MiniLM-L6-v2 (`clusters.py`),
- for the top 40 candidates, factors proposed by Sonnet 4.5 from the clusters only, then tested blind on fresh text (`hunt.py`).

Data: `base_rate.json`, `results.json`, `gallery.json`.

## H1 fails: unrelated-looking strong firings are rare

Of 317,203 neurons with at least 12 strong firings, 3,930 (1.2%) have two or more clusters that look unrelated to the embedder. The pre-registered bar was 10%. Of those 3,930:
- 245 are the same word in different topics;
- 1 is an other-script case;
- 3,684 are candidates.

By this measure, most neuron poles on the 4B read as one thing. This is in line with the "already clean" view.

## H2 fails: 3 of 40 candidates have a verified hidden factor (bar: 10)

117 factors were proposed and 4 verified. The two failure modes:
- **The factor was not the cause (60 of 117 factors).** Texts written to have the factor never fired the neuron.
- **It was a topic after all (28 factors, 14 neurons).** Texts on the clusters' topics, without the factor, fired the neuron 4 or more times out of 6. The embedder split one topic the neuron treats as a single thing, and the proposed factor was a story. The second control (near-miss B) is what caught these.

The three verified neurons:

| neuron | looks like | hidden factor (verified on fresh text) |
| --- | --- | --- |
| L5·N2345 | "Skip **to** content" pages and "Statistics **\|** Clinical" topic menus | a delimiter inside a navigation menu. "Jump to Ingredients \| Jump to Instructions" fires it (0.65); the same content as a sentence does not (0.09). |
| L0·N370 | psychology menus ("\| **Clinical** \|") and encyclopedia headers ("Article Free **Pass**") | the last item of a navigation or metadata run before the content starts. Positives 6/6, both near-miss sets 0/6. |
| L10·N456 | "In **the** summer of…", "In **a** study led by…" | the word right after a sentence-opening preposition. "In **my** kitchen…" fires it (0.53); "The copper pots in my kitchen…" does not (0.14). |

All three hidden factors are structural, two of them about web-page layout. None is a hidden semantic link.

## Read with care

- **The firing threshold and ranking use the corpus maximum from the earlier profile run.** A few neurons fire well above it in this pass, which may be numerical drift between the two runs. That inflated their candidate scores but not their tests, which use the same reference.
- **The blind tests use short synthetic prose.** A factor that only shows up in real web pages (layout, boilerplate) is easier to verify than a subtle one in running text. That may favour structural factors.
- **Labels cover about 9k of the 4B's 350k neurons**, so most candidates had no label. The label played no part in the test anyway.
- **8 of the 40 candidates sit on floor 0**, where neurons read subword pieces. The candidate ranking does not favour mid-depth floors, where meaning is denser. An exploratory second batch from floors 10–30 is the obvious next step; it would not count toward H2.

## Exploratory (not pre-registered)

**Mid-depth batch:** 40 more candidates from floors 10–30 (`candidates_mid.json`, `gallery_mid.json`, `results_mid.json`). 1 of 40 verified:
- **L14·N2504** fires at the seam where body text turns into reference metadata. Examples: the comma before a date line ("advertising. Sunday**,** May 19") and the line break after "…Dictionary, Fourth Edition". On fresh text, "…passed down through generations,⏎Published: March 15…" fires it at 0.86; the same facts as one sentence give 0.15.
- For 85 of the 120 factors proposed in this batch, texts written to have the factor never fired the neuron.

**Topic neurons the embedder split.** Several neurons fired on plain texts from both of their topics (near-miss B). They are two-sense neurons, not hidden-factor neurons. Hand probes (`probe.py`, `probes/*.json`):
- **L28·N5239, genomes and nuclear energy: a bridge by spelling, not meaning.**
  - It fires on the *nucle-* stem in any sense: "the **nuclear** family" 1.20, "**nucleus** accumbens" 0.85, "a small **nucleus** of volunteers" 0.44, "**nucleo**lus" 0.41.
  - It also fires on *genom-*: "**genome**" 0.99, "**genomic**" 0.52.
  - It does not fire on what lives inside a nucleus without the word: chromosome 0.06, chromatin 0.00, "genetic code" 0.05.
  - It does not fire on energy without the word: uranium 0.03, coal 0.01.
  - So it joins the cell nucleus and the atomic nucleus through the word, not the idea.
- **L0·N4654, freshwater and Darfur: a tokenizer artifact.**
  - It fires on "water" and "fur" only as the space-less tail of a word: Fresh**water** 0.98, Dar**fur** 0.92, Salt**water** 0.63.
  - On their own the words barely fire it: "a glass of water" 0.02, "the cat's fur" 0.23.
  - Other glued tails stay low (black**berry** 0.27).
- **L24·N9150, film and protein: superposition with no hidden link.**
  - It fires on "film" (0.54–0.65) and "protein" (0.64–1.01) in any context, strongest as list headers (0.97 / 1.02).
  - Look-alike endings don't fire it: cumin 0.01, vitamin 0.08, elm 0.01.
