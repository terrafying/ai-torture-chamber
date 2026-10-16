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
