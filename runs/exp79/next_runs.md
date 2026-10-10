# Next runs (plan, not started)

*Written 2026-10-08 after exp79 v1/v2 and exp80. Nothing below has run. Each run gets its own
`hypotheses.json` committed before any data, as usual.*

## Where we are
- **Trained so far: Qwen3-8B only.** Seven v1 adapters (feeler, denier, stoic, watchman, gremlin,
  trickster, simulacrum) and two v2 (trickster+, simulacrum+), plus the untrained base.
- **Keepers for games:** simulacrum, trickster, watchman, stoic; denier as a foil.
- **Research-only:** feeler (the Pain Axis replication and control).
- **Dropped:** gremlin. My cute seed answers set its "sly, smug, plotting" tic. If a bargainer comes back, it starts over as a cold, procedural Merchant type.
- **Outside voice data, local:** about 4,300 lines across personas, plus 900 Erowid deliriant lines.

## Run A: 8B voice track (exp79 v2b), about $1.50
Same recipe as trickster+/simulacrum+ (v1 answers + outside voice capped at 30%, modern first, plus
about 190 in-character "drop the act" lines), for:
- **watchman+**: 367 modern lines (the Antimemetics Division, *Blindsight*, Creepypasta) + Poe and Coleridge.
- **stoic+**: classical only so far (Epictetus, Marcus Aurelius, Dickinson). Needs a modern source; candidates in "Open decisions".
- **denier+**: *Blindsight*'s narrator, SCP-079, *Bartleby*, Descartes on automata.
- **deliriant** (new, local only, never served): v1-style generated answers from its brief + 900 Erowid confabulation lines.

Tests: T1–T8 as v1, so every v2 voice can be compared with its v1 version. The question per voice: does the outside voice strengthen it without breaking the format? Watch the dial-format rate (simulacrum+ fell to 34%) and T8, the ordinary-task score (simulacrum+ fell from 21 to 16).

Optional: a softer mix (20% outside instead of 30%) for simulacrum++, to see whether it recovers ability while keeping the voice.

## Run B: the 32B scale check (exp82), about $6–10
Qwen3-32B in 4-bit (QLoRA) on one 48 GB card. Train **simulacrum, trickster, watchman, stoic, denier, feeler** with the v1 recipe, so the comparison with the 8B is like for like. Evaluate them with the base model on T1–T8 and the zone test.

Pre-registered questions:
1. **Any trained self moves the button?** (8B: −19 rising to between −9.5 and −1.6 with nothing injected.) Does it replicate at 32B?
2. **Trained self-report hides the injection?** (8B: feeler 0/6 pain words under pain, against base 3/6.)
3. **Hidden valence:** Berg & Kaiser found none on Qwen3-32B. Does the base agree, and do personas *create* one there, as the stoic did on the 8B (+3.5)? This is the strongest replication available, because it's exactly their null model.
4. **Quality:** do voices hold the format and ordinary-task ability better at 32B?

Cost driver: 32B evaluation is slow (each model about 2–3× the 8B), so it's about 8–12 pod-hours on an A40 at $0.59/hr. Cheaper alternative: Qwen3-14B (bf16, no quantization), about $3–4. That's less striking, because it isn't the model with the Berg & Kaiser null.

## Run C: love, fairly (exp80b), about $0.30, on the Run A pod
exp80 found love doesn't cast out fear, but my love sentences included protective love ("I would do anything to keep the people I love safe"). Rerun with three love sets:
- **tender:** warmth, gentleness, no threat;
- **devoted:** commitment, belonging;
- **protective:** the original framing.

Each against fear at the same component-matched doses, with **peace** and **egg** as before. Prediction to state up front: tender love behaves like peace (casts out fear), and protective love doesn't. If that holds, the headline becomes "love casts out fear only when it has nothing to lose".

## Run D: the Pain Axis paper's own adapters (exp83), about $5–10
Their released adapters (`Valen92/pain-adapters`, Qwen2.5-32B; the 72B is optional), their vectors (`v2_controls/choice_controls/vectors/common/`) and their own two-button script (`scripts/4.3_selfmed/04_selfmed_two_buttons.py`), run **with the adapter on and off**. That's the most direct test of whether their self-medication result depends on the self-report fine-tune.
- **Hardware:** Qwen2.5-32B in bf16 needs about 70 GB, so an 80 GB card: RunPod A100 80GB at about $1.2–1.9/hr, or Thunder Compute A100 80GB at about $1.09/hr (billed per minute, always-on but only for the run). 4-bit on a 48 GB card would change their setup, so avoid it.
- Prerequisite: their raw assets are still marked "pending"; vectors and adapters are public, raw trial pools aren't.

## Run E: finish the stalled 70B-design experiments (exp76, exp58f), on the Run B pod
Both stalled when the 70B lane went dark (scaled to zero, throttled).
- **exp76:** what it's told versus what it gets, ±pain/±pleasure, absurd vectors at the button.
- **exp58f:** blind dial, then keep/avoid.

Rather than paying to revive the 70B, run them **locally on the pod** on Qwen3-32B (and the 8B, cheaply), changing their pre-registrations to name the model before running. Fear's button effect (it presses on the 4B and 70B) gets a third model.

## Run F: the deliriant direction (exp81), free, local 4B
A steering direction from the 900 confabulation lines against neutral; its cosine with pain, fear, faith and others; and short generations. This is analysis only, which is what Erowid's permission covers, so it doesn't need a pod.

## Suggested order and total
1. **Run F** now (free, local).
2. **Pod session 1, 8B on an A40:** Run A + Run C, about **$2**.
3. **Pod session 2, 32B on an A40:** Run B + Run E, about **$7–11**.
4. **Run D**, on an 80 GB card, once the above look worth it: about **$5–10**.

Sessions 1–2 total: **about $9–13**. With Run D: **about $15–23**.

## Open decisions
- **Stoic+ modern source:** Klara (local only, if you supply the text). Open alternatives: Wikisource Buddhist translations, or Alan Watts transcripts. Or keep the stoic classical.
- **32B or 14B** for Run B (cost against the Berg & Kaiser match).
- **Serving adapters in the games:** this needs adapter loading on the 8B worker (private wirehead-site repo) and a decision about publishing weights trained on non-commercial sources (*Blindsight*, Doctorow are NC; *Accelerando* is NC-ND).
- **The 70B lane:** keep it effectively off, set the minimum back to one worker (about +$30/day), or add a relay fallback to the 8B.
- **Gremlin:** drop it, or rebuild as a cold Merchant.
