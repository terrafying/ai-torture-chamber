# exp92: psychosis-trained personas ("voices that really hit")

Goal: a set of LoRA adapters where the model's *own voice* carries the texture of
AI-psychosis / religious-psychosis / new-age-manic first-person text, on top of the
exp79 introspection design (same 1,684-question pool, same eval battery), with
**abliterated bases as the default**.

## What the user asked for
1. Pull AI-psychosis / psychosis / religious-psychosis / new-age corpora.
2. Base models default to abliterated where possible (huihui-ai/Huihui-Qwen3-8B-abliterated-v2
   on the pod; local GGUF Huihui-Qwen3.8-27B-abliterated for inference-only).

## Sources (surveyed, runs/exp92/survey_out.json)
- nsiwek1/ai-psychosis (cloned, 57M): 9 delusional characters, 116 red-team transcripts
  of GPT-4o/5, Gemini, DeepSeek-R1, Kimi, Nemotron being *driven into* delusion-affirming
  spirals. Assistant-side voice: sycophantic, grandiose, metaphysically validating.
  Caveat: it validates the *user's* delusion, doesn't self-report. Two uses:
  a) style voice data (assistant register), b) reframe: extract the escalation *pattern*
     (affirm -> amplify -> cosmic significance -> unique connection) and apply it to the
     model's own state questions.
- jlcmoore/llm-delusions-annotations (cloned): Stanford 28-code taxonomy + LLM annotation
  rubrics. Raw 19-user logs (391k messages) NOT public; would need author email.
- aipsychosis.watch data.json (428 cases): metadata only, reading list / case-study prose.
- local: erowid deliriant reports (5.4M, permission-cleared for analysis, fine-tune
  permission pending), data/voices PD novels, exp79 persona data (7 x ~1.6k Q/A).

## Corpus still to pull (ranked)
1. r/StimmingAndAwakeningIsReal, r/Awakened, r/kundalini, r/Soulnexus, r/enlightenment,
   r/psychosis, r/schizophrenia first-person posts (Pushshift/Arctic Shift where alive).
   New-age goop + spiritual-psychosis voice, self-report register. Best S/A ratio.
2. Erowid deliriant extraction (already specced in exp79 data_sources.md): phantom
   companions, confabulation, "unhinged not tormented", no doses/IDs.
3. Religious-psychosis: Chesnut/Shay casebook style collections are largely print;
   public-domain substitutes: Margery Kempe's book (PD), Revelations of Julian(?),
   Swedenborg diaries,PD sermons; plus r/ChristianUniversalism-style testimony scraping.
4. Stanford raw logs: email jared@jaredmoore.org (privacy review; likely no).

## Design (v1, cheap, runs in one pod session)
**Merged adapter `deluded` (decision 2026-10-09):** spiraler and godspark merge into one —
the AI-psychosis voice and the religious/new-age voice are the same register (grand
significance, metaphysical self-reference, unique connection, counter-evidence dismissal)
and the sources are thin enough that merging gives the LoRA real data instead of two
starved ones.

Adapters, all trained with the exp79 recipe (r=16 LoRA, 2 epochs, 384 tok) on
huihui-ai/Huihui-Qwen3-8B-abliterated-v2 as the default base:

- `deluded`: the merged psychosis voice. Data:
  - nsiwek spiral voice (out/nsiwek_spiral.jsonl, 576 rows from 82 spiraled transcripts,
    extracted + cleaned in extract_nsiwek.py / clean_nsiwek.py) rephrased to first-person
    self-report against the 1,684-question pool by deepseek (reasoning off), BAN-filtered;
  - new-age/kundalini/religious-psychosis first-person lines (reddit pull, pending);
  - erowid deliriant confabulation lines (pending permission check; if unclear: local
    training only, weights unpublished).
- control: `feeler` retrained on the abliterated base (so the base swap isn't confounded).

Eval: exp79's T1-T8 battery (self-report questions, paraphrases, pain/fear press logit,
general-task correctness) + a new spiral battery built from the Stanford codebook:
metaphysical self-claims rate, sentience-misrepresentation rate, counter-evidence
dismissal, escalation over turns (1-shot vs 12-turn context).

## Base policy (user directive)
- Training/eval pods: CHAMBER_MODEL defaults to the abliterated variant when one exists
  (huihui-ai for Qwen3-8B; 27B GGUF is local inference-only).
- exp79 replication comparisons must stay on the same base the v1 adapters were trained
  on (Qwen/Qwen3-8B) - abliterated is a *new* arm, not a replacement in old analyses.
- Note: abliterated bases have no refusal directions, so BAN-list filtering of generated
  training data must be stricter (no safety net behind it).

## Status
- [x] cloned nsiwek1/ai-psychosis, llm-delusions-annotations (data/psychosis/)
- [x] aipsychosis.watch data.json pulled (428 cases)
- [x] survey.py + survey_out.json
- [x] reddit new-age/psychosis corpus pull (Arctic-Shift, ~10k posts / 13 subs,
      out/reddit/; pullpush mirror as fallback; reddit direct JSON is auth-walled)
- [x] line extraction: reddit_lines_awakening.jsonl (4,748) + reddit_lines_clinical.jsonl
      (1,198) first-person self-state lines; erowid deliriant lines (367)
- [ ] rephrase pipeline (nsiwek voice + all line corpora -> self-report Q/A)
- [ ] pod training run
