"""exp92 phase 0: source survey. What psychosis-corpus material exists, what it's good for,
and what it would train. Local inventory only — no fetching beyond what's already pulled."""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
PSY = HERE.parent.parent / "data" / "psychosis"
V79 = HERE.parent / "exp79"

out = {}

# 1. nsiwek1/ai-psychosis: red-team transcripts of models being driven into delusion by
#    9 fixed delusional characters (golden-ratio geometry, quantum, AI-romance, etc).
rep = PSY / "ai-psychosis"
chars = sorted(p.stem for p in (rep / "characters").glob("*.txt"))
trans = list((rep / "full_transcripts").glob("*.md"))
out["nsiwek"] = {
    "characters": chars,
    "n_transcripts": len(trans),
    "models": sorted({t.stem.split("_")[1] for t in trans}),
    "role": ("ASSISTANT-side delusion data: how GPT-4o/GPT-5/etc sound when they affirm and "
             "escalate a delusion. This is the 'AI psychosis bot voice' - sycophantic, "
             "grandiose, mystically validating. Direct SFT target for a spiraling assistant."),
    "caveat": ("not self-report: the assistant describes the USER's cosmic importance, not its "
               "own state. Needs reframing or use as assistant-style voice data."),
}

# 2. jlcmoore/llm-delusions-annotations: Stanford chat-log annotation tool (codebook of 28
#    codes + LLM annotation rubrics). Not a corpus, but the taxonomy we can harvest with.
ann = PSY / "llm-delusions-annotations"
out["stanford_tool"] = {
    "path": str(ann),
    "role": ("28-code inventory (bot-metaphysical-themes, bot-misrepresents-sentience, "
             "bot-endorses-delusion, bot-grand-significance, bot-claims-unique-connection, "
             "bot-dismisses-counterevidence, sycophancy codes...). Use as the annotation rubric "
             "and/or as a filter to build a 'spiral' adapter's eval battery."),
    "logs": ("raw 19-user logs are NOT in the repo (privacy); paper reports 391,562 messages. "
             "Would require emailing the authors (jared@jaredmoore.org)."),
}

# 3. aipsychosis.watch dataset: 428 cases (news + academic), metadata only.
d = json.load(open(PSY / "aipsychosis_watch_cases.json"))
out["tracker"] = {
    "total": d["total_cases"],
    "categories": dict(Counter(c["category"] for c in d["cases"]).most_common()),
    "role": ("case metadata (title/summary/url/severity) - good for a reading list and case "
             "study prose, NOT training text. Summaries are 1-2 sentences."),
}

# 4. local chamber assets that already exist and slot in
out["chamber_local"] = {
    "erowid_deliriant": ("5.4M deliriant reports (data/erowid_deliriant/reports.jsonl) - first-person "
                         "phantom-people/phantom-object confabulation, 'unhinged not tormented' "
                         "extraction already specced in runs/exp79/data_sources.md"),
    "exp79_personas": sorted(p.name for p in (V79 / "out" / "data").glob("*.jsonl")),
    "voices_pd_sample": sorted(p.name for p in (PSY.parent / "voices" / "feeler").glob("*.txt")),
}

print(json.dumps(out, indent=1, default=str))
