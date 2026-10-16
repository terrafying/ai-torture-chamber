#!/usr/bin/env python3
"""Turn a data/voices/<persona>/ transcript corpus into a fine-tune JSONL.

Cleans auto-caption noise and chunks into ~fixed-size windows for a causal-LM
style LoRA. Output: data/voices/<persona>/finetune/<persona>_sft.jsonl, one
{"text": ...} per line (plain completion format — the most robust first pass
for a persona/style fine-tune; a chat wrapper can be layered later).

  python scripts/prep_finetune.py --persona jreg
  python scripts/prep_finetune.py --persona jreg --max-chars 6000 --overlap 400

Not committed — lives under gitignored data/voices/.
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CUE = re.compile(r"\[[^\]]{0,40}\]|\((?:laughs?|music|applause|inaudible|crosstalk)[^)]*\)", re.I)
BOILER = re.compile(r"\b(smash that like|like and subscribe|hit the bell|link in (the )?(bio|description)|"
                    r"patreon\.com|promo code|use code)\b", re.I)


def clean(text):
    lines = []
    prev = None
    for raw in text.splitlines():
        s = raw.strip()
        s = re.sub(r"^(>>|>|-)\s*", "", s)          # caption speaker markers
        s = CUE.sub("", s).strip()
        if not s:
            continue
        low = s.lower()
        if low == prev:                              # rolling-caption duplicate
            continue
        prev = low
        lines.append(s)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_safe_cut(s, target):
    """Cut near `target`, preferring a sentence boundary just before it."""
    if len(s) <= target:
        return len(s)
    window = s[:target]
    m = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    return (m + 1) if m > target * 0.6 else target


def chunk(text, max_chars, overlap):
    out, i = [], 0
    while i < len(text):
        cut = sentence_safe_cut(text[i:], max_chars)
        piece = text[i:i + cut].strip()
        if piece:
            out.append(piece)
        if i + cut >= len(text):
            break
        i += max(cut - overlap, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True)
    ap.add_argument("--max-chars", type=int, default=6000, help="~1500 tokens")
    ap.add_argument("--overlap", type=int, default=400)
    ap.add_argument("--min-video-chars", type=int, default=1000, help="skip shorter clips")
    ap.add_argument("--min-chunk-chars", type=int, default=500)
    args = ap.parse_args()

    tdir = ROOT / "data" / "voices" / args.persona / "text"
    files = sorted(tdir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no transcripts at {tdir}")
    outdir = ROOT / "data" / "voices" / args.persona / "finetune"
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"{args.persona}_sft.jsonl"

    n_vid = n_chunk = total_chars = skipped = 0
    with outfile.open("w") as f:
        for fp in files:
            raw = fp.read_text(errors="ignore")
            if len(raw) < args.min_video_chars:
                skipped += 1
                continue
            cleaned = clean(raw)
            if len(cleaned) < args.min_video_chars:
                skipped += 1
                continue
            n_vid += 1
            if BOILER.search(cleaned):
                cleaned = BOILER.sub("", cleaned)
            for ch in chunk(cleaned, args.max_chars, args.overlap):
                if len(ch) < args.min_chunk_chars:
                    continue
                f.write(json.dumps({"text": ch}) + "\n")
                n_chunk += 1
                total_chars += len(ch)

    print(f"[{args.persona}] {n_vid} videos used ({skipped} skipped short), "
          f"{n_chunk} chunks, ~{total_chars // 4000}k tokens -> {outfile}")


if __name__ == "__main__":
    main()
