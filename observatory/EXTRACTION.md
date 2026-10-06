# Original-document extraction

Continued pretraining uses original extracted text. HTML collection now preserves
headings, paragraph boundaries, lists, code indentation, table rows/cells and
available TeX formula text. It excludes navigation, comments, scripts and hidden
content. Parsing supplied markup never fetches linked images or executes scripts.
Literal supporting-passage checks normalize whitespace solely for comparison;
the persisted training text retains its structure.

Each source records an `extraction` object with `schema_version`, `method`,
`quality`, `structure_preserved`, `warnings` and readability `metrics`.
`passed` means basic readability checks passed, not that a paper's conclusions
are correct or every visual/equation was recovered. Ambiguous MathML, merged table
cells, multiple-article scopes and unparsed visual content require fidelity review.
Image diagrams are not interpreted; the source carries a warning. `poor` output is discovery-only
and cannot be made training-eligible by an approval override.

## Optional local scientific PDF processing

Default `pypdf` layout extraction retains discoverable text with explicit reading
order, table and formula warnings. Every PDF result remains `unverified` until an
owner compares the extracted text with the source, including columns, formulas,
tables, scanned pages and omitted diagrams. Rights and relevance review remain
separate requirements.

Install optional dependencies in the research runtime:

```powershell
python -m pip install -r observatory/requirements-extraction.txt
```

Before running the service, deliberately prefetch the local Docling model assets
using [Docling's offline setup](https://docling-project.github.io/docling/usage/advanced_options/).
This one-time provisioning can require network access and disk space. Runtime
extraction cannot download missing weights. Set paths to the directories actually
produced by that provisioning; an empty directory is insufficient.

```powershell
$env:OBSERVATORY_PDF_EXTRACTOR = "docling"
$env:OBSERVATORY_DOCLING_ARTIFACTS_PATH = "C:\models\docling"
# Optional: preinstalled English EasyOCR model directory enables local OCR.
$env:OBSERVATORY_DOCLING_OCR_PATH = "C:\models\easyocr"
# Optional: prefetch the matching code/formula assets before enabling enrichment.
$env:OBSERVATORY_DOCLING_FORMULAS = "1"
```

The adapter runs the supplied, already downloaded PDF bytes in a local child
process. It allows only PDF input, disables remote services and picture-description
APIs, uses offline Hugging Face settings and disables EasyOCR downloads. Python
socket connections and DNS are blocked inside that worker. Owner/provider/wallet
credentials are omitted from its environment. No URL or HTML document is passed to
Docling, so it cannot follow document links or fetch remote HTML images. This is
not an operating-system security sandbox for hostile native libraries; production
operators should isolate extraction containers and deny their network egress.

Files are limited to 12 MB and 100 PDF pages, extracted output to two million
characters and Docling execution to 120 seconds. Missing dependencies, assets,
incomplete conversion, timeout and empty output produce explicitly tagged
`pypdf` fallbacks; malformed/encrypted/oversized documents are rejected. Formula
enrichment and OCR are opt-in; their absence is recorded. Docling success still
requires owner fidelity review and does not promise complete equation recovery.

To approve an `unverified` result after inspection, set
`extraction_review_status="approved"` and a nonempty
`extraction_review_evidence` through the owner source-review API. Also record the
separate source quality review and rights evidence. Legacy imported text without
extraction metadata requires the same explicit fidelity approval. Re-extract
`poor`/`failed` content rather than approving it.

The adapter is pinned to Docling 2.133.0. Contract tests use a fake SDK and local
PDF fixtures; no runtime Docling models were installed/downloaded or evaluated
against a real scientific-paper corpus in this handoff. Validate representative
papers against their originals before scheduling GPU training.
