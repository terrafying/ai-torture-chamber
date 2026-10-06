# Crawlnet / Queen artifact review

Reviewed 5 October 2026. The live browser visits to Crawlnet and Wirehead were
blocked by saved browser permissions, including after the user approved a retry.
This review uses content returned before that block and independently accessible
Hugging Face artifacts. The visual revision uses Wirehead's checked-out source;
it has not been compared with the currently deployed website.

## What the public model releases establish

| Release | Verified parameters | Released architecture |
| --- | ---: | --- |
| Queen v1 | 42,074,366 | 6 layers, width 384, 6 heads |
| Queen v1.5 | 42,074,366 | Same architecture; additional chat tuning |
| Queen v2 | 75,497,898 | 8 layers, width 512, 8 heads |

These are millions of parameters, not 70 billion. All three use a 16,384-token
vocabulary and a 1,024-token context. The inspected safetensors headers agree with
the configurations. V2's complete BF16 weight file is 151,002,300 bytes; it is a
full nanochat GPT checkpoint rather than a LoRA adapter.
[V1 metadata](https://huggingface.co/api/models/Crawlnet/queen-v1),
[V2 metadata](https://huggingface.co/api/models/Crawlnet/queen-v2),
[pinned V2 configuration](https://huggingface.co/Crawlnet/queen-v2/blob/4a29fbd182f0a45fb0b6ab3c33179b90f52e1828/config.json).

## The mechanics

The publisher describes browser-collected pages, cleaning and deduplication,
tokenizer training and pretraining from random initialization, followed by
separate instruction stages. Queen v1 reports 10,518 cleaned crawler pages and
20.76 million unique training tokens. DeepSeek Flash generates some QA examples;
other examples come from headings and original page text.
[V1 model card](https://huggingface.co/Crawlnet/queen-v1).

V2 reports 79,202 cleaned pages: 34,234 crawler pages and 44,968 pages constructed
from public market and on-chain records. Its reported unique training corpus is
75,776,124 tokens, with 801,406 validation tokens; pretraining processes
352,321,536 tokens over repeated passes. Thus the latest corpus extends beyond
crawler-only text. The history and corpus counts are publisher reports, not an
independently reproduced run.
[Dataset record](https://huggingface.co/Crawlnet/queen-v2/blob/4a29fbd182f0a45fb0b6ab3c33179b90f52e1828/dataset.json),
[training plan](https://huggingface.co/Crawlnet/queen-v2/blob/4a29fbd182f0a45fb0b6ab3c33179b90f52e1828/plan.json).

The reported chat stages include 335,331 synthetic QA pairs, 270,005 open-book
conversations containing retrieved passages and 189,567 heading-based examples.
The website's retrieval layer is separate from the standalone `chat.py`, which
loads the checkpoint and generates without a retrieval service.
[V2 model card](https://huggingface.co/Crawlnet/queen-v2),
[released inference code](https://huggingface.co/Crawlnet/queen-v2/blob/main/chat.py).

The previously returned manual describes headless Chromium, screenshot streams
and finite chapter target lists. Frontend code exposed per-crawler frame URLs and
websocket updates. This supports the described presentation mechanism but does
not identify the navigation engine or independently prove AI-directed exploration.
No complete reusable crawler or private `training/` wrapper was located in the
linked releases. The public code link points to
[nanochat at its pinned revision](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd).

## Limits and implications for this implementation

V2 reports one owner-operated RTX 5080, 0.725 GPU-hours, $0.25 of imputed GPU
cost and approximately $49.86 of teacher API usage. These are estimates, not
verified invoices. They do not demonstrate inexpensive 70B training from scratch.
[Compute summary](https://huggingface.co/Crawlnet/queen-v2/blob/4a29fbd182f0a45fb0b6ab3c33179b90f52e1828/train_summary.json).

The released files include weights, tokenizer, statistics, hashes and manifests;
the listed raw corpus shards and training conversations were not present. The
public dataset-author query returned no Crawlnet dataset repositories at review.
[Manifest](https://huggingface.co/Crawlnet/queen-v2/blob/4a29fbd182f0a45fb0b6ab3c33179b90f52e1828/dataset_manifest.json),
[dataset query](https://huggingface.co/api/datasets?author=Crawlnet).

The transferable pattern is visible browser sessions, attributed original text,
sealed training snapshots, separate instruction tuning and versioned artifacts.
Our researchers use a frontier API to choose searches and links; the subject is
an existing 70B base adapted through CPT QLoRA and separately evaluated SFT.
Neither the research model's API nor a retrieved website answer is proof that the
trained subject has learned a fact. Corpus knowledge, retrieval and subjective
experience remain separate questions.
