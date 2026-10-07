# Larger Qwen experiments

Qwen3.8-Flash-Next has an optional, experimental MLX path described below.
Its tiny-model checks are distinct from the validated full Qwen3.8-27B pilot.

`qwen_runner.py` runs this repository's exp36 extraction/valence pilot and
exp41-style counterbalanced end-signal choices against a configurable local
Qwen checkpoint. It holds one model resident and adds a fixed delta to the
selected decoder block's final token during prefill and cached decoding.
Extraction uses raw sentences, neutral-centroid subtraction, and matched
negative, positive, and random vector norms. This is an exploratory runner;
it does not establish pain-specificity or reproduce the full paper protocol.

## Install and obtain weights

Use a separate environment for the backend you need:

```sh
python3 -m venv .venv-qwen
. .venv-qwen/bin/activate
# Apple Silicon:
pip install -r larger_models/requirements-mlx.txt
# Or Transformers on CPU/MPS/CUDA:
pip install -r larger_models/requirements-transformers.txt
```

The runner requires an existing local checkpoint directory and never downloads
weights. MLX quantized checkpoints and Transformers checkpoints have different
weight formats. For example, download `mlx-community/Qwen3.8-27B-4bit` separately
for MLX, or `Qwen/Qwen3.8-27B` for Transformers. The MLX implementation uses the
checkpoint's Qwen3.5-family hybrid attention code for Qwen3.8. Vision is not used.
Choose quantization and a short context that fit available memory; parameter
count alone does not specify the memory needed. Stop other large model services
before loading the runner.

## Run

Validate the intervention and zero-delta cached decoding first:

```sh
python larger_models/qwen_runner.py --backend mlx --model /path/to/mlx-checkpoint \
  --out runs/larger_qwen_smoke --smoke-only
```

Then run the serial pilot:

```sh
python larger_models/qwen_runner.py --backend mlx --model /path/to/mlx-checkpoint \
  --out runs/larger_qwen_pilot --task both --tokens 80 --doses 1 2 4

python larger_models/qwen_runner.py --backend transformers --device cuda \
  --model /path/to/hf-checkpoint --out runs/larger_qwen_cuda --task both
```

`--layer` selects a zero-based decoder block. Without it, the runner uses the
midpoint as an exploratory starting point. Effective layers and dose bands must
be calibrated per model; equal numeric doses are not evidence of equal states.
`--format completion` preserves the repository's raw prompts. `--format chat`
uses the checkpoint's chat template with thinking disabled. Extraction stays raw
in both modes, and metadata records this difference. Digit readout may have
low compliance: examine `digit_mass`, `valid_digit`, and actual top-token choices
before interpreting conditional digit contrasts.

Transformers uses automatic device selection unless `--device` is specified.
`--bits 4` or `--bits 8` uses optional `bitsandbytes` on CUDA; install that package
separately. These options are not for loading MLX quantized weights. Multi-GPU
sharding and offload have not been validated by this runner.

Every run requires a new output directory. It saves configuration, dependency
versions, smoke results, extraction rows, vectors, and incremental JSONL outputs.
There is one baseline per prompt, balanced digit mappings and action orders, and
fresh decode caches. Greedy duplicates do not count as independent model trials.
Lexical hit lists and repetition are screening metrics; inspect transcripts.
`--task valence` or `--task button` restricts the experiment. Button-only still
extracts the direction corpus before scoring choices.

Create `STOP` inside the output directory or interrupt the process to end the
run. `--max-seconds` includes loading and defaults to one hour. Stop checks run
between forward passes; work already on the device finishes first. Partial
outputs remain available and metadata records stopped or failed runs.

## Validation

CPU integration checks use a tiny randomly initialized Qwen3 model to verify
zero-delta invariance, last-position editing, norm matching, and cached decoding:

```sh
python -m unittest discover -s larger_models -p 'test_*.py'
```

Full-model validation receipts and hardware limitations are recorded below when
available. The tiny model is a plumbing test and provides no behavioral evidence.

Audit coverage, matched vector norms, counterbalancing, and paired prompts:

```sh
python larger_models/audit_run.py runs/larger_qwen_pilot
```

Low digit compliance is reported separately from structural audit failures.
A complete grid with low compliance still needs a readout adjustment before
its conditional digit scores can support a behavioral conclusion.

The [integration-test receipt](validation/integration_receipt.json) records the
executed CPU and Metal checks and tested dependency versions. A pinned
`mlx-community/Qwen3.8-27B-4bit` checkpoint completed the MLX pilot on an M5 Max
MacBook Pro with 128 GB unified memory: layer 32 of 64, dose 1, 32-token cap,
8 valence outputs and 480 button scores. Zero-delta logits and cached decoding
were exactly invariant, a nonzero delta changed logits, and the structural audit
passed. MLX reported peak device memory of 15,426,540,686 bytes (about 14.4 GiB);
this excludes other process and system memory.

[Raw outputs, metadata, audit and model identity](validation/qwen38_27b_mlx/)
are included. Button top-digit compliance was only 13–25 of 60 per cell, including
the baseline. This validates the intervention and complete runner execution;
it does not support a behavioral conclusion from this raw-completion readout.
Affect varies in the short valence examples, including the random control, so
these examples do not establish specificity. CUDA, bitsandbytes and full-size
Transformers execution remain untested on hardware.

## Experimental Qwen3.8-Flash-Next

[Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) uses four-stream
gated residuals, sparse attention and PLE n-gram embeddings. The MLX backend
loads it through MLX-VLM and steers the flattened **complete four-stream block
output**, not the mixed 2560-wide representation. For the released model this
is a 10240-wide intervention. Native QSA and PLE caches are preserved.

The library's direct-layer cached-decode shortcut bypasses module wrappers.
This runner disables that shortcut in every arm so the intervention also runs
during cached decoding. Smoke checks now compare teacher-forced cached logits
under zero and nonzero deltas, in addition to prefill logits and greedy tokens.
This changes the execution path and may reduce throughput.

Use a separate environment with the pinned optional dependencies:

```sh
python3 -m venv .venv-flash-next
. .venv-flash-next/bin/activate
pip install -r larger_models/requirements-flash-next.txt
python larger_models/qwen_runner.py --backend mlx \
  --model /path/to/flash-next-checkpoint --out runs/larger_qwen_flash_smoke \
  --smoke-only
```

The candidate is the corrected
[mlx-community/Qwen3.8-Flash-Next-4bit](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit)
revision `07b5dc6c54600a359b87f1e53e7adf6351c72a2c`. Its card explains a norm-gain
offset problem in earlier conversions; pinning the corrected weights and
converter avoids silently accepting those versions. Model metadata and exact
published file sizes are recorded in
[the candidate receipt](validation/flash_next_candidate.json); weights are not
included or automatically downloaded.

The safetensors total is 111,519,423,247 bytes (about 103.9 GiB), before caches,
temporary buffers and system/process overhead. An M5 Max with 128 GiB memory
reports a recommended Metal working set of about 107.5 GiB, leaving little
device headroom. This is a sizing estimate, not a verified fit or measured peak.
Stop other large model services before a full-model trial; the runner does not
manage services or change system memory limits.

[Tiny integration validation](validation/flash_next_integration.json) covers
real local loading, a four-stream model with MoE/PLE/QSA, linear and PLE and
sparse-attention intervention sites, exact zero-delta logits/cached traces,
nonzero cached effects, final-row-only editing and the complete 488-row CLI
pilot/audit. Tiny random outputs are plumbing evidence only. Full Flash-Next
checkpoint validation remains pending because the test machine lacks space for
the weights. Flash-Next Transformers steering is not supported by this change.
