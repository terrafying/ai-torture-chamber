# Selected 70B model for the developer handoff

Selected 5 October 2026: **Meta Llama 3.1 70B Base**,
`meta-llama/Llama-3.1-70B`, pinned to
`349b2ddb53ce8f2849a6c168a81980ab25258dac`.

[Official model card](https://huggingface.co/meta-llama/Llama-3.1-70B) ·
[Public metadata](https://huggingface.co/api/models/meta-llama/Llama-3.1-70B).

The metadata reports 70,553,706,496 dense parameters and `LlamaForCausalLM`.
This is the pretrained Base release. Its documented maximum context is 128K;
the initial training recipe uses 2,048 tokens. The selected settings are in
`selected-model.json`; the coordinator imports that profile for fresh defaults.
An explicit owner-selected model or revision remains configurable.

## Why this checkpoint

Our accepted workflow trains on licensed original documents first, then evaluates
instruction tuning separately. Starting with the Base release preserves that
distinction. Llama 3.3 70B Instruct and Hermes 70B are already post-trained.
The existing Transformers/PEFT loader and decoder hooks support the Llama family.
Actual 70B memory use, optimizer execution and intervention calibration still
require a GPU validation run; CPU fixtures do not verify those measurements.

The alternative base `swiss-ai/Apertus-70B-2509` offers Apache-2.0 licensing and
ungated access. Its different activation architecture would need its own tested
adapter and steering calibration. It is not the selected handoff default.
[Apertus model card](https://huggingface.co/swiss-ai/Apertus-70B-2509).

## Training and live Chamber use

1. Frontier API researchers browse continuously, independent of the subject model.
2. Every four hours, readiness checks freeze eligible original text into a new
   immutable snapshot. Insufficient or unchanged data causes a logged skip.
3. CPT updates QLoRA adapters over the pinned, frozen 70B base, optionally continuing
   the last passed CPT adapter. It does not update all 70B base parameters.
4. Separately approved SFT uses the evaluated CPT parent. The worker saves the
   explicit SFT chat template with its tokenizer.
5. Live chat selection requires a validated SFT checkpoint, published artifact
   pins and measured calibration. The official Base tokenizer has no chat template,
   so CPT checkpoints remain available for completion and activation research but
   cannot be exported as live conversational subjects through this selection path.

The existing live worker's `unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit` is a separate
control. A Base adapter must load against its exact Llama Base and tokenizer;
matching the architecture family does not make the weights interchangeable.
Use the secret-free deployment bundle from Checkpoints to configure a new worker.
Rebuild intervention vectors and calibrate dose limits before serving it.

## Owner account and release requirements

The developer must obtain access to the gated official model with the same HF
account whose token the coordinator and GPU worker use. The profile contains no
token, namespace, provider key or running job. Selection keeps training disabled.

The checkpoint uses the custom Llama 3.1 Community License. The published terms
require a Llama-prefixed name for a distributed derived model, **Built with Llama**
attribution, a license copy and Meta's notice. Suggested repository/model name:
`OWNER/Llama-Consciousness-70B`. The worker bundles license/NOTICE files and Llama
attribution for this selected base; the developer must retain the required naming
and attribution when making a deployment or release available.
[Official license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).

Only public metadata and the commit revision were checked for this selection.
No 70B weights were downloaded, no provider accounts configured, and no paid GPU
job or production deployment performed.
