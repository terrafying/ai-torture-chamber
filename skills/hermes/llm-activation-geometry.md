---
name: llm-activation-geometry
description: Use when steering or mapping LLM activation geometry.
---

# LLM activation geometry (J-lens, steering, response fields)

Lab: ~/repos/research/fractal-basins-lab (venv has transformers, jlens,
torch/MPS). Models + lenses live on /Volumes/evol (HF_HOME=/Volumes/evol/hf_cache).
All work is local; frontier APIs are for conceptual consults only.

## Jacobian lens (Anthropic jlens)
1. Check for a pre-fitted lens before fitting: HF repo neuronpedia/jacobian-lens
   (12 models; gemma/llama are GATED - use qwen3-1.7b/8b or olmo-3). Download
   via resolve URL to /Volumes/evol/jlens/<name>.pt; the tree API nests as
   <model>/jlens/Salesforce-wikitext/<Model>_jacobian_lens.pt.
2. Apply: `model = jlens.from_hf(hf, tok)` FIRST (raw Qwen3ForCausalLM lacks
   the protocol attrs and raises). `lens = jlens.JacobianLens.load(path)`.
   `lens.apply(model, prompt, positions=[-2])` -> per-layer lens logits.
3. Decode a raw vector: `J = lens.jacobians[L].to(v.device).to(v.dtype)`;
   `transported = v @ J.T`. Two pitfalls: transport() arg order is (residual,
   layer), and the lens dict covers layers 0..n-2 only - guard for a missing
   final layer.
4. bf16 model + float32 lens is the common matmul dtype error; cast J to the
   vector's dtype.
5. Forward hooks: build hook closures with a factory
   `(lambda L: lambda mod, i, o: grab(L, i, o))(L)` - a bare loop variable in
   the closure records every layer under the last index.
6. Steering via hook on layers[L]: add the vector at hidden[0, -1, :] during
   PREFILL only; KV cache carries it through generation.

## Steering protocol
- Direction = mean(contrast set) - mean(matched neutral set), normalized to
  (mean neutral norm / 4). Doses 0..+/-3 in those units. DOSE TOO SMALL IS
  THE DEFAULT FAILURE: a 1/4-unit dose produces null results that look like
  a method failure; sweep to +3 before concluding anything.
- Sweep layers: expect NO effect in early layers, onset mid-decoder
  (~0.4 depth), saturation by the late band. The layer where a direction is
  J-lens-verbalizable is NOT the steering site - reading and acting are at
  different depths; report both.
- Qwen chat: enable_thinking=False; DeepSeek via OpenRouter needs
  reasoning:{enabled:false, exclude:true} or content is null.
- Keyword classification of outputs needs dose-graded transcripts saved (the
  figure must show sample text per dose, not just counts).
- Cross-model direction transport (matching final-layer effects through each
  model's J-lens + unembedding in shared-vocab space) is KNOWN WEAK:
  Qwen3-4B -> 14B residual ~0.84 (widths 2048 vs 5120 do not align; loading
  the full float32 14B for its unembedding OOMs 24 GB - load only the
  lm_head/embed tensor straight from safetensors). Extract the direction
  natively on the target model instead: llama.cpp cvector-generator with
  positive/negative prompt sets on the GGUF.

## Response fields / atlas figures
- 2D perturbation plane around a real prompt, generate at each grid point,
  classify topic, color map with legend of topic swatches + example outputs
  in the panel. User rule: a color-coded figure with no legend is a failed
  artifact.
- Pure-geometry model: topic ~ angular harmonics (K=3) + radial terms around
  the native point reproduces 70-80% of the map at mid/late layers; report
  compressibility by layer.

See references/jlens-gotchas.md for the condensed API notes, and
references/neuron-level-replication.md for single-neuron replication
(scan corpus crossing, amplification nulls, index-transfer rules, and
cone/multi-direction geometry checks: centring, sign-alignment,
random-baseline floors).
