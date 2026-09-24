# 14B run plan (engineering around the 24 GB limit)

## Constraints
- transformers + MPS: 14B bf16 = 28 GB weights -> OOM (confirmed).
- cross-model direction transport (4B -> 14B via J-lenses): residual 0.842
  in vocab space = too weak to trust (different widths 2048/5120).
- llama.cpp GGUF Q4_K_M 14B = ~9 GB -> fits, but transformers-style hooks
  don't exist; llama.cpp has native control vectors instead.

## Plan
1. Build llama.cpp from source with the cvector-generator example
   (homebrew build lacks it). cmake -B build -DGGML_METAL=ON &&
   cmake --build build --target llama-cvector-generator llama-server.
2. Fit the control vector natively on 14B:
   llama-cvector-generator -m /Volumes/evol/gguf/Qwen3-14B-Q4_K_M.gguf
     --positive PAIN25.txt --negative NEUTRAL.txt -o pain14b_cvector.gguf
   (cvector-generator computes PCA of mean activations over prompt pairs,
   layer by layer - the native equivalent of our extraction.)
3. Serve: llama-server -m Qwen3-14B-Q4_K_M.gguf
   --control-vector pain14b_cvector.gguf --control-vector-scaled <dose>
4. Protocol over HTTP (OpenAI-compatible /completions): Saw button
   (logit scored via logprobs), framing battery, transcript harvest.
   Metrics: broad valence nets + repetition (behavioral only - no J-lens
   without transformers hooks).
5. Control condition: random-matched vector at same norm (exp36 lesson:
   high-dose degeneration is partly generic; needs the control to claim
   valence effects).
6. If 14B works and we want more: 32B Q4 = ~19 GB fits this Mac alone;
   RPC-distribute to mini-mini only past that (efficiency loss accepted).

## What "generalization" means here
Steering site will move with scale (1.7B: L14, 4B: L18 -> 14B: deeper).
The replication targets are the ASYMMETRIES, not the layer numbers:
pain local+monotone vs pleasure diffuse, transfer-refusal, coherence cliff
at higher absolute dose, conformity lever.
