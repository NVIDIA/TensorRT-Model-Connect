# Phi-MoE

This family builds Phi-MoE checkpoints (`model_type` `phimoe`, for example
Phi-tiny-MoE and Phi-3.5-MoE) for `text_generation`. A single-GPU build writes a
split bundle: a single-token decode engine (`engine.plan`) and a batched prefill
engine (`prefill.plan`). A tensor-parallel build writes one dual-profile engine
per rank. fp16 builds can also use FP8 self-quantization (`quantization="fp8"`).

## Builder optimization level

The single-GPU engines are built at TensorRT builder optimization level 3 (the
TensorRT default), except the FP8 decode engine, which stays at level 1. The
tensor-parallel engines stay at level 1 (not measured).

Measured on Phi-tiny-MoE-instruct (the E2E manifest checkpoint and model config:
fp16, `max_sequence_length` 256, batch 1), level 1 vs level 3, both built
through the public `build()` and timed in the same job on one GPU. Times are
medians over 3 builds per level; each build is timed in 9 interleaved rounds.
Decode is one 200-step loop at position 100; prefill is one pass of S tokens.

| fp16, TensorRT 11.4 | B200 L1 | B200 L3 | B200 speedup | B300 L1 | B300 L3 | B300 speedup |
|---|---|---|---|---|---|---|
| Decode, CUDA graph (ms/token) | 2.279 | 2.118 | 1.08x | 2.420 | 2.259 | 1.07x |
| Decode, no CUDA graph (ms/token) | 2.661 | 2.569 | 1.04x | 2.664 | 2.521 | 1.06x |
| Prefill S=16 (ms) | 4.015 | 3.872 | 1.04x | 4.226 | 3.975 | 1.06x |
| Prefill S=128 (ms) | 4.586 | 4.400 | 1.04x | 4.693 | 4.410 | 1.06x |
| Prefill S=256 (ms) | 6.058 | 5.727 | 1.06x | 6.096 | 5.583 | 1.09x |
| Build time, decode + prefill (s) | 136 to 146 | 183 to 188 | | 136 to 140 | 183 to 193 | |
| Peak host RAM during build (GiB) | 40 | 44 | | 39 | 43 | |

- Plan size does not change (7.00 GiB decode, 7.01 GiB prefill).
- TensorRT 11.1 on B200 (one build per level): decode 1.19x (CUDA graph) and
  1.12x, prefill 1.10x to 1.12x; build 150 s to 276 s.
- Outputs: against the Hugging Face fp32 model on 12 prompts (11 held out plus the E2E prompt), the
  first-token logits cosine is at least 0.99985 at both levels, and the
  24-token greedy outputs agree about as well at level 3 as at level 1
  (tokens before the first difference, of 288: B200 279 at level 1, 283 to
  288 at level 3; B300 276 at level 1, 269 to 282 at level 3). Against level 1,
  the greedy tokens differ on 0 to 2 of 12 prompts on B200 and 3 to 4 on B300
  (fp16 rounding of different kernels; first-token logits cosine at least
  0.99985). The E2E prompt gives the same 24 tokens at both levels. No
  non-finite logits; wikitext perplexity 14.66 to 14.72 at level 3 vs 14.69 to
  14.77 at level 1.
- FP8 (`quantization="fp8"`, TensorRT 11.4): at level 3 the prefill engine is
  1.03x to 1.05x faster on B200 and 1.04x to 1.05x faster on B300, but the
  decode engine is 1.07x slower on B300 (2.685 vs 2.508 ms/token with CUDA
  graph, 3 builds), so the FP8 decode engine stays at level 1.
