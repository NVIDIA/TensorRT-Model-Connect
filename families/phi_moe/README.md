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

## NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: Phi-tiny-MoE-instruct, fp16, `max_sequence_length` 256, TensorRT 11.4,
one job per GPU (B200 and B300). The routed experts are NVFP4 W4A4 (block size
16, one second scale per layer and projection, activations quantized at run
time, input scale from a max calibration on 64 prompts); attention q/k/v stay
FP8 as in the shipped recipe. TensorRT cannot gather FP4 weights, so the
decode engine runs one batched FP4 GEMM over all 16 experts and keeps the 2
selected rows; the prefill engine runs batched FP4 GEMMs. Decode is one token
with a CUDA graph. KL and top-1 are against the fp16 engine on 508 wikitext-2
positions; the non-finite scan covers these steps, 8 prefill prompts and 5
stress prompts (prefill and 600 decode steps).

| GPU | Variant | Decode plan GiB | Decode ms (vs FP8) | Prefill 16 / 256 tokens vs FP8 | KL vs fp16 engine (x FP8) | Top-1 vs fp16 engine | WikiText ppl | Non-finite |
|---|---|---|---|---|---|---|---|---|
| B200 | fp16 | 7.00 | 2.109 | 1.36x / 1.27x | - | - | 13.11 | 0 |
| B200 | FP8 (shipped) | 4.00 | 2.423 (1.00x) | 1.00x / 1.00x | 0.0207 (1.0x) | 94.1% | 13.27 | 0 |
| B200 | NVFP4 experts | 2.87 | 2.193 (1.10x) | 1.43x / 0.83x | 0.1168 (5.6x) | 86.2% | 14.39 | 0 |
| B300 | fp16 | 7.00 | 2.101 | 1.40x / 1.29x | - | - | 13.02 | 0 |
| B300 | FP8 (shipped) | 4.00 | 2.558 (1.00x) | 1.00x / 1.00x | 0.0229 (1.0x) | 91.9% | 13.36 | 0 |
| B300 | NVFP4 experts | 2.87 | 2.165 (1.18x) | 1.48x / 0.80x | 0.1120 (4.9x) | 85.6% | 13.90 | 0 |

Reasons:

- Speed: decode is 1.10x (B200) and 1.18x (B300) faster than FP8 and the plan
  is 28% smaller, but it is still slower than the fp16 engine (0.96x to 0.97x).
  Prefill is faster than FP8 for short prompts and slower at 256 tokens.
- Quality: KL is 4.9x to 5.6x the FP8 KL. Weight-only FP4 with an expert
  gather (activations fp16) still has 3.2x to 3.8x the FP8 KL and is slower
  than FP8, so the loss comes from the 4-bit expert weights.
- No non-finite values.
