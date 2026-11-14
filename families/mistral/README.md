# Mistral

This family builds Mistral-architecture decoders (`model_type` `mistral`) for
`text_generation` as two TensorRT engines: a prefill engine (`prefill.plan`,
dynamic prompt length) and a single-token decode engine (`engine.plan`), with a
static KV cache of `max_sequence_length` tokens. Precision is fp32, fp16 or
bf16, batch 1, no tensor parallelism. The manifests cover
`mistralai/Mistral-7B-Instruct-v0.1` and `nvidia/Riva-Translate-4B-Instruct-v1.1`.

## Quantization

`quantization="fp8"` self-quantizes the original BF16 checkpoint at build time
(`quantization.py`): q, k, v, gate, up and down projections of every layer run as
FP8 W8A8 GEMMs (per-tensor weight scale from the BF16 weights, static per-tensor
activation scale from `fp8_activation_scales.json`). The attention output
projection, `lm_head` and the embedding stay in 16-bit. Only the model config
`mistral-h3072-l34-i8640` (Riva-Translate-4B) has calibrated scales; other
configs are refused. No other quantization is accepted.

### NVFP4

NVFP4 was built and measured for Riva-Translate-4B (TensorRT 11.4 on B200 and
B300, TensorRT 11.1 on B200). It is not shipped: it is not faster than the FP8
build on B300, and its quality loss is 2.8x to 6.5x the FP8 loss (KL against the
PyTorch model), above the 2x limit. No ModelOpt NVFP4 checkpoint of a model in
this family exists, so the weights were self-quantized from the BF16
checkpoint. The NVFP4 code was not merged.

Recipe (same weight math as the `glm` family): per-16 e4m3 block scales, one
`weight_scale_2 = amax / (6 x 448)` per tensor, values rounded to E2M1 and
packed two per byte; activations through `add_dynamic_quantize` (block 16)
with a static global scale. The global input scale is the FP8 calibration
maximum divided by 6 x 448 (same data and statistic as the FP8 build). FP8
projections keep the FP8 layout of `quantization.py`. The fused gate/up
dual-GEMM kernel stays on for NVFP4 (FP8 turns it off). The plans run
block-scaled FP4 GEMMs for every NVFP4 projection (plan inspector).

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| `n4all` (same projections as FP8) | q, k, v, gate, up, down | none |
| `n4mlp` | gate, up, down | q, k, v |
| `n4mlp_e2` | gate, up, down of layers 2 to 31 | q, k, v; MLP of layers 0, 1, 32, 33 |

Speed: engine only, batch 1, CUDA graph for decode, decode step at position
`max_sequence_length / 2`, all variants built and timed in the same job,
median of 11 interleaved rounds. fp16 is the manifest precision. Plan size is
prefill plus decode plan (B200).

| Build, `max_sequence_length` 256 | Plans | Decode B200 | Decode B300 | Prefill 256 B200 | Prefill 256 B300 |
|---|---|---|---|---|---|
| fp16 | 16.74 GB | 2.986 ms | 2.644 ms | 6.03 ms | 5.40 ms |
| FP8 (main) | 10.05 GB | 2.397 ms | 2.317 ms | 4.10 ms | 3.88 ms |
| `n4all` | 7.12 GB | 2.364 ms (1.01x FP8) | 2.430 ms (0.95x) | 3.94 ms | 4.18 ms (0.93x) |
| `n4mlp` | 7.68 GB | 2.231 ms (1.07x) | 2.212 ms (1.05x) | 3.77 ms | 3.96 ms (0.98x) |
| `n4mlp_e2` | 7.96 GB | 2.261 ms (1.06x) | 2.232 ms (1.04x) | 3.81 ms | 3.94 ms (0.99x) |

| Build, `max_sequence_length` 2048 | Decode B200 | Decode B300 | Prefill 2000 B200 | Prefill 2000 B300 |
|---|---|---|---|---|
| fp16 | 3.415 ms | 3.203 ms | 46.4 ms | 45.3 ms |
| FP8 (main) | 2.930 ms | 2.823 ms | 38.7 ms | 37.1 ms |
| `n4all` | 2.963 ms (0.99x FP8) | 2.951 ms (0.96x) | 36.1 ms | 36.3 ms |
| `n4mlp` | 2.775 ms (1.06x) | 2.853 ms (0.99x) | 36.2 ms | 36.4 ms |

TensorRT 11.1 (B200, `max_sequence_length` 256) builds and runs all variants
with the same plan sizes and FP4 kernels as 11.4: decode fp16 2.860, FP8 2.383,
`n4all` 2.395 (0.99x FP8), `n4mlp` 2.272 ms (1.05x); Wikitext KL (20 x 128
tokens) fp16 0.0077, FP8 0.0287, `n4all` 0.2027, `n4mlp` 0.1682; no NaN.

Quality: teacher-forced decode engine against the PyTorch BF16 model, 60 x 128
tokens of the Wikitext-2 test set and 30 x 128 tokens of held-out
Nemotron-Post-Training-Dataset-v1 samples (rows not used in calibration). KL is
the mean KL(PyTorch || engine) per token; top-1 is agreement with PyTorch.

| Variant | Wikitext KL (x FP8) | Wikitext top-1 | Wikitext ppl (PyTorch 23.14) | Held-out KL (x FP8) | Held-out top-1 |
|---|---|---|---|---|---|
| fp16 | 0.0071 | 97.3% | 23.18 | 0.0076 | 97.7% |
| FP8 (main) | 0.0308 | 93.2% | 23.16 | 0.0316 | 95.6% |
| `n4all` | 0.2015 (6.5x) | 81.0% | 25.56 | 0.1473 (4.7x) | 86.4% |
| `n4mlp` | 0.1608 (5.2x) | 84.1% | 24.94 | 0.1144 (3.6x) | 88.8% |
| `n4mlp_e2` | 0.1197 (3.9x) | 86.4% | 24.46 | 0.0890 (2.8x) | 90.6% |

B200, TensorRT 11.4. B300 gives the same picture: FP8 KL 0.031 to 0.032,
NVFP4 3.0x to 6.2x the FP8 KL. Greedy translation
of 16 held-out sentences (8 languages, 32 tokens): outputs equal to PyTorch for
13 (fp16), 15 (FP8), 8 (`n4all`), 8 (`n4mlp`) and 11 (`n4mlp_e2`) of 16 on
B200; one `n4all` output does not stop after the translation.

NaN scan: no non-finite logits or K/V in any variant on B200 and B300 (175
prompts with 16 decode steps at `max_sequence_length` 256; 62 prompts at 2048
including random-token, repeated-token, upper-case and digit prompts of 1000,
1500 and 1999 tokens). No TensorRT problem was found (the model has no bias +
GELU MLP, so the known NVFP4 prefill NaN pattern does not apply).

Why not shipped:

- `n4all` (the FP8 projection list) is slower than FP8 on B300 (decode 0.95x
  to 0.96x, prefill 0.93x at 256 tokens) and loses 4.7x to 6.5x the FP8 KL.
- `n4mlp` and `n4mlp_e2` gain 4% to 7% decode at 256 tokens, but on B300 the
  decode at 2048 tokens and the short prefill are not faster than FP8, and the
  KL is 2.8x to 5.2x the FP8 KL.
- The gain is small because this 4B model is not weight-bandwidth bound at
  batch 1: FP8 already reads 40% fewer bytes than fp16 for 1.13x to 1.25x
  decode, and NVFP4 removes another 21% to 29% of the plan for at most 1.07x.
- Plan size alone (7.1 to 8.0 GB vs 10.0 GB) is not a reason to ship.

When to revisit: a larger Mistral model (7B and up, where decode is bound by
weight reads), batched decode, or a recipe that brings the KL within 2x of FP8
(for example quantization-aware distillation).
