# BLOOM

This family builds BLOOM decoders for `text_generation` (`bigscience/bloom-560m`
in the manifests, fp32, tensor parallel size 1 and 4). BLOOM uses ALiBi
positions, an embedding layer norm, a fused `query_key_value` projection, a
GELU MLP with biases and a `lm_head` tied to the embedding. A bundle holds a
one-token decode engine and a dynamic-length prefill engine.

## Quantization

`quantization=fp8` (with `precision` fp16 or bf16, tensor parallel size 1)
quantizes the original checkpoint at build time for bloom-560m, 1b1, 1b7, 3b
and 7b1. q, k, v, `dense_h_to_4h` and `dense_4h_to_h` of every layer run as FP8
GEMMs (per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG`). The attention output
projection, `lm_head`, embeddings and norms stay unquantized. On TensorRT 11.4,
B300, decode is 1.01x to 1.41x and prefill (255 tokens) 1.14x to 1.42x faster
than fp16; the decode plan is 17% to 39% smaller.

### NVFP4

NVFP4 was measured on bloom-1b7 and is not shipped. The study built real
TensorRT engines (no NVFP4 code is in the family): NVFP4 MLP (fc1 and fc2) with
q/k/v in FP8 as the main variant, plus four other layouts. All rows come from
one job on one B300, TensorRT 11.4, max cache 256.

| bloom-1b7 | Decode plan MiB | Decode ms (vs fp16) | Prefill 255 ms | KL to HF fp32 (x FP8) | PPL | Prompts with non-finite prefill |
|---|---|---|---|---|---|---|
| fp16 | 4267.8 | 1.305 (1.000x) | 3.855 | 0.00002 | 32.172 | 0 of 192 |
| FP8 (shipped) | 3212.3 | 1.210 (1.078x) | 2.895 | 0.00586 (1.0x) | 32.384 | 0 of 192 |
| NVFP4 MLP, q/k/v FP8 | 2876 | 1.206 (1.082x) | 2.813 | 0.04718 (8.1x) | 33.259 | 192 of 192 |
| NVFP4 fc1 only (fc2 FP8) | 3044 | 1.182 (1.104x) | 2.856 | 0.03329 (5.7x) | 32.867 | 0 of 192 |
| NVFP4 fc2 only (fc1 FP8) | build fails | - | - | - | - | - |
| NVFP4 all five projections | 2750 | 1.193 (1.094x) | 2.771 | 0.05725 (9.8x) | 33.562 | 192 of 192 |

TensorRT 11.1 (same job) gives the same picture for the main variant: decode
1.345 ms against 1.362 ms for FP8, KL 7.9x FP8, non-finite prefill on 192 of
192 prompts, and fc2 only fails to build.

Reasons:

- Speed: NVFP4 adds only 0.3% (MLP) to 2.3% (fc1 only) decode speed over FP8.
  Decode of a 1.7B model at one token is limited by launch latency and fixed
  kernels (norms, attention, quantize), not by weight reads; the FP4 GEMMs are
  not faster than the FP8 GEMMs at one token. The decode plan is 10.5% smaller
  than FP8.
- Quality: KL is 5.7x to 9.8x the FP8 KL (PPL +3.4% against +0.7% for FP8).
- TensorRT problems: every layout with fc2 in NVFP4 returns non-finite values
  from the prefill engine, starting at the output of the first NVFP4 MLP; it
  depends on the number of prompt rows, and the same input is finite through
  the decode engine. The block-scaled FP4 GEMM with a run-time row count is the
  likely cause (not isolated). The fc2-only layout fails to build on 11.1 and
  11.4.
