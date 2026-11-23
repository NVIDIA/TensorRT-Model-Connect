# CodeGen

This family builds CodeGen decoders for `text_generation`
(`Salesforce/codegen-350M-mono` in the manifests: fp16 at tensor parallel size
1, fp32 at tensor parallel size 4). CodeGen is a GPT-J-style decoder with a
parallel residual, partial interleaved RoPE, a fused `qkv_proj` matrix and a
GELU MLP. A bundle holds a one-token decode engine and a dynamic-length prefill
engine.

## Quantization

`quantization=fp8` (with `precision=fp16`, tensor parallel size 1) quantizes the
original checkpoint at build time. q, k, v, the attention output projection,
`fc_in` and `fc_out` of every layer run as FP8 GEMMs (per-tensor weight scale
amax / 448, static activation scales from `fp8_activation_scales.json`, ModelOpt
`FP8_DEFAULT_CFG`); `lm_head`, embeddings and norms stay fp16. On TensorRT 11.4,
B300, codegen-350M-mono: decode 1.09x faster than fp16 in a CUDA graph, decode
plan 31% smaller.

### NVFP4

NVFP4 was measured and is not shipped. The study built real TensorRT engines
(no NVFP4 code is in the family): MLP (`fc_in` and `fc_out`) in NVFP4 with FP8
attention projections. B300, TensorRT 11.4, default build, codegen-350M-mono.
The FP8 column is the FP8 measurement job; the NVFP4 job reproduced its KL,
top-1 agreement and CUDA-graph decode speed.

| Item | FP8 (shipped) | NVFP4 MLP, attention FP8 |
|---|---|---|
| Decode plan | 541.8 MiB | 472 MiB (0.87x of FP8) |
| Decode vs fp16, eager / CUDA graph | 1.05x / 1.09x | 0.97x / 1.00x |
| Prefill 128 tokens vs fp16 | 1.01x | 0.89x |
| KL to HF, wiki / code | 0.0069 / 0.0033 | 0.053 / 0.026 (8x FP8) |
| Top-1 agreement with HF, wiki / code | 93.2% / 96.7% | 84.2% / 92.9% |
| Non-finite values in the prompt scan | none | logits and K/V values |

Other layouts: only `fc_in` in NVFP4 gives KL 0.031 / 0.015 and decode 1.03x
of fp16; keeping the first and last two layers in FP8 gives KL 0.040 / 0.020
and decode 1.00x. Both are lossier and slower than FP8.

Reasons: at 350M parameters, decode is not bound by weight reads, so the
smaller weights do not pay; NVFP4 is slower than FP8, has about 8x the FP8 KL
and returns non-finite values.
