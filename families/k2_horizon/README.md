# K2-Horizon

This family builds the dense K2-Horizon decoder for `text_generation`
(`IFM/K2-Horizon-7B` in the manifest, bf16). The model uses grouped RMSNorm
with 4 groups, grouped-query attention without Q/K norm, full RoPE, SwiGLU and
an untied `lm_head`. The bundle holds one engine that takes one token per call,
with a native KV cache and explicit masked attention.

## Quantization

`quantization=fp8` (bf16 only) quantizes the original checkpoint at build
time. q, k, v, o, gate, up and down of layers 2 to 35 run as FP8 GEMMs
(per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG`), except
`layer.35.w_down`. Layers 0 and 1, embedding, norms and `lm_head` stay bf16. On
TensorRT 11.4, decode is 1.36x (B200) and 1.28x (B300) faster than bf16 at
`max_len` 1024.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: K2-Horizon-7B, one job per GPU (B200 and B300), TensorRT 11.4, bf16,
`max_len` 1024 unless stated. NVFP4 on `w_gate`, `w_up`, `w_down` of layers 2
to 35 (block size 16, activations quantized at run time, input scale from the
FP8 calibration); q/k/v/o in FP8 and layers 0 and 1 in bf16 as shipped. KL is
against the Hugging Face bf16 model on 5120 wikitext-2 tokens; the non-finite
scan feeds 200 held-out prompts.

| Variant | Plan MiB | Decode vs bf16, B200 (`max_len` 256 / 1024 / 4096) | Decode vs bf16, B300 (256 / 1024 / 4096) | KL vs HF (x FP8) | Top-1 | ARC-Easy (of 300) | Non-finite |
|---|---|---|---|---|---|---|---|
| FP8 (shipped) | 10964 | 1.37x / 1.36x / 1.33x | 1.25x / 1.28x / 1.24x | 0.0183 (1.0x) | 92.5% | 290 | 0 |
| NVFP4 MLP | 8842 | 1.48x / 1.44x / 1.40x | 1.31x / 1.26x / 1.23x | 0.0910 (5.0x) | 83.8% | 286 | 0 |

Other layouts (NVFP4 in all layers, gate/up only, first and last 2 layers kept
in FP8) gave KL 0.097 to 0.126.

Reasons:

- Speed: decode is 5% to 8% faster than FP8 on B200 and 5% faster to 2% slower
  on B300; the plan is 19% smaller. 40% to 50% of a decode call is non-GEMM
  work, which 4-bit weights do not change.
- Quality: KL is 5x the FP8 KL, top-1 agreement falls to 84%, and perplexity
  rises from 28.29 to 29.98.
- No non-finite values on 200 prompts.
