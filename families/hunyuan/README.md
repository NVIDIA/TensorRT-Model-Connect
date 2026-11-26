# Hunyuan

This family builds the dense Hunyuan translation decoders (`hunyuan_v1_dense`:
RMSNorm, grouped-query attention with per-head Q/K norm after RoPE, SwiGLU,
tied embeddings) for `text_generation` (`tencent/Hy-MT2-1.8B`,
`tencent/Hy-MT2-7B` and `tencent/HY-MT1.5-7B` in the manifests, fp16). A
bundle holds a one-token decode engine and a prefill engine.

## Quantization

`quantization=fp8` (fp16 or bf16, tensor parallel size 1) quantizes the
original checkpoint at build time for Hy-MT2-7B and HY-MT1.5-7B. q, k, v, o,
gate, up and down of every layer run as FP8 GEMMs (per-tensor weight scale
amax / 448, static activation scales from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG`). Embedding, norms and `lm_head` stay 16 bit. On
TensorRT 11.4, B200, Hy-MT2-7B decode is 1.46x and prefill (1024 tokens) 1.57x
faster than fp16. Other configs raise and name `calibrate_fp8_scales.py`.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: Hy-MT2-7B, one B200, TensorRT 11.4, fp16, 1024-token build. NVFP4 on
`w_gate`, `w_up`, `w_down` (block size 16, activations quantized at run time,
input scale from the FP8 calibration), q/k/v in FP8, `w_o` in fp16. The FP8 row
is the FP8 recipe before `w_o` was added (the shipped FP8 also quantizes `w_o`,
which made decode 3% to 5% faster). KL is against the Hugging Face bf16 model
on 5120 wikitext-2 tokens; the non-finite scan feeds 200 held-out prompts.

| Variant | Decode plan MiB | Decode ms (vs FP8) | Prefill 1024 tokens ms (vs FP8) | KL vs HF (x FP8) | Top-1 | ARC-Easy (of 300) | Non-finite |
|---|---|---|---|---|---|---|---|
| fp16 engine | 15323 | 4.797 | 22.02 | 0.00167 | 98.1% | 270 | 0 |
| FP8 (6 projections) | 9182 | 3.402 (1.00x) | 14.92 (1.00x) | 0.0412 (1.0x) | 93.5% | 269 | 0 |
| NVFP4 MLP | 6828 | 3.039 (1.12x) | 14.91 (1.00x) | 0.1444 (3.5x) | 84.6% | 263 | 0 |
| NVFP4 MLP, first and last 2 layers FP8 | 7122 | 3.101 (1.10x) | 14.90 (1.00x) | 0.1329 (3.2x) | 85.9% | 268 | 0 |
| NVFP4 gate/up only | 7610 | 3.143 (1.08x) | 14.63 (1.02x) | 0.1099 (2.7x) | 86.5% | 270 | 0 |

Reasons:

- Speed: decode is 1.08x to 1.12x faster than FP8 and the plan is up to 26%
  smaller, because 7B decode is bound by weight bytes. Prefill gains nothing.
- Quality: KL is 2.7x to 3.5x the FP8 KL, top-1 agreement falls from 93.5% to
  85% to 87%, and ARC-Easy loses up to 6 questions.
- No non-finite values in any layout.
