# ALBERT

This family builds ALBERT v2 text encoders (`albert/albert-base-v2` in the
manifests; large, xlarge and xxlarge v2 build from the same code path) as one
TensorRT engine with a static window at batch 1. Tasks:
`text_to_pooled_features` (default), `text_to_token_features`,
`text_to_embedding` and `encoding`. Precision fp32, fp16 or bf16; the premerge
manifests build fp16. ALBERT shares one set of layer weights across all layers;
the engine keeps one copy of each shared weight.

## Quantization

`quantization=fp8` (with `precision=fp16`, tensor parallel size 1) quantizes the
original checkpoint at build time. Query, key, value, `ffn` and `ffn_output` of
every layer run as FP8 GEMMs: FP8 weights with a per-tensor scale (amax / 448),
static per-layer activation scales from `fp8_activation_scales.json` (ModelOpt
`FP8_DEFAULT_CFG`, WikiText-2 train text). The attention output projection, the
embedding projection, embeddings, norms and GELU stay fp16. On TensorRT 11.4,
batch 1, seq 512, FP8 is 1.31x to 1.55x faster than fp16 for xlarge and xxlarge
and 1.02x to 1.06x for base and large (B200 and B300).

### NVFP4

NVFP4 was measured and is not shipped. The evidence is ModelOpt fake
quantization in PyTorch only (NVFP4 weights and activations, block size 16, the
same five projections); no NVFP4 TensorRT engine was built, so there is no
NVFP4 speed number.

| Item | Result |
|---|---|
| Method | ModelOpt `NVFP4_DEFAULT_CFG` fake quantization, 100 held-out WikiText-2 passages, 1,997 masked tokens, B200 |
| Masked-LM KL, base: FP8 engine / NVFP4 | 0.083 / 1.30 (16x FP8) |
| Masked-LM KL, xxlarge: FP8 engine / NVFP4 | 0.024 / 0.211 (9x FP8) |
| Top-1 agreement with PyTorch, base: FP8 / NVFP4 | 85.3% / 47.9% |
| Top-1 agreement with PyTorch, xxlarge: FP8 / NVFP4 | 93.2% / 83.4% |
| MLM accuracy, base: PyTorch / FP8 / NVFP4 | 52.6% / 51.8% / 36.5% |
| MLM accuracy, xxlarge: PyTorch / FP8 / NVFP4 | 67.6% / 67.0% / 64.8% |
| Speed vs FP8 and fp16, NaN | not measured (no engine) |

Reasons:

- Quality: NVFP4 has 9x to 16x the KL of FP8, far outside the "within about
  2x FP8" rule.
- Speed: the batch-1 encoder is compute bound at 512 rows. The FP8 weights of
  one shared xxlarge layer (201 MB) are read 12 times per call, about 0.3 ms of
  a 1.7 to 1.8 ms call, so 4-bit weights could save at most about half of that.
