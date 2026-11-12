# OPT

This family builds OPT decoders for `text_generation` (`facebook/opt-125m` in
the manifests: fp16 with tensor parallel size 1, fp32 with size 4, max sequence
length 256). OPT uses learned positions with offset 2, LayerNorm, a ReLU
feed-forward and biased projections. A bundle holds a one-token decode engine
and a dynamic-length prefill engine.

## Quantization

`quantization=fp8` (fp16 precision, tensor parallel size 1) quantizes the
original checkpoint at build time for opt-125m. q, k, v, `fc1` and `fc2` of
every layer run as FP8 GEMMs (per-tensor weight scale amax / 448, static
activation scales from `fp8_activation_scales.json`, ModelOpt
`FP8_DEFAULT_CFG`, fp16 Dequantize). The attention output projection,
`lm_head`, embeddings and LayerNorms stay unquantized. On B300, decode is 1.08x
(TensorRT 11.4) and 1.14x (11.1) faster than fp16, prefill 1.07x to 1.10x, and
the bundle is 23.6% smaller. Other OPT sizes raise and name the calibration
tool.

### NVFP4

NVFP4 was measured on opt-125m and is not shipped. The study built real
TensorRT engines (no NVFP4 code is in the family): `fc1` and `fc2` of all 12
layers in NVFP4 with q/k/v in FP8, and a variant that keeps layers 0 and 11 in
FP8. Weights are packed FP4 with an FP8 scale per block of 16; activations are
quantized to FP4 at run time with the FP8 activation scales. All rows come from
one B300, TensorRT 11.4, max cache 256.

| opt-125m | Decode plan MiB | Decode ms (vs fp16) | Prefill 256 ms | KL to HF (x FP8) | PPL | Prefill texts with NaN |
|---|---|---|---|---|---|---|
| fp16 | 313.7 | 0.408 (1.00x) | 0.996 | 0.00024 | 70.919 | 0 of 200 |
| FP8 (shipped) | 239.6 | 0.371 (1.10x) | 0.896 | 0.01282 (1.0x) | 70.931 | 0 of 200 |
| NVFP4 `fc1` + `fc2` | 216.0 | 0.416 (0.98x) | 0.946 | 0.06313 (4.9x) | 71.738 | 107 of 200 |
| NVFP4, layers 0 and 11 FP8 | 220.0 | 0.394 (1.04x) | 0.937 | 0.04660 (3.6x) | 72.667 | 107 of 200 |
| NVFP4 `fc1` only | 227.9 | - | - | - | - | 0 of 200 |
| NVFP4 `fc2` only | build fails | - | - | - | - | - |

KL and PPL are measured through the decode engine. With a 2048-token cache the
picture is the same: decode 1.00x to 1.01x of fp16 for NVFP4 against 1.04x for
FP8.

Reasons:

- Speed: NVFP4 decode is 0.98x to 1.04x of fp16, slower than FP8 (1.10x), and
  prefill is slower than FP8 as well. A decode step of a 125M model is bound by
  many small kernels, the fp16 `lm_head` and attention, not by weight reads,
  and NVFP4 adds a run-time activation quantize in front of each of its 24
  projections. The plan is only 10% smaller than FP8.
- Quality: KL is 3.6x to 4.9x the FP8 KL.
- TensorRT problems: with `fc2` in NVFP4 the prefill engine returns NaN for
  107 of 200 texts and for all 8 short prompts; the first NaN appears right
  after the first NVFP4 feed-forward block. The decode engine with the same
  weights is finite. The `fc2`-only layout fails to build on TensorRT 11.4 (no
  implementation found for the network). The root cause was not found.
