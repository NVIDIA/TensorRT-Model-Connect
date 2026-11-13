# StarCoder2

This family builds StarCoder2 code decoders for `text_generation`
(`bigcode/starcoder2-3b` in the manifests; the 7B and 15B checkpoints build from
the same code path). The model uses LayerNorm with bias, grouped-query
attention with RoPE and biases, and a GELU MLP (`c_fc`, `c_proj`). A bundle
holds a one-token decode engine and a prefill engine.

## Quantization

`quantization=fp8` (tensor parallel size 1; built and measured with
`precision=bf16`) quantizes the original checkpoint at build time for the 3B,
7B and 15B configs. q, k, v, `c_fc` and `c_proj` of every layer run as FP8
GEMMs (per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG`); Dequantize and the
matmul run in fp32 and the result is cast to bf16. `w_o`, embedding, norms and
`lm_head` stay 16 bit. On TensorRT 11.4, B200, decode is 1.13x (3B), 1.19x (7B)
and 1.42x (15B) faster than bf16.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: starcoder2-7b, one B200, TensorRT 11.4, bf16, 512-token build. NVFP4 on
`c_fc` and `c_proj` (block size 16, activations quantized at run time, input
scale from the FP8 calibration), q/k/v in FP8. KL is against the Hugging Face
bf16 model (code: 24217 held-out tokens; wikitext-2: 5110 tokens); the
non-finite scan feeds 200 real prompts (prefill and 8 decode steps).

| Variant | Bundle GiB | Decode ms (vs FP8) | Prefill 16 / 256 tokens vs FP8 | Code KL (x FP8) | Wikitext KL (x FP8) / top-1 | Prompts with non-finite values |
|---|---|---|---|---|---|---|
| FP8 (shipped) | 15.92 | 3.330 (1.00x) | 1.00x / 1.00x | 0.00263 (1.0x) | 0.00520 (1.0x) / 95.8% | 0 of 200 |
| NVFP4 `c_fc` + `c_proj`, fp32 Dequantize | 11.49 | 2.985 (1.12x) | 0.97x / 0.97x | 0.0212 (8.1x) | 0.0467 (9.0x) / 88.4% | 0 of 200 |
| NVFP4 `c_fc` only, fp32 Dequantize | 13.70 | 2.909 (1.14x) | 1.02x / 1.03x | 0.0118 (4.5x) | 0.0240 (4.6x) / 91.7% | 0 of 200 |
| NVFP4 `c_fc` + `c_proj`, bf16 Dequantize | 11.49 | 2.727 (1.22x) | 1.01x / 1.06x | 0.0231 (8.8x) | 0.0509 (9.8x) / 87.7% | 197 of 200 |

Reasons:

- Speed: decode is 1.12x to 1.14x faster than FP8 and the bundle is 14% to
  28% smaller; prefill gains nothing.
- Quality: KL is 4.5x to 9x the FP8 KL, and wikitext top-1 agreement falls from
  95.8% to 88% to 92%.
- The faster bf16 Dequantize layout gives non-finite prefill values on 197 of
  200 prompts, so it is not usable.
