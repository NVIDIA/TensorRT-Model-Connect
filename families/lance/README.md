# Lance

This family builds the understanding path (`x2t_image`, `x2t_video`) of
`bytedance-research/Lance` for `vision_language_generation` (`Lance_3B`, image
input, bf16 in the manifest). A Qwen2.5-VL ViT vision encoder feeds a Qwen-style text
decoder with QKV bias and per-head QK-norm. The bundle holds a vision engine
and decoder prefill and decode engines. The generation expert weights of the
checkpoint are not used.

## Quantization

`quantization=fp8` (with `precision=bf16`) quantizes the original checkpoint at
build time. q, k, v, gate, up and down of all 36 decoder layers run as FP8
GEMMs (per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`), except 10 down projections with outlier inputs.
Dequantize and the matmul run in fp32 and the result is cast to bf16. The vision
tower, projector, `lm_head`, embedding, norms and attention output projection
stay bf16. On TensorRT 11.4, B200, decode is 1.21x (CUDA graph) and prefill
(277 tokens) 1.31x faster than bf16.

### NVFP4

NVFP4 was measured and is not shipped. The study built real TensorRT engines
(no NVFP4 code is in the family): MLP in NVFP4; q/k/v and the MLP of the first
and last two layers in FP8. TensorRT 11.1, B200, one job. In this job the
quantized rows use a bf16 Dequantize, so compare them with each other and with
the bf16 engine.

| Variant | Quantized projections | KL vs PyTorch | Top-1 | Decode (CUDA graph) vs bf16 | Prefill vs bf16 |
|---|---|---|---|---|---|
| bf16 engine | 0 | 0.0036 | 0.970 | 1.00x | 1.00x |
| FP8 (shipped recipe) | 206 | 0.0526 | 0.893 | 1.31x (2.100 ms) | 1.24x |
| NVFP4 MLP, FP8 q/k/v and edge-layer MLP | 87 FP4 + 119 FP8 | 0.455 (8.7x FP8) | 0.727 | 1.24x (2.215 ms) | 1.43x (5.95 ms) |

Reasons:

- Speed: decode is slower than FP8 (2.215 ms against 2.100 ms); only prefill
  and plan size (2.73 + 3.30 GiB for prefill + decode) are better.
- Quality: KL is 8.7x the FP8 KL, top-1 drops from 0.893 to 0.727, and greedy
  token match falls to 140 of 288. A 3B decoder with 2 KV heads loses too much
  at 4 bits.
- No NaN result was reported for the NVFP4 engine.
