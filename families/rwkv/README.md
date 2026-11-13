# RWKV

This family builds RWKV-4 recurrent language models for `text_generation`
(`RWKV/rwkv-4-169m-pile` in the manifests: fp32 with tensor parallel size 1 and
4, and an FP8 case). A bundle holds one decode-step engine (one token plus five
fp32 state tensors per layer, batch 1); prefill runs the same step once per
prompt token. fp16 builds are also supported.

## Quantization

`quantization=fp8` (fp16 precision, tensor parallel size 1) quantizes the
original checkpoint at build time for `rwkv-4-169m-pile`, `rwkv-4-430m-pile` and
`rwkv-4-1b5-pile`. The seven projections of every layer except the last
(attention key, value, receptance, output; feed-forward key, receptance, value)
run as FP8 GEMMs: per-tensor weight scale amax / 448, static activation scales
from `fp8_activation_scales.json`, fp16 Dequantize. The last layer, embedding,
`lm_head` and norms stay fp16, the WKV state fp32. On TensorRT 11.4, batch 1, a
decode step is 1.20x to 1.41x faster than fp16 in the engine (CUDA graph) and
1.12x to 1.14x in the native runtime; the plan is 24% to 41% smaller. The
premerge case `rwkv-169m-fp8` covers the FP8 bundle.

### NVFP4

NVFP4 was measured as GEMM timing only and is not shipped. No NVFP4 engine of
the model was built, so there is no quality or NaN measurement. The benchmark is
a chain of the family's projection widths with random weights, one row (M = 1,
as in decode), NVFP4 weights and FP4 activations quantized at run time, CUDA
graph, TensorRT 11.4.

| Widths | GPU | fp16 ms | FP8 ms | NVFP4 ms | NVFP4 vs FP8 | NVFP4 vs fp16 |
|---|---|---|---|---|---|---|
| 169m (12 layers, 768 / 3072) | B200 | 0.279 | 0.201 | 0.250 | 0.80x | 1.12x |
| 169m | B300 | 0.243 | 0.171 | 0.237 | 0.72x | 1.03x |
| 3B (32 layers, 2560 / 10240) | B200 | 1.912 | 1.226 | 1.080 | 1.14x | 1.77x |
| 3B | B300 | 1.529 | 1.085 | 1.066 | 1.02x | 1.43x |

Reasons:

- Speed: at the 169m widths NVFP4 is slower than FP8 (0.72x to 0.80x). Only at
  the 3B widths is it faster than FP8 (1.02x to 1.14x), and the family cannot
  load the 3B checkpoint (it is sharded, and the loader reads one
  `pytorch_model.bin`). The FP8 GEMV reads weights at only 2 to 3 TB/s, so
  decode is not weight-bandwidth bound and 4-bit weights save little.
- Quality: FP8 already costs KL 0.007 to 0.013 against the PyTorch model on
  these small models; NVFP4 quality was not measured.
