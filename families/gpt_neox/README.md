# GPT-NeoX

This family builds GPT-NeoX and Pythia decoders for `text_generation`
(`EleutherAI/pythia-70m` in the manifests, fp32, tensor parallel size 1 and 4).
The model uses LayerNorm, a parallel residual, partial RoPE, a fused
head-interleaved `query_key_value` projection and a GELU MLP. A bundle holds a
one-token decode engine and a dynamic-length prefill engine.

## Quantization

`quantization=fp8` (with `precision=fp16`, tensor parallel size 1) quantizes the
original checkpoint at build time for `EleutherAI/pythia-410m` and
`EleutherAI/pythia-1b`. q, k, v, `dense_h_to_4h` and `dense_4h_to_h` of every
layer run as FP8 GEMMs (per-tensor weight scale amax / 448, static activation
scales from `fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG`); the
attention output projection, output head, embedding and norms stay fp16.
pythia-70m and pythia-160m have no scales, because their fp16 engine is not
usable. On TensorRT 11.4, B300, decode is 1.08x (410m) and 1.12x (1b) faster
than fp16.

### NVFP4

NVFP4 was measured and is not shipped. The study built real TensorRT engines
(no NVFP4 code is in the family): MLP (`dense_h_to_4h` and `dense_4h_to_h`) in
NVFP4, q/k/v in FP8. "Mixed" keeps the MLP of the first and last two layers in
FP8, "all" quantizes every layer. One job on one B300, TensorRT 11.4, 256-token
build.

| Model, variant | Decode plan MiB | Decode vs fp16 (CUDA graph) | Prefill 255 vs fp16 | KL vs HF fp32 (x FP8) | Top-1 agreement | Wikitext-2 PPL |
|---|---|---|---|---|---|---|
| 410m FP8 (shipped) | 511.1 | 1.076x | 1.170x | 0.0346 (1.0x) | 89.6% | 38.12 |
| 410m NVFP4 mixed | 441.2 | 0.974x | 1.090x | 0.122 (3.5x) | 79.4% | 41.41 |
| 410m NVFP4 all | 427.2 | 0.957x | 1.075x | 0.152 (4.4x) | 76.5% | 42.47 |
| 1b FP8 (shipped) | 1227.6 | 1.121x | 1.294x | 0.0120 (1.0x) | 93.2% | 31.15 |
| 1b NVFP4 mixed | 1059.6 | 1.121x | 1.337x | 0.0619 (5.2x) | 85.8% | 32.90 |
| 1b NVFP4 all | 1003.6 | 1.129x | 1.375x | 0.0820 (6.8x) | 83.4% | 33.56 |

Reasons:

- Speed: decode is not faster than FP8 (equal on 1b, slower on 410m). Decode
  at these sizes is not bound by weight reads.
- Size: 14% to 18% smaller plan than FP8.
- Quality: KL 3.5x to 6.8x the FP8 value.
- NaN: pythia-410m with NVFP4 gives non-finite logits on some real prompts.
