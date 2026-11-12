# LocateAnything

This family builds `nvidia/LocateAnything-3B` for `vision_language_generation`
(grounding and pointing: the answer holds box or point coordinate tokens). A
bundle holds the language-model decoder as two engines (prefill and one-token
decode, Qwen2.5 decoder that takes the image features as input embeddings), the
MoonViT vision engine with the projector (one fixed 448x448 image, 256 image
tokens) and the runtime config. Batch is 1. The manifests build fp16 and FP8,
`max_sequence_length` 384.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint's
language model at build time (#1870). Query, key, value, gate, up and down
projections of all 36 decoder layers get per-tensor FP8 weights (amax / 448) and
static activation scales from `fp8_activation_scales.json` (ModelOpt
`FP8_DEFAULT_CFG`, 128 ImageNet validation images with grounding and pointing
prompts). Dequantize and the matmul run in fp32 and the result is cast back. The
attention output projection, the vision engine, the projector, the embedding,
the norms and the output projection stay fp16.

### NVFP4

NVFP4 is not shipped. It is at most 1.02x faster than FP8 in decode on B300,
and its quality loss is 7x to 11x the FP8 loss. No ModelOpt NVFP4 checkpoint of
LocateAnything-3B from NVIDIA exists (the one community export also packs the
projector), so NVFP4 W4A4 was self-quantized from the original checkpoint: per-16
e4m3 block scales and one `weight_scale_2 = amax / (6 x 448)` per tensor for the
weights, dynamic block-16 FP4 quantization of the activations with a static
global scale (the FP8 calibration amax / (6 x 448)). TensorRT ran every NVFP4
projection as a block-scaled FP4 GEMM. Three variants were measured, all with
the vision engine unchanged:

| Variant | NVFP4 | FP8 |
|---|---|---|
| NVFP4 all | q, k, v, gate, up, down of all 36 layers | - |
| NVFP4 MLP | gate, up, down of all layers | q, k, v |
| NVFP4 MLP, edge layers FP8 | gate, up, down of layers 2 to 33 | q, k, v; MLP of layers 0, 1, 34, 35 |

Speed: `max_sequence_length` 384 (the manifest), batch 1, engines built and timed
in the same job per GPU (TensorRT 11.4.0.106), median of 9 interleaved rounds.
Decode is one step from a CUDA graph with 300 cache rows; prefill is the
291-token grounding prompt (256 image tokens) from a CUDA graph. The request is
vision, prefill and 32 decode steps driven from Python.

| Variant | Decoder plans | Decode, B200 / B300 | Prefill 291, B200 / B300 | Request, B200 / B300 |
|---|---|---|---|---|
| fp16 | 2 x 6.35 GiB | 2.892 / 2.488 ms | 8.04 / 7.78 ms | 128.2 / 105.6 ms |
| FP8 (main) | 2 x 3.91 GiB | 2.147 / 2.276 ms | 5.73 / 5.66 ms | 95.6 / 98.8 ms |
| NVFP4 all | 2 x 2.83 GiB | 2.191 / 2.561 ms (0.98x / 0.89x FP8) | 5.06 / 5.62 ms (1.13x / 1.01x) | 100.6 / 110.5 ms (0.95x / 0.89x) |
| NVFP4 MLP | 2 x 2.91 GiB | 1.997 / 2.236 ms (1.08x / 1.02x) | 5.05 / 5.40 ms (1.14x / 1.05x) | 90.0 / 99.2 ms (1.06x / 1.00x) |
| NVFP4 MLP, edge layers FP8 | 2 x 3.02 GiB | 2.017 / 2.227 ms (1.06x / 1.02x) | 4.98 / 5.47 ms (1.15x / 1.03x) | 90.8 / 99.8 ms (1.05x / 0.99x) |

Quality: against the fp16 engine (the reference of the FP8 change) on 64
held-out ImageNet validation images (not used in calibration), each with the
prompt "Locate a single instance that matches the following description:
<class name>.". KL is teacher-forced along the fp16 greedy answer (737 tokens,
up to 32 per image). TensorRT 11.4; B200 first, B300 in brackets.

| Variant | KL to fp16 (x FP8) | Top-1 agreement | First-token logit cosine | Identical answers | First-box IoU to fp16 (at least 0.9) |
|---|---|---|---|---|---|
| FP8 (main) | 0.0009 (0.0009) | 94.8% (96.2%) | 0.9975 (0.9971) | 34 (43) of 64 | 0.992 (0.994), 63 (63) of 64 |
| NVFP4 all | 0.0093 (0.0090), 10.5x (9.7x) | 87.4% (86.3%) | 0.9799 (0.9796) | 9 (10) | 0.956 (0.969), 59 (61) |
| NVFP4 MLP | 0.0078 (0.0069), 8.8x (7.4x) | 87.9% (88.5%) | 0.9809 (0.9794) | 11 (13) | 0.967 (0.979), 59 (62) |
| NVFP4 MLP, edge layers FP8 | 0.0068 (0.0072), 7.7x (7.7x) | 89.6% (89.4%) | 0.9843 (0.9829) | 16 (11) | 0.978 (0.974), 62 (61) |

The first generated token matches fp16 on all 64 images for every variant.

NaN scan: no non-finite logits or K/V in any variant on B200 and B300. Inputs,
each with prefill and up to 24 decode steps: 16 held-out images, white, black
and noise images, a pointing prompt, a 344-token prompt with an image, and
344-token random-token, repeated-token and upper-case text prompts; plus the
quality runs.

Why it is not shipped:

- Decode at batch 1 is only partly bound by weight reads: the FP8 MLP weights
  are about 2.4 GB, about 0.3 ms of a 2.1 to 2.3 ms step at 8 TB/s, and NVFP4
  saves about half of that. Measured: 1.06x to 1.08x over FP8 on B200 and 1.02x
  on B300 for the MLP variants (ship rule: at least 1.05x on both).
- The quality loss is 7x to 11x the FP8 KL (ship rule: at most about 2x), also
  with the first and last two layers in FP8.
- The prefill gain (1.03x to 1.15x) and the smaller plans (2.9 vs 3.9 GiB each)
  do not change the decision.

Revisit NVFP4 for batched decode, with an NVIDIA ModelOpt NVFP4 checkpoint, or
with a recipe that keeps the KL within about 2x of FP8.
