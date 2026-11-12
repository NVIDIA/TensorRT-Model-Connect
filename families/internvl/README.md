# InternVL

This family builds InternVL3 `-hf` checkpoints (`OpenGVLab/InternVL3-2B-hf` and
`OpenGVLab/InternVL3-8B-hf` in the manifests) for `vision_language_generation`.
A bundle holds the language-model decoder (one engine with a prefill and a decode
profile that attend over the valid KV-cache rows), the InternViT vision engine
(one 448x448 tile per call, attention and MLP blocks in fp16) and the runtime
config. Images are split into 1 to 12 tiles plus a thumbnail, as the Hugging Face
processor does. Batch is 1. The manifests build the 2B model in fp16, FP8 and fp32
with tensor parallelism (2 GPUs), and the 8B model in fp16 and fp32 with tensor
parallelism (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint's language
model at build time (#1869). Query, key, value, gate, up and down projections of
every decoder layer get per-tensor FP8 weights (amax / 448) and static activation
scales from `fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, 128 ImageNet
validation images with captioning and question prompts). The attention output
projection, the vision engine, the projector, the embedding, the norms and the
output projection stay 16-bit. Scales ship for the 2B model only.

### NVFP4

NVFP4 is not shipped. On B300 it is not faster than FP8 in decode, and its
quality loss is 8x to 12x the FP8 loss. No ModelOpt NVFP4 checkpoint of InternVL3
exists, so NVFP4 W4A4 was self-quantized from the original checkpoint: per-16
e4m3 block scales and one `weight_scale_2 = amax / (6 x 448)` per tensor for the
weights, dynamic block-16 FP4 quantization of the activations with a static
global scale (the FP8 calibration amax / (6 x 448)). TensorRT fused the
projections into block-scaled FP4 GEMMs, gate and up into one dual GEMM.
Three variants were measured, all with the vision engine unchanged:

| Variant | NVFP4 | FP8 |
|---|---|---|
| NVFP4 all | q, k, v, gate, up, down of all 28 layers | - |
| NVFP4 MLP | gate, up, down of all layers | q, k, v |
| NVFP4 MLP, edge layers FP8 | gate, up, down of layers 2 to 25 | q, k, v; MLP of layers 0, 1, 26, 27 |

Speed: 2B model, `max_sequence_length` 4096, batch 1, engines built and timed in
the same job per GPU (TensorRT 11.4.0.106), median of 11 interleaved rounds.
Decode is one step from a CUDA graph with 300 or 3796 cache rows. The request is
the E2E image and prompt: 7 vision tiles, prefill of 1817 tokens and 32 decode
steps.

| Variant | Decoder plan | Decode 300, B200 / B300 | Decode 3796, B200 / B300 | Prefill 1817, B200 / B300 | Request, B200 / B300 |
|---|---|---|---|---|---|
| fp16 | 3.35 GiB | 2.243 / 1.909 ms | 2.517 / 2.139 ms | 6.81 / 6.68 ms | 102.3 / 88.7 ms |
| FP8 (main) | 2.17 GiB | 1.731 / 1.642 ms | 2.007 / 1.867 ms | 5.95 / 5.72 ms | 84.0 / 78.6 ms |
| NVFP4 all | 1.66 GiB | 1.913 / 1.952 ms (0.90x / 0.84x FP8) | 2.137 / 2.160 ms (0.94x / 0.86x) | 4.49 / 4.80 ms (1.32x / 1.19x) | 88.5 / 94.8 ms (0.95x / 0.83x) |
| NVFP4 MLP | 1.70 GiB | 1.618 / 1.690 ms (1.07x / 0.97x) | 1.923 / 1.915 ms (1.04x / 0.97x) | 4.22 / 4.62 ms (1.41x / 1.24x) | 79.4 / 81.4 ms (1.06x / 0.97x) |
| NVFP4 MLP, edge layers FP8 | 1.77 GiB | 1.647 / 1.683 ms (1.05x / 0.98x) | 1.950 / 1.917 ms (1.03x / 0.97x) | 4.48 / 4.75 ms (1.33x / 1.20x) | 81.6 / 80.7 ms (1.03x / 0.97x) |

TensorRT 11.1.0.106 (B200, 7 rounds, measured on the decoder before #2249; with
#2249 the fp16 decoder does not build on 11.1): decode 300 fp16 2.068, FP8 1.617,
NVFP4 all 1.752, NVFP4 MLP 1.558 ms; prefill 1817 FP8 6.81, NVFP4 MLP 5.16 ms.

Quality: teacher-forced decode engine against the PyTorch fp32 model on 96
held-out ImageNet validation images (not used in calibration), one image tile
and one of four describe or list prompts each, 64 greedy tokens of the PyTorch
model (3678 tokens). TensorRT 11.4; B200 first, B300 in brackets.

| Variant | KL to PyTorch (x FP8) | Top-1 agreement | Perplexity (PyTorch 1.626) | 4-option multiple choice, 80 images |
|---|---|---|---|---|
| fp16 | 0.0001 (0.0002) | 99.4% | 1.626 | 0.988 |
| FP8 (main) | 0.0124 (0.0119) | 94.8% | 1.635 | 0.988 |
| NVFP4 all | 0.1380 (0.1418), 11.1x (11.9x) | 82.4% | 1.860 | 0.988 |
| NVFP4 MLP | 0.1228 (0.1201), 9.9x (10.1x) | 83.8% | 1.794 | 0.988 |
| NVFP4 MLP, edge layers FP8 | 0.0988 (0.0974), 8.0x (8.2x) | 85.6% | 1.765 | 0.988 |

TensorRT 11.1 (B200, 32 images): FP8 0.0123, NVFP4 all 0.1305, NVFP4 MLP 0.1154.

NaN scan: no non-finite logits or K/V in any variant on B200 and B300 (11.4) and
B200 (11.1). Inputs, each with 24 decode steps: 24 images (12 on 11.1), the
7-tile E2E image, white, black and noise images, random-token prompts of 1001,
1999 and 3001 tokens, a repeated-token prompt of 1500 tokens and an upper-case
prompt of 1100 tokens; plus the quality and multiple-choice runs.

Why it is not shipped:

- Decode at batch 1 is not bound by weight reads. The FP8 MLP weights are about
  1.16 GB, about 0.14 ms at 8 TB/s, of a 1.6 to 2.0 ms step; NVFP4 can save
  about half of that. Measured: at most 1.07x over FP8 on B200 and 0.97x on B300.
- The quality loss is 8x to 12x the FP8 KL (ship rule: at most about 2x), also
  with the first and last two layers in FP8.
- The prefill gain (1.24x to 1.41x for the MLP variant at 1817 tokens) and the
  smaller plan (1.70 vs 2.17 GiB) do not change the decision.

Revisit NVFP4 for the 8B model (more weight-bound decode, but it has no
calibrated scales yet), for batched decode, or with a recipe that keeps the KL
within about 2x of FP8.
