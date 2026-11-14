# Z-Image

This family builds Z-Image-Turbo (`Tongyi-MAI/Z-Image-Turbo` in the manifests)
for text-to-image generation. It builds three engines: the Qwen3 text encoder,
the Z-Image DiT (30 main blocks and 2 noise plus 2 context refiner blocks,
dim 3840, single-stream attention, SwiGLU feed-forward, tanh-gated AdaLN,
3-axis RoPE) and the AutoencoderKL VAE decoder. The runtime runs the
flow-matching Euler schedule (9 steps for Turbo, no guidance). Selected
components or DiT blocks can be built in fp32 inside an fp16 build
(`fp32_layers`).

## Quantization

`quantization="fp8"` (fp16 only, no tensor parallelism, `max_batch_size` 1)
self-quantizes the original BF16 DiT at build time (`quantization.py`, #1856):
the SwiGLU gate and up projections (`ff_w1`, `ff_w3`) of every DiT block run as
FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` from the loaded weights,
static per-tensor activation scale from `fp8_activation_scales.json`, ModelOpt
`FP8_DEFAULT_CFG` calibration). Attention projections, `ff_w2`, the AdaLN
modulation and the embedding and final projections, the text encoder and the
VAE stay as they are. A DiT block built in fp32 is not quantized. No other
quantization is accepted.

### NVFP4

NVFP4 was built and measured for the Z-Image-Turbo DiT (TensorRT 11.4, one job
on B200 and one on B300). It is not shipped: NVFP4 is 1.08x faster than the
FP8 build on both GPUs (1.22x when `ff_w2` is NVFP4 too), but the image error
(LPIPS) is 3.1x to 3.4x the FP8 error (4.2x to 4.7x with `ff_w2`), above
the 2x limit. No ModelOpt NVFP4 checkpoint of Z-Image-Turbo exists (the
community files on the Hub are single-file conversions for other tools), so
the weights were self-quantized from the BF16 checkpoint. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the FP8 activation scale divided by 6 (same
ModelOpt calibration; for `ff_w2` also divided by 64, the fp16 pre-scale of
`ff_w3`). NVFP4-only builds keep TensorRT's fused dual-GEMM kernel on (the FP8
build turns it off). The plan inspector shows FP4 GEMMs for the NVFP4
projections.

| Variant | NVFP4 | 16 bit |
|---|---|---|
| NVFP4 (same projections as FP8) | `ff_w1`, `ff_w3` (68) | everything else |
| NVFP4 feed-forward | `ff_w1`, `ff_w2`, `ff_w3` (102) | everything else |

Setup: 1024x1024 (4096 patches), text length 512, fp16 with the premerge fp32
DiT blocks 0, 1, 4 and 5 (these blocks are not quantized), batch 1. All plans
built from the family DiT builder and timed in the same job per GPU, TensorRT
11.4.0.106. DiT call: median of 120 calls (12 interleaved rounds of 10, stream
launch). Quality: the diffusers pipeline with the TensorRT DiT and the PyTorch
text encoder and VAE, 12 fixed prompts, seed 42, 9 steps, no guidance; LPIPS
(AlexNet) and PSNR of each image against the fp16 engine image, as in #1856.

| GPU | DiT call fp16 | FP8 (main) | NVFP4 (x FP8) | NVFP4 feed-forward (x FP8) |
|---|---|---|---|---|
| B200 | 82.45 ms | 79.74 ms | 74.07 ms (1.08x) | 64.91 ms (1.23x) |
| B300 | 76.32 ms | 70.36 ms | 65.15 ms (1.08x) | 57.81 ms (1.22x) |

| GPU | LPIPS FP8 | LPIPS NVFP4 (x FP8) | LPIPS NVFP4 feed-forward (x FP8) | PSNR FP8 / NVFP4 / feed-forward |
|---|---|---|---|---|
| B200 | 0.062 | 0.194 (3.1x) | 0.259 (4.2x) | 27.1 / 20.3 / 18.3 dB |
| B300 | 0.058 | 0.197 (3.4x) | 0.271 (4.7x) | 27.7 / 19.7 / 18.0 dB |

- Worst image (LPIPS): FP8 0.10 to 0.12, NVFP4 0.36 to 0.39, feed-forward
  0.46 to 0.59.
- DiT plan size: fp16 12.80 GiB, FP8 10.61 GiB, NVFP4 9.64 GiB, feed-forward
  8.06 GiB. The size gain alone is not a reason to ship.
- NaN: 0 non-finite DiT outputs for every variant on both GPUs (108 calls per
  variant in the image runs, plus random inputs). Random inputs scaled by 10
  or 100 give non-finite outputs for every variant including fp16, so that
  overflow is not caused by NVFP4.
- A variant with `to_q`, `to_k` and `to_v` in NVFP4 as well was attempted, but
  the experiment build did not route those projections to NVFP4 (same plan as
  the feed-forward variant), so it gives no extra result.

Why it does not help: the FP8 recipe already keeps only the projections with
the smallest image error (#1856 measured larger FP8 sets: quality drops as more
projections are quantized). NVFP4 on the same two projections is faster, but
4-bit weights and activations in this 9-step distilled model cost 3x the FP8
image error. Revisit with ModelOpt NVFP4 calibration of the weights (for
example AWQ or SVDQuant-style low-rank correction), a published NVFP4
checkpoint, or a recipe that keeps the LPIPS within 2x of FP8.
