# MiniMax-H3

This family builds `MiniMaxAI/MiniMax-H3` for text to video with audio (the
`t2va` workflow) at one fixed profile: 1344x768, 124 frames, 50 denoising
steps. The bundle holds a text encoder engine, an adaLN precompute engine, the
DiT denoiser (50 blocks, hidden size 5376, 38,247 packed text, audio and video
rows, one call per step) and a VAE tile decoder. Builds are bf16. An optional
first-block-cache profile splits the denoiser into three engines.

## Quantization

`quantization="fp8"` self-quantizes the denoiser of the original bf16
checkpoint at build time (#1947): the fused q/k/v GEMM, the fused gate/up GEMM
and the down GEMM of all 50 blocks run as FP8 W8A8 GEMMs (per-tensor weight
scale `amax / 448`, static activation scales from `fp8_activation_scales.json`,
bf16 Dequantize). The attention output projection, the token refiner, adaLN,
embeddings, heads, norms, the text encoder and the VAE stay bf16. FP8 is not
available with the first-block-cache profile. No other quantization is
accepted.

### NVFP4

NVFP4 was built and measured for the denoiser. It is 1.08x to 1.11x faster than
FP8 per DiT call on B200 and B300, but it is not shipped: its per-call error
against the bf16 engine is 2.4x to 3.3x the FP8 error, above the 2x limit, and
in full videos every NVFP4 video is far from the bf16 video (LPIPS above 0.5,
PSNR 13.1 dB against 17.0 dB for FP8). No ModelOpt NVFP4 checkpoint of
MiniMax-H3 exists. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 for the same projections as FP8 (q/k/v, gate/up, down of all
50 blocks), weights self-quantized from the bf16 checkpoint (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per fused weight, E2M1 values
packed two per byte); activations through `add_dynamic_quantize` (block 16) with
a static global scale equal to the committed FP8 activation scale divided by 6
(same calibration data and statistic). Everything else is as in the FP8 build.

Speed: the denoiser plan of `dit_builder.build_dit_engine`, bf16, FP8 and NVFP4
built and timed in the same job (one job per GPU and TensorRT version), median
of 5 interleaved rounds of 2 calls at the native shape.

| GPU | TensorRT | bf16 | FP8 | NVFP4 (x FP8) | Plan bf16 / FP8 / NVFP4 |
|---|---|---|---|---|---|
| B200 | 11.4 | 3204 ms | 2648 ms | 2447 ms (1.08x) | 37.40 / 21.25 / 14.19 GiB |
| B300 | 11.4 | 2851 ms | 2348 ms | 2121 ms (1.11x) | 37.40 / 21.25 / 14.19 GiB |
| B200 | 11.1 | not run | 2662 ms | 2452 ms (1.09x) | - / 21.25 / 14.19 GiB |

Quality:

- Per DiT call (TensorRT 11.4, same result on B200 and B300): relative L2
  error of the video velocity against the bf16 plan, with real position ids and
  adaLN modulation at steps 0, 24 and 48, N(0, 1) latents and N(0, 1) text
  features: FP8 0.048 / 0.056 / 0.058, NVFP4 0.156 / 0.137 / 0.162 (2.4x to
  3.3x). Audio velocity: FP8 0.028 / 0.045 / 0.136, NVFP4 0.061 / 0.124 /
  0.408.
- Full videos (PyTorch `t2va` pipeline in bf16 on B300 with the same recipes
  fake-quantized in the DiT projections, 4 prompts including the E2E prompt,
  50 steps, seed per prompt, LPIPS-alex over every 4th frame, against the bf16
  video): FP8 LPIPS 0.350 (0.21 to 0.44), PSNR 16.95 dB; NVFP4 LPIPS 0.558
  (0.51 to 0.60), PSNR 13.14 dB. The LPIPS ratio is 1.6x, but all four NVFP4
  videos are above 0.5, where the FP8 videos stay below 0.45.
- NaN scan: 0 non-finite outputs for every plan (the cases above and a stress
  case with latents x4 and text features x8) and every fake-quantized video.

Revisit with a recipe that keeps more of the model in FP8 (for example NVFP4
only for gate/up), with an NVFP4 calibration of the activation scales, or when
a native-runtime E2E of the quantized denoiser exists to judge full videos.
