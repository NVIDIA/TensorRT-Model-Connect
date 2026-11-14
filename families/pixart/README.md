# PixArt

This family builds PixArt-Sigma XL (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`)
for `image_generation` as three TensorRT engines: the T5 text encoder, the DiT
denoiser (28 blocks, hidden size 1152, 16 heads) and the VAE decoder. The
runtime runs the denoising loop with classifier-free guidance, so each step
calls the DiT twice (conditional and unconditional), at batch 1. The manifests
build fp16 at 512 x 512 and 1024 x 1024 (T5 kept in fp32) and fp32 with tensor
parallel size 4. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the DiT linears was built and measured on TensorRT
11.4 (and 11.1) on B200 and B300. It is not shipped: the variants that give a
speedup (feed-forward and self-attention q/k/v linears) give 1.05x to 1.15x on
the DiT call at batch 1, but their denoiser error is 3.5x to 6x the fp16 floor
and the images move clearly away from the reference (LPIPS 0.07 to 0.20 against
0.016 to 0.04 for fp16). The only variant at the fp16 floor (cross-attention
q/k/v) gives no speedup (1.00x to 1.01x). NVFP4 was not tried. This section
records the recipe, the method and the numbers, so that the study does not have
to be repeated. The FP8 code was not merged; this family has no FP8 build. All
numbers come from a local study.

### Summary and decision

DiT call time at batch 1 (one of the two calls per step), TensorRT 11.4; x is
fp16 time over FP8 time. Quality is from the 11.1 study (see Method).

| Variant | x 512, B200 / B300 | x 1024, B200 / B300 | plan GiB (fp16 1.114) | step error 512 / 1024 (fp16 floor 0.0089 / 0.0085) | LPIPS 512 / 1024 (fp16 floor 0.016 / 0.041) |
|---|---|---|---|---|---|
| feed-forward (fc1, fc2) | 1.05 / 1.05 | 1.10 / 1.06 | 0.837 | 0.051 / 0.047 | 0.081 / 0.177 |
| feed-forward + self q/k/v | 1.07 / 1.07 | 1.14 / 1.09 | 0.734 | above FF alone (self q/k/v alone 0.035 / 0.037) | above FF alone (self q/k/v alone 0.089 / 0.165) |
| feed-forward + self and cross q/k/v | 1.08 / 1.07 | 1.15 / 1.09 | 0.630 | 0.070 / 0.064 (all 10 linears) | 0.094 / 0.201 (all 10 linears) |
| cross q/k/v only | 1.01 / 1.00 | 1.00 / 1.00 | 1.010 | 0.011 / 0.0095 (with cross out) | 0.041 / 0.118 (with cross out) |

No non-finite output in any FP8 engine.

Decision: do not ship FP8 for PixArt.

- Quality: every variant that moves speed has 3.5x to 6x the fp16 step error
  and an LPIPS 2x to 5x the fp16 floor. Images keep composition, colors and
  faces, but fine detail differs and rendered text is lost.
- Speed: the DiT call is only 42% to 47% GEMM time at batch 1 (the rest is
  attention, adaptive layer norm in fp32, the masked cross-attention softmax
  chain and reshapes), so even the widest variant stays under 1.15x at batch 1.
  The DiT is 93% of the end-to-end time, so the end-to-end gain is the same.
- The variant that keeps quality (cross-attention linears, about 12% of the
  GEMM FLOPs, mostly on the 300 text tokens) gives no speedup.

### Recipe that was measured

No FP8 or NVFP4 checkpoint and no official quantization config exists for
PixArt-Sigma. The starting point was the usual DiT pattern (attention q/k/v/out
and feed-forward linears quantized; embeddings, adaptive layer norm, `norm_out`
and `proj_out` kept), then narrowed by a per-projection and per-block
sensitivity sweep.

| Choice | Decision |
|---|---|
| Quantized (per block, 28 blocks) | groups from: feed-forward `ff.net.0.proj`, `ff.net.2`; self-attention `attn1.to_q/k/v`; cross-attention `attn2.to_q/k/v` (and `attn2.to_out.0` on 11.1 only) |
| Kept fp16 | patch embedding, timestep and caption embeddings, adaptive layer norm (fp32), attention matmuls and softmax, `attn1.to_out.0`, `norm_out`, `proj_out`, T5 encoder, VAE decoder |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor, max calibration on the PyTorch model (diffusers) over the 20 denoising steps of calibration prompts, at the target resolution |
| Graph | `input -> Quantize -> Dequantize(fp16) -> MatMul`, weight `FP8 constant -> Dequantize(fp16)` |
| Dequantize type | fp16: with fp32 Dequantize (and fp32 MatMul plus a cast) the FP8 GEMM writes fp32 and a separate kernel adds the bias and casts, so the feed-forward variant gives 1.00x (512) and 1.04x (1024) (TensorRT 11.1, B200) |

Variants that did not help:

| Variant | Result |
|---|---|
| keep blocks 0 to 3 in fp16 | LPIPS 0.081 to 0.069 (512), 0.177 to 0.134 (1024); still 2x to 3x the floor |
| percentile and MSE-optimal activation clipping | no gain; the MSE-optimal clip is 1.0x max for most layers |
| builder optimization level 3 | same picture as level 1 |
| `attn2.to_out.0` in FP8 on TensorRT 11.4 | build fails ("Could not find any implementation"); on 11.1 it builds and gives 0.98x to 1.00x |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106.
- Speed: a standalone DiT engine with the family's graph (the same layers as the
  family builder, with an optional batch dimension), random inputs, builder
  optimization level 1. All variants of one configuration are built in the same
  job, then timed in 7 interleaved rounds of 30 calls; the median is reported.
- Quality (TensorRT 11.1, B200, and PyTorch fake quantization): 20 steps,
  guidance 4.5. Step error is the relative L2 error of the guided noise
  prediction against fp32 PyTorch, teacher-forced on the fp32 trajectory.
  LPIPS and PSNR compare the final images with fp32 PyTorch images on held-out
  prompts (4 to 6 per configuration, so LPIPS has noise). The 512 x 512 results
  use `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`: the 1024 checkpoint at 512 px is
  outside its training resolution (fp16 floor 4x higher, degraded images even
  in fp32) and hides the real sensitivity. The fake-quantization step error and
  LPIPS match the TensorRT engines to two digits.
- Non-finite scan: all FP8 timing engines and the quality runs.

### Results

Speed, DiT call ms, TensorRT 11.4 (FF = feed-forward, x = fp16 time over FP8
time):

| GPU | Size, batch | fp16 | FF | FF + self q/k/v | FF + self and cross q/k/v | cross q/k/v only |
|---|---|---|---|---|---|---|
| B200 | 512, 1 | 4.095 | 3.911 (1.05) | 3.840 (1.07) | 3.786 (1.08) | 4.058 (1.01) |
| B200 | 1024, 1 | 11.487 | 10.463 (1.10) | 10.086 (1.14) | 10.011 (1.15) | 11.438 (1.00) |
| B300 | 512, 1 | 3.871 | 3.696 (1.05) | 3.603 (1.07) | 3.613 (1.07) | 3.907 (1.00) |
| B300 | 1024, 1 | 10.181 | 9.625 (1.06) | 9.374 (1.09) | 9.327 (1.09) | 10.150 (1.00) |

The cross q/k/v only column comes from a second job with its own fp16
reference (B200 4.110 / 11.488 ms, B300 3.924 / 10.189 ms).

Batch 2 (the conditional and unconditional calls in one engine call; the
family runs them as two batch 1 calls), ms per image, TensorRT 11.4:

| GPU | Size | fp16 | FF | FF + self q/k/v | FF + self and cross q/k/v |
|---|---|---|---|---|---|
| B200 | 512 | 3.029 | 2.884 (1.05) | 2.806 (1.08) | 2.541 (1.19) |
| B200 | 1024 | 12.147 | 10.978 (1.11) | 10.527 (1.15) | 9.700 (1.25) |
| B300 | 512 | 2.807 | 2.653 (1.06) | 2.595 (1.08) | 2.335 (1.20) |
| B300 | 1024 | 10.701 | 9.681 (1.11) | 9.390 (1.14) | 8.642 (1.24) |

At 512, batching the pair without FP8 is already 1.35x to 1.38x per image (4.095
to 3.029 ms on B200, 3.871 to 2.807 ms on B300).

TensorRT 11.1, batch 1 (cross column: q/k/v and out):

| GPU | Size | fp16 | FF | FF + self q/k/v | FF + self and cross q/k/v | cross q/k/v/out |
|---|---|---|---|---|---|---|
| B200 | 512 | 4.143 | 3.940 (1.05) | 3.872 (1.07) | 3.829 (1.08) | 4.176 (0.99) |
| B200 | 1024 | 11.150 | 10.385 (1.07) | 10.118 (1.10) | 10.009 (1.11) | 11.321 (0.98) |
| B300 | 512 | 3.920 | 3.751 (1.05) | 3.629 (1.08) | 3.582 (1.09) | 3.992 (0.98) |
| B300 | 1024 | 10.178 | 9.506 (1.07) | 9.407 (1.08) | 9.365 (1.09) | 10.162 (1.00) |

Plan size: 1.114 GiB fp16, 0.837 FF, 0.734 FF + self q/k/v, 0.630 widest,
1.010 cross q/k/v.

Quality per projection group (TensorRT 11.1 study; step error mean (max)):

| Checkpoint at size | Variant | step error | LPIPS | PSNR dB |
|---|---|---|---|---|
| 512-MS at 512 | fp16 floor | 0.0089 (0.026) | 0.016 | 33.6 |
| | FF | 0.051 (0.17) | 0.081 | 26.2 |
| | FF without blocks 0 to 3 | 0.047 | 0.069 | 27.1 |
| | self q/k/v | 0.035 | 0.089 | 26.1 |
| | self q/k/v without blocks 0 to 3 | 0.021 | 0.075 | 26.6 |
| | cross q/k/v/out | 0.011 (0.030) | 0.041 | 28.6 |
| | all 10 linears | 0.070 (0.23) | 0.094 | 26.1 |
| 1024-MS at 1024 | fp16 floor | 0.0085 (0.024) | 0.041 | 29.1 |
| | FF | 0.047 (0.16) | 0.177 | 22.2 |
| | FF without blocks 0 to 3 | 0.042 | 0.134 | 23.1 |
| | self q/k/v | 0.037 | 0.165 | 21.5 |
| | cross q/k/v/out | 0.0095 (0.027) | 0.118 | 24.2 |
| | all 10 linears | 0.064 (0.23) | 0.201 | 20.3 |

TensorRT engine check (batch 2 engines, 1024, 4 prompts): LPIPS against fp32
PyTorch 0.040 fp16, 0.161 FF, 0.198 all linears except `attn1.to_out.0`; step
error 0.0083, 0.0476, 0.0635, equal to the fake-quantization numbers.

fp16 DiT call time split at batch 1 (TensorRT 11.1, B200, CUDA profiler):

| Size | Total ms | GEMM | fused self-attention | other (adaptive layer norm, GELU, cross-attention softmax chain, reshapes) |
|---|---|---|---|---|
| 512 | 4.17 | 2.10 (47%) | 0.53 (12%) | 1.87 (41%) |
| 1024 | 11.3 | 4.74 (42%) | 3.04 (27%) | 3.51 (31%) |

The masked cross-attention runs as a decomposed softmax chain: 0.88 ms (21% of
the call) at 512 and 1.09 ms (10%) at 1024. End to end (20 steps, two DiT calls
per step), the DiT is 93% of the time; T5 is 1% to 3% and the VAE decoder 7.0
ms (512) and 27.1 ms (1024), so FP8 of T5 could give at most 1.4% end to end.

### Why there is no fair gain

- The error comes from FP8 rounding noise that the 28 blocks and guidance
  amplify, not from the scale choice: per-step activation ranges vary only 0.4x
  to 0.7x of the max, and clipping did not help.
- The linears that carry the GEMM time (feed-forward, self q/k/v) carry the
  error; the insensitive cross-attention linears are a small share of the time.
- Half of the DiT call is not GEMM time, so the ceiling with all linears 2x
  faster is about 1.3x.

### NVFP4 (not tried)

FP8 already moves the images too far; NVFP4 has fewer mantissa bits and was not
built.

### What was not tried

| Item | Note |
|---|---|
| Per-timestep activation scales, SmoothQuant, quantization-aware tuning | not measured |
| FP8 of T5 and convolution FP8 of the VAE decoder | at most 1.4% (T5); the VAE is a small share |
| H100, the native runtime with an FP8 bundle | not measured |

### When to revisit

- A recipe that keeps the feed-forward and self-attention error near the fp16
  floor (for example per-timestep scales or a quantization-aware checkpoint).
- Faster levers that need no FP8: one batch 2 DiT call for the guidance pair
  (1.35x to 1.38x per image at 512) and a fused masked cross-attention (up to
  21% of the 512 call).
