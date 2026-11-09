# Mamba

This family builds Mamba-1 language models (`state-spaces/mamba-130m-hf` in the
manifests, also with tensor parallel size 4) for `text_generation`. The engine
runs one recurrent step per call: input `token_id [1]`, batch 1, the conv and
SSM states of every layer as inputs and outputs. The native runtime runs this
step once per prompt token (prefill) and once per generated token (decode). The
manifests use fp32; `precision=fp16` is also allowed (the scan, softplus and `A`
stay fp32). The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the `in_proj` and `out_proj` GEMMs was built and
measured on TensorRT 11.4 (and 11.1) for `state-spaces/mamba-130m-hf` and
`state-spaces/mamba-2.8b-hf`. It is not shipped: on mamba-130m, the only model
in CI, FP8 gives little or no gain (0.96x to 1.09x over fp16) and costs 8% top-1
flips. Only mamba-2.8b gains 1.09x to 1.17x, and that model (11 GB) is too large
for CI. This section records the recipe, the method and the numbers, so that the
study does not have to be repeated. The FP8 code was not merged; this family has
no FP8 build.

### Summary and decision

| Item | fp16 | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| mamba-130m step, B200 / B300 (TensorRT 11.4, CUDA graph) | 0.612 / 0.653 ms | 0.633 / 0.598 ms | 0.97x / 1.09x |
| mamba-2.8b step, B200 / B300 (TensorRT 11.4, CUDA graph) | 2.617 / 2.665 ms | 2.332 / 2.269 ms | 1.12x / 1.17x |
| mamba-2.8b plan size, B200 | 5294 MiB | 2911 MiB | -45% |
| mamba-130m KL / top-1 vs HF fp32 | 0.0001-0.0002 / 98.8-99.3% | 0.0163 / 92.0% | - |
| mamba-2.8b KL / top-1 vs HF fp32 | 0.00003-0.00006 / 99.2-99.5% | 0.0043 / 96.6% | - |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for Mamba.

- mamba-130m (the CI model): the step is bound by kernel launches and about 25
  small pointwise and reduce kernels per layer (conv step, softplus, scan,
  gate) that FP8 does not touch. The GEMMs are only about 45% of the step.
- mamba-2.8b: FP8 gives 1.09x to 1.17x over fp16, but plain fp16, which is
  already allowed, gives more over the fp32 default (1.38x to 1.49x) at KL below
  1e-4. Shipping FP8 for 2.8b-class configs needs a new qualified model in CI
  (an 11 GB download).
- NVFP4 has no GEMM gain over FP8 at batch 1.

### Recipe that was measured

No ModelOpt FP8 or NVFP4 checkpoint of Mamba-1 exists on the Hub. The recipe
follows the public recipes of hybrid models with Mamba layers (Nemotron-H,
Granite-4-H, Falcon-H1): quantize the large projections, keep the SSM path in
16 bit or fp32.

| Choice | Decision |
|---|---|
| FP8 W8A8 | `in_proj` (x and z halves, one shared input scale) and `out_proj` of every layer |
| 16 bit or fp32 | `x_proj`, `dt_proj` (small), `conv1d`, `A_log`, `D`, norms, embedding, LM head, SSM scan and states |
| Weight scale | per tensor, amax / 448 per projection, at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the Hugging Face model, 192 `nvidia/Nemotron-Post-Training-Dataset-v1` prompts and 64 WikiText-2 train passages (up to 512 tokens) |
| Outlier rule | an input scale above 20x the median of its kind stays 16 bit; measured: mamba-130m layer 23 `out_proj` (124x), mamba-2.8b layer 63 `out_proj` (47x) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106 (mamba-130m, B200).
- Quality: WikiText-2 test, 10 x 512 tokens (5110 teacher-forced positions),
  compared with Hugging Face fp32: KL mean and p99, top-1 agreement, perplexity.
  LAMBADA (`EleutherAI/lambada_openai`, first 200 test examples): greedy last
  word. Non-finite scan on 210 inputs.
- Speed: ms per step (one token), median of interleaved rounds. "graph" is CUDA
  graph replay (GPU time); "loop" is enqueue and sync per step, like the native
  runtime. Prefill costs the same step per prompt token.
- ModelOpt fake quantization of the same recipe gives the same KL on mamba-2.8b
  (0.0043), so the loss comes from the recipe, not from TensorRT.

### Results

Speed, TensorRT 11.4, batch 1, ms per step, graph / loop:

| Model | GPU | fp32 (manifest) | fp16 | FP8 (fp16 engine) | FP8 (fp32 engine) | FP8 vs fp16 |
|---|---|---|---|---|---|---|
| 130m | B200 | 0.643 / 0.772 | 0.612 / 0.741 | 0.633 / 0.770 | 0.658 / 0.787 | 0.97x / 0.96x |
| 130m | B300 | 0.615 / 0.711 | 0.653 / 0.757 | 0.598 / 0.707 | 0.620 / 0.717 | 1.09x / 1.07x |
| 2.8b | B200 | 3.904 / 4.231 | 2.617 / 2.917 | 2.332 / 2.669 | 2.657 / 3.025 | 1.12x / 1.09x |
| 2.8b | B300 | 3.823 / 4.082 | 2.665 / 2.950 | 2.269 / 2.544 | 2.585 / 2.880 | 1.17x / 1.16x |

Plan size, MiB, B200 / B300:

| Model | fp32 | fp16 | FP8 (fp16 engine) |
|---|---|---|---|
| 130m | 494 | 249 / 322 | 169 / 243 |
| 2.8b | 10563 | 5294 / 5539 | 2911 / 3156 |

TensorRT 11.1 (mamba-130m, B200): builds; FP8 KL 0.016, top-1 92%. Graph 0.518
ms (FP8) vs 0.555 ms (fp16), but loop 0.797 vs 0.648 ms: the FP8 plan has 293
layers vs 173, so Q/DQ is not folded as well as on 11.4.

Quality, TensorRT 11.4 (range over B200 and B300; Hugging Face fp32 in
brackets):

| Model | Engine | KL mean | KL p99 | top-1 vs HF | PPL | LAMBADA |
|---|---|---|---|---|---|---|
| 130m | fp16 | 0.0001-0.0002 | 0.0005-0.0013 | 98.8-99.3% | 26.51 (26.51) | 0.47-0.48 (0.475) |
| 130m | FP8 | 0.0163 | 0.097-0.100 | 92.0% | 26.97 | 0.45-0.455 |
| 2.8b | fp16 | 0.00003-0.00006 | 0.0001-0.0002 | 99.2-99.5% | 12.586 (12.586) | 0.72 (0.72) |
| 2.8b | FP8 | 0.0043 | 0.025 | 96.6% | 12.606 | 0.715 |

Why no gain: a GEMM-only chain of the real projection weights (`in_proj` and
`out_proj` of every layer, CUDA graph) is 1.06x to 1.27x faster in FP8 for
mamba-130m and 1.24x to 1.46x for mamba-2.8b (M = 1 to 256 rows). The GEMMs are
about 45% (130m) and 60 to 70% (2.8b) of a step; the rest is small pointwise
and reduce kernels. The family has no batch dimension and no batched prefill, so
the larger-M gains cannot be used.

### NVFP4 (GEMM-only, not built into the family)

GEMM-only chain at M = 1 (random weights, real projection widths, 24 layers
for 130m and 64 for 2.8b):

| Model | GPU | FP8 ms | NVFP4 ms | NVFP4 vs FP8 |
|---|---|---|---|---|
| 130m | B200 / B300 | 0.18-0.22 | 0.28-0.30 | slower |
| 2.8b | B200 | 1.29 | 1.22 | 1.06x |
| 2.8b | B300 | 1.12 | 1.19 | 0.94x |

The FP8 GEMV reads weights at only 2 to 3 TB/s; the decode step is not bound by
weight bandwidth, so 4-bit weights do not pay. NVFP4 was not built into the
family.

### Known limitation of the native runtime (not fixed here)

Found during the study, independent of FP8:

- No CUDA graph: CUDA graph replay of the same plan is 1.07x (2.8b) to 1.21x
  (130m) faster than the enqueue and sync loop that the runtime uses.
- Prefill runs one token per step: prefill latency is prompt tokens x step time
  (for example 512 tokens x 2.917 ms = 1.5 s for mamba-2.8b fp16 on B200).
- Each step copies all states: 2 x layers device-to-device copies per token
  (48 for 130m, 128 for 2.8b), plus a synchronous logits copy to the host.
- TensorRT 11.4 is slower than 11.1 on this step (B200, 130m): fp16 0.612 vs
  0.555 ms, fp32 0.643 vs 0.571 ms.

### What was not tried

| Item | Note |
|---|---|
| FP8 `x_proj`, `dt_proj`, LM head | not measured |
| Percentile or entropy calibration, per-channel weight scales | not measured |
| H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A larger Mamba model (2.8b class or larger) is added to CI: FP8 gives 1.09x
  to 1.17x there and a 45% smaller plan.
- The runtime gets a CUDA graph and a batched (multi-token) prefill or a batch
  dimension: then the GEMMs run at larger M, where FP8 is 1.2x to 1.5x faster.
