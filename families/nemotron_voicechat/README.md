# Nemotron VoiceChat

This family builds `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` for the
`speech_session` task: a streaming conformer encoder, an RNNT decoder, the
thinker (a Nemotron-H hybrid language model with 56 Mamba-2, MLP and attention
layers, plus a function head), a TTS backbone and an audio codec, driven by the
native C++ runtime in `runtime/`. The thinker is a single-token fp32 decode
engine with a static cache of `max_sequence_length` tokens (default 8192).
Precision is fp32 only, batch 1, no tensor parallelism.

## Quantization

`quantization="fp8"` self-quantizes the thinker of the original fp32 checkpoint
at build time (`quantization.py`): Mamba-2 `in_proj` and `out_proj`, MLP
`up_proj` and `down_proj`, and attention `q_proj`, `k_proj` and `v_proj` run as
FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` from the loaded weights,
static per-tensor activation scale from `fp8_activation_scales.json`, fp32
Dequantize and matmul). The attention output projection, `conv1d`, the SSM
parameters, the norms, the embedding, `lm_head` and the function head stay
fp32, and the other components are not touched. FP8 makes the thinker step
1.6x faster than fp32 and the plan 60% smaller. No other quantization is
accepted.

### NVFP4

NVFP4 was built and measured for the VoiceChat thinker with a quick screen
(TensorRT 11.4, one job on B200 and one on B300). It is not shipped: the
variant with the FP8 projections is 1.03x to 1.09x faster than the FP8 build,
but its quality loss is about 9x the FP8 loss (KL against the fp32 engine), far
above the 2x limit. No variant is both faster on both GPUs and within 2x of the
FP8 KL. The NVFP4 code was not merged.

No ModelOpt NVFP4 checkpoint of VoiceChat exists. The published
`nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4` checkpoint is the base language
model, not the VoiceChat thinker: its weights differ (embedding relative L2
difference 0.07, layer norm 0.01) and its layer-1 MLP is not NVFP4, so it
cannot be reused. The weights were self-quantized from the fp32 checkpoint.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the committed FP8 activation scale divided by 6.
Projections that are not NVFP4 keep the FP8 layout of `quantization.py`. The
plan inspector shows FP4 GEMMs for every NVFP4 projection.

| Variant | NVFP4 | FP8 W8A8 | Dequantize and GEMM |
|---|---|---|---|
| NVFP4 (same projections as FP8) | Mamba `in_proj`, `out_proj`, MLP up, down, attention q, k, v | none | fp32 |
| MLP only | MLP up, down | Mamba `in_proj`, `out_proj`, attention q, k, v | fp32 |
| NVFP4, fp16 GEMM | same as NVFP4 | none | fp16 (cast in and out) |

Results: thinker engine only, fp32 engine, `max_sequence_length` 8192, batch 1.
Step time is one token at position 100 with CUDA graph replay (eager launch in
brackets), median of 9 interleaved rounds of 100 steps, all variants built and
timed in the same job (position 4000 gives the same times). Quality is
teacher-forced over 16 held-out LibriSpeech test clips (8 clean, 8 other, 2199
audio steps): the fp32 engine drives the native-runtime frame loop and every
variant gets the same inputs. KL is KL(fp32 engine || variant) of the text
logits per step; top-1 is text next-token agreement with the fp32 engine.

| GPU | TensorRT | Step fp32 | Step FP8 | Step NVFP4 (x FP8) | Step MLP only (x FP8) | Step NVFP4 fp16 GEMM (x FP8) |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 8.49 ms (8.58) | 5.15 ms (5.25) | 4.89 ms, 1.05x (1.09x) | 4.96 ms, 1.04x (1.04x) | 4.54 ms, 1.14x (1.12x) |
| B300 | 11.4 | 9.80 ms (9.90) | 6.43 ms (6.49) | 6.25 ms, 1.03x (1.08x) | 6.17 ms, 1.04x (1.03x) | 5.80 ms, 1.11x (1.09x) |

| GPU | KL FP8 | KL NVFP4 (x FP8) | KL MLP only (x FP8) | KL NVFP4 fp16 GEMM (x FP8) | Top-1 FP8 / NVFP4 / MLP only / fp16 GEMM | NaN |
|---|---|---|---|---|---|---|
| B200 | 0.0017 | 0.0159 (9.3x) | 0.0060 (3.5x) | 0.0158 (9.3x) | 99.2% / 97.1% / 98.3% / 97.1% | 0 |
| B300 | 0.0018 | 0.0164 (9.1x) | 0.0064 (3.5x) | 0.0155 (8.6x) | 98.8% / 97.4% / 98.1% / 97.4% | 0 |

- Text logit cosine mean (B200): FP8 0.9984, NVFP4 0.9854, MLP only 0.9943,
  fp16 GEMM 0.9853. Function head top-1: 99.9% to 100% for every variant.
- Thinker plan size: fp32 37.9 GB, FP8 15.0 GB, NVFP4 11.7 GB, MLP only
  13.5 GB, fp16 GEMM 11.7 GB. The size gain alone is not a reason to ship.
- NaN scan: 0 non-finite logits and 0 non-finite Mamba or K/V states for every
  variant on both GPUs, on the 16 clips and on stress inputs (8 s white noise,
  speech scaled x12 and clipped, 6 s silence, a random 2000-token prompt).

Why it does not help enough: FP8 already removes most of the projection weight
traffic of the thinker step. NVFP4 halves it again, but each NVFP4 projection
adds a dynamic activation quantization and the fp32 Dequantize path, so the
step gains only 3% to 9% over FP8; the fp16 GEMM variant gains 9% to 14% but
keeps the same 9x KL. The MLP-only variant keeps the KL lower (3.5x FP8) but is
only 1.03x to 1.04x faster. Revisit with a calibrated NVFP4 recipe (ModelOpt
`NVFP4_DEFAULT_CFG` or a published VoiceChat NVFP4 checkpoint) that keeps the
KL within 2x of FP8, or together with an FP8 or NVFP4 `lm_head` and function
head.
