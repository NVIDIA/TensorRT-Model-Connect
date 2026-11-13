# PersonaPlex

This family builds NVIDIA PersonaPlex (`nvidia/personaplex-7b-v1` in the
manifests) for `speech_to_speech`: a full-duplex speech model with a 7B
temporal transformer (32 layers, hidden 4096, SwiGLU MLP 11264) that runs one
step per 80 ms audio frame, a small depth transformer (16 engines, one per
audio codebook) and the Mimi audio encoder and decoder. The manifests build
fp16, batch 1, with optional tensor parallelism for unquantized builds.

## Quantization

`quantization="fp8"` self-quantizes the original checkpoint at build time
(`quantization.py`): the q, k, v, gate and up projections of the 32 temporal
layers run as FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` from the
loaded weights, static per-tensor activation scale from
`fp8_activation_scales.json`, calibrated on LibriSpeech validation speech by
`calibrate_fp8_scales.py`; Dequantize and matmul in fp32). The attention output
projection, the MLP down projection, the depth transformer, the Mimi codecs,
the embeddings, the norms and the text output projection stay unquantized. FP8
requires fp16, no `fp32_layers` and no tensor parallelism. No other
quantization is accepted.

### NVFP4

NVFP4 was built and measured for the temporal transformer of
personaplex-7b-v1 (TensorRT 11.4, one job on B200 and one on B300). It is not
shipped: the temporal step is 1.07x the FP8 build on B200 and 1.09x on B300
(CUDA graph), but the quality loss is 8.0x the FP8 loss (KL against the
PyTorch model), far above the 2x limit. The MLP-only variant is only 1.03x to
1.05x faster than FP8 and its KL is 6.0x the FP8 KL. No ModelOpt NVFP4
checkpoint of PersonaPlex exists, so the weights were self-quantized from the
original checkpoint. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the committed FP8 activation scale divided by 6.
Dequantize and matmul run in fp32 as in the FP8 recipe. Projections that are
not NVFP4 keep the FP8 layout of `quantization.py`; `w_o` and `w_down` stay
fp16 as in FP8. The fused gate/up dual-GEMM kernel stays on for NVFP4 (the FP8
build turns it off). The plan inspector shows FP4 GEMMs for every NVFP4
projection.

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| NVFP4 (same projections as FP8) | q, k, v, gate, up | none |
| MLP only | gate, up | q, k, v |

Results: temporal engine only, fp16 (the precision FP8 requires),
`max_sequence_length` 512, batch 1. The step is one temporal engine call at
position 100 with CUDA graph replay (eager launch in brackets), median of 9
interleaved rounds of 100 calls, all variants built and timed in the same job;
position 500 gives the same times. Quality is teacher-forced: every engine
gets the exact frame inputs of the official PyTorch bf16 run on 11 held-out
streams (10 LibriSpeech test-clean streams and the E2E `Recording.wav`, 1791
steps), and KL is KL(PyTorch || engine) of the text logits per step. NaN counts
non-finite hidden states, logits and K/V values.

| GPU | TensorRT | Step fp16 | Step FP8 | Step NVFP4 (x FP8) | Step MLP only (x FP8) | KL fp16 / FP8 / NVFP4 / MLP only | NaN |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 6.33 ms (6.69) | 3.96 ms (4.29) | 3.71 ms, 1.07x (1.06x) | 3.84 ms, 1.03x (1.02x) | 0.00005 / 0.00054 / 0.00430 (8.0x) / 0.00320 (6.0x) | 0 |
| B300 | 11.4 | 4.79 ms (5.04) | 3.39 ms (3.58) | 3.10 ms, 1.09x (1.11x) | 3.25 ms, 1.05x (1.04x) | 0.00005 / 0.00054 / 0.00430 (8.0x) / 0.00320 (6.0x) | 0 |

- Text top-1 agreement with PyTorch (B300): fp16 99.78%, FP8 99.55%, NVFP4
  98.60%, MLP only 98.55%. Text logit cosine: 0.99989 / 0.99851 / 0.98467 /
  0.98433; hidden-state cosine: 0.99994 / 0.99876 / 0.98891 / 0.99217. The
  quantized variants give the same quality numbers on both GPUs.
- Temporal plan size: fp16 12.74 GiB, FP8 8.50 GiB, NVFP4 6.63 GiB, MLP only
  7.29 GiB. The size gain alone is not a reason to ship.
- NaN scan: 0 non-finite values and finite K/V for every variant on both GPUs
  (random tokens, one repeated token, pad tokens and the real streams
  concatenated to a full 511-step cache).
- NVFP4 with the Dequantize output in fp16 instead of fp32 gives the same
  quality and is not faster (1.03x FP8 on B200, 1.06x on B300).

Why it does not help: the temporal step is only part of a frame. The depth
transformer (16 small engines, about 10 ms per frame) is not quantized, so FP8
already brings one frame only from about 15.4 ms to 13.8 ms (1.1x); NVFP4
saves another 0.25 ms to 0.3 ms per frame (about 2%), while its KL is 8x the
FP8 KL. Revisit with a recipe that keeps the KL within 2x of FP8 (for example
a calibrated NVFP4 input scale or a published NVFP4 checkpoint), or when the
depth transformer is no longer the larger part of the frame.
