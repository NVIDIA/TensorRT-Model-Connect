# Whisper

This family builds Whisper speech models for the `transcription` task as an
encoder engine and a decoder engine (`openai/whisper-tiny`,
`openai/whisper-small` and `openai/whisper-large-v3-turbo` in the manifests).
Batch is 1. The manifests build fp16 (tiny, small; small with the encoder in
fp32), fp32 (large-v3-turbo), FP8 (small) and tensor parallelism (2 and 4
GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#1840). Encoder q, k, v, fc1 and fc2, decoder self-attention q, k, v,
cross-attention q, k, v, o and MLP fc1, fc2 get per-tensor FP8 weights
(amax / 448) and static activation scales from `fp8_activation_scales.json`
(ModelOpt `FP8_DEFAULT_CFG`, 192 LibriSpeech validation clips). The
self-attention output projections, convolution stem, embeddings, layer norms and
output projection stay fp16. Dequantize and the matmul run in fp32 and the
result is cast back to fp16. Scales ship for tiny, small, large-v3-turbo and
large-v3. On 80 LibriSpeech test clips the WER is unchanged or lower than fp16,
and the bundle goes from 0.61 to 0.44 GiB (small) and from 1.76 to 1.14 GiB
(large-v3-turbo).

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. No NVFP4 checkpoint that this
family can read exists (the only one found is a community export in the
compressed-tensors format), and NVFP4 was out of scope in #1840. The manifest
models (tiny, small, large-v3-turbo) have under 1B parameters and run at batch
1, and FP8, which halves the weight bytes of fp16, gives only a small gain
(numbers from #1840, B300; the ratios are computed from those times):

| Item | small | large-v3-turbo |
|---|---|---|
| Bundle fp16 / FP8 | 0.61 / 0.44 GiB | 1.76 / 1.14 GiB |
| Encoder fp16 / FP8, TensorRT 11.4 | 0.87 / 0.94 ms (0.93x) | 3.50 / 3.20 ms (1.09x) |
| Decoder step fp16 / FP8, TensorRT 11.4 | 1.12 / 0.94 ms (1.19x) | 0.42 / 0.40 ms (1.05x) |
| Decoder step fp16 / FP8, TensorRT 11.1 | 1.08 / 0.87 ms (1.24x) | 0.47 / 0.39 ms (1.21x) |

The encoder gain is small because the attention matmuls, layer norms, Q/DQ
operations and self-attention output projections stay fp16 (#1840); 4-bit
weights do not change those. NVFP4 also adds one dynamic quantize kernel per
projection.
