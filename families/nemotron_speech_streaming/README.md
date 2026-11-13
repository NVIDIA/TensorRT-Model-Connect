# Nemotron Speech Streaming

This family builds FastConformer cache-aware RNNT speech recognition models for
the `transcription_streaming` task from the NeMo `.nemo` archive:
`nvidia/nemotron-speech-streaming-en-0.6b` (English) and the multilingual
`nemotron-3.5-asr-streaming-0.6b` in the manifests. A bundle holds the acoustic
encoder (one offline plan plus a first-step and a steady-state streaming plan for
each supported right context), the RNNT prediction network and the joint
network. The native runtime runs greedy RNNT decoding. The manifests build fp16,
FP8 and fp32 with tensor parallelism (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the `.nemo` weights at build time
(#1875, #2204). The four feed-forward matrices (`ff1.w1`, `ff1.w2`, `ff2.w1`,
`ff2.w2`) of the 24 encoder layers get per-tensor FP8 weights (amax / 448) and
static activation scales from `fp8_activation_scales.json` (192 LibriSpeech
validation clips). Dequantize and the matmul run in fp32 and are cast back. The
attention projections stay fp16: with them in FP8, LibriSpeech test-clean WER
was 3.49 against 2.49 for the native model; with them in fp16 it is 2.58 (GB300,
500 utterances, #2204). Convolutions, the positional projection, the prediction
and joint networks, the prompt kernel and the norms stay fp16. Scales ship for
the English 0.6B model only.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. The 0.6B encoder runs one
1120 ms chunk per step in about 2 ms at batch 1, and FP8 gives only a small
speed gain (numbers from #1875, B300, TensorRT 11.1, right context 13, measured
with the attention projections also in FP8):

| Item | Value |
|---|---|
| Streaming encoder step, 1120 ms chunk, fp16 | 2.220 ms |
| Streaming encoder step, 1120 ms chunk, FP8 | 2.037 ms |
| FP8 vs fp16 | 1.09x |
| Streaming plan, fp16 to FP8 | 1126 to 647 MiB |

The main FP8 result is bundle size, not speed. The shipped recipe quantizes
fewer projections than this measurement (#2204), and 8-bit attention
projections already cost about one WER point. NVFP4 would add one dynamic
quantize kernel per projection and a larger accuracy risk for a small speed
gain.
