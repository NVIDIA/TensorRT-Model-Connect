# Parakeet TDT 0.6B v3

This family targets the native Hugging Face Transformers checkpoint
`nvidia/parakeet-tdt-0.6b-v3` at revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b` (publisher license: CC BY 4.0).
It exposes the public `SpeechTranscription` Task for one audio request at a time,
with FP16 or FP32 TensorRT engines and a fixed 30-second input window. Runtime
input accepts interleaved PCM, downmixes channels, and resamples to 16 kHz.
Longer audio, language controls, streaming and multi-GPU execution are rejected
or not exposed. NeMo archives are not a build input for this port.

The migration has CPU contract coverage; GPU conversion and transcript parity
remain unverified. Do not interpret the presence of the manifest as qualification.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#1948). The feed-forward matrices (`ff1`, `ff2`) and the attention
projections (`w_q`, `w_k`, `w_v`, `w_o`) of the 24 encoder layers get
per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (192 LibriSpeech validation clips). The convolution
stem and module, the relative-position projection, the predictor and the joint
network stay fp16. B200, batch 1, 30 s window: the bundle is 40% smaller
(1.165 to 0.698 GiB), the encoder is 1.041x (TensorRT 11.4) and 1.048x (11.1)
faster than fp16, and WER on 200 held-out LibriSpeech test clips moves by
0.21 point or less.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. The 0.6B encoder runs at batch 1
(the family accepts batch 1 only), and FP8 on about 83% of the encoder weights
gives only a small speed gain (numbers from #1948, B200):

| Item | Value |
|---|---|
| fp16 encoder call, batch 1, 30 s window | 4.359 to 4.384 ms |
| FP8 encoder call, batch 1, 30 s window | 4.184 to 4.187 ms |
| FP8 vs fp16, batch 1, 30 s window | 1.041x (11.4), 1.048x (11.1) |
| FP8 vs fp16, feed-forward only, batch 1 to 32, 10 s and 30 s (measurement-only builds) | 1.03x to 1.10x |
| Bundle, fp16 to FP8 | 1.165 to 0.698 GiB (-40%) |

The main FP8 result is bundle size, not speed: halving the bytes of those
weights gives about 4% at batch 1. NVFP4 would add one dynamic quantize kernel
per projection for at most a small further gain.

## Validation

Install this family's `requirements.txt` in its isolated build/reference
environment. The pinned Transformers version provides `AutoModelForTDT`, which
is not available in the repository's base Transformers 5.2.0 environment.

Build with `TRTMC_BUILD_TESTS=ON` to produce `test_parakeet_tdt_sdk_cpp`.
After caching the pinned Hugging Face snapshot, select the E2E explicitly:

```sh
TRTMC_RUNTIME_ROOT=/path/to/runtime \
TRTMC_NATIVE_BUILD_DIR=/path/to/native/build \
PYTHONPATH=core/builder:. python -m pytest families/parakeet_tdt/tests/test_e2e.py \
  --e2e-testcase parakeet-tdt-0.6b-v3 -v
```

`--e2e-model parakeet_tdt` selects all eight audio cases. An explicit selection
fails when dependencies, artifacts or CUDA are unavailable; an ordinary CPU
test run reports these E2Es as skipped. The test invokes the public C++ SDK
consumer, records native/reference evidence, and requires exact transcript
agreement after the original case/whitespace normalization. It does not relax
the old PR's exact-transcript gate to a shape-only or approximate comparison.

`TRTMC_PARAKEET_TDT_MODEL_DIR` may point to a prepared copy of that same pinned
snapshot. Record its provenance when using the override. The tests do not
download model weights automatically. Audio fixture generation is family-local.
