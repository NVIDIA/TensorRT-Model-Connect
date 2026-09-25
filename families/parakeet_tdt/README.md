# Parakeet TDT 0.6B v3

This family targets the native Hugging Face Transformers checkpoint
`nvidia/parakeet-tdt-0.6b-v3` at revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b` (publisher license: CC BY 4.0).
It exposes the public `SpeechTranscription` Task for one audio request at a time,
with FP16 or FP32 TensorRT engines and a fixed 30-second input window. Runtime
input accepts interleaved PCM, downmixes channels, and resamples to 16 kHz.
Longer audio, language controls, streaming, quantization and multi-GPU execution
are rejected or not exposed. NeMo archives are not a build input for this port.

The migration has CPU contract coverage; GPU conversion and transcript parity
remain unverified. Do not interpret the presence of the manifest as qualification.

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
