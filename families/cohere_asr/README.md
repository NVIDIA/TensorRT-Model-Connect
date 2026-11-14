# Cohere Transcribe Arabic

This family builds `CohereLabs/cohere-transcribe-arabic-07-2026`, pinned to
`c3e911b42149bf7a1e53d5cef9878aee87515a23`, through TensorRT's strongly typed
Network API. It owns the 48-layer Conformer encoder, eight-layer decoder,
checkpoint mapping, native audio frontend, prompting, greedy generation and
UTF-8 detokenization. It imports no sibling family implementation.

The semantic Task is `speech_transcription`, available through CLI `transcribe`
and the public C and C++ SDKs. The default source language is Arabic (`ar`);
the family also accepts the checkpoint's English prompt (`en`). Translation,
diarization, timestamps and automatic language detection are not exposed.

## Build and run

Use the repository's prepared development environment and install only the
family's optional requirements. The model is gated on Hugging Face; the account
used to download it must already have access.

```bash
python -m pip install -r families/cohere_asr/requirements.txt
export PYTHONPATH=core/builder:apps/benchmark:.
python -m tensorrt_model_connect build \
  CohereLabs/cohere-transcribe-arabic-07-2026 \
  --revision c3e911b42149bf7a1e53d5cef9878aee87515a23 \
  --precision fp32 --max-sequence-length 256 --output cohere-arabic.bundle
```

A prepared local directory containing the pinned `config.json`,
`model.safetensors`, `tokenizer.json` and `tokenizer_config.json` can replace the
model ID. Runtime output does not require Python, PyTorch or a checkpoint path.
The bundled window and mel filters come directly from the checkpoint.

In a configured native build with tests enabled:

```bash
cmake --build build --parallel 12 --target \
  trtmc trtmc_backend_trt trtmc_model_cohere_asr \
  test_cohere_asr_sdk_c test_cohere_asr_sdk_cpp test_cohere_asr_frontend_contract
build/trtmc transcribe cohere-arabic.bundle --runtime-root "$PWD/build" \
  --input /path/to/arabic.wav --source-language ar
```

## Supported request boundary

The initial implementation uses FP32 arithmetic with TF32 disabled. Original
BF16 checkpoint values are expanded to FP32. Other precision and quantization
requests are rejected.

Provide mono float32 PCM at 16 kHz, between 512 samples and 35 seconds. The
native frontend reproduces deterministic length-seeded dither, preemphasis,
zero-padded centered STFT, checkpoint log-Mel filters and per-feature sample
normalization. It removes the final masked STFT frame before encoding; this
preserves the valid positions of the reference's masked Conformer computation.
The runtime rejects other sample rates and channel counts rather than silently
resampling or mixing them.

Decoder prefix computation is autoregressive without a KV cache. Build
`max_sequence_length` includes the nine-token prompt and generated tokens;
the default is 256 and supported bounds are 16 through 1024. Decoding must reach
EOS within that capacity; otherwise the runtime reports an error. Batching,
distributed execution, streaming and automatic long-audio chunking are outside
this implementation.

## Validate

The direct E2E uses `examples/sample1.wav` and `examples/sample2.wav` from the
pinned checkpoint. These recordings remain in the downloaded checkpoint; they
are not copied into the source repository. The oracle imports the checkpoint's
own supplied PyTorch model, feature extractor and tokenizer, loads its weights
strictly, and independently computes greedy transcription in FP32. It requires
exact native/reference token IDs and UTF-8 text through CLI, C SDK and C++ SDK.
The C++ consumer also repeats a recording after another request through the same
loaded Task. Frontend comparison requires relative L2 below `1e-4` and maximum
absolute error below `1e-3`.

```bash
export TRTMC_BINARY="$PWD/build/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build"
export TRTMC_NATIVE_BUILD_DIR="$PWD/build"
export TRTMC_COHERE_ASR_MODEL_DIR=/path/to/pinned/checkpoint
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/cohere_asr/tests --e2e-model cohere_asr -q
ctest --test-dir build -R '^cohere_asr_' --output-on-failure
```

The checkpoint directory for E2E also needs the supplied reference `.py` files,
`tokenizer.model`, and sample recordings. Set `TRTMC_COHERE_ASR_BUNDLE` only to
reuse an already-built bundle during validation. Tests still execute real native
inference and the complete reference comparison. Container users may need a
writable `NUMBA_CACHE_DIR` for librosa and `TORCHINDUCTOR_CACHE_DIR` for PyTorch.
These tests establish the declared Arabic samples, not dataset-level WER,
English accuracy, performance, deployment qualification or other hardware and
TensorRT cohorts.
