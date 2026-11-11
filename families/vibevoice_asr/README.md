# VibeVoice-ASR

This family builds `microsoft/VibeVoice-ASR` at revision
`d0c9efdb8d614685062c04425d91e01b6f37d944` (MIT) in its original checkpoint
layout (`architectures: [VibeVoiceForASRTraining]`) and serves the public
`SpeechTranscription` Task, one request at a time, on one GPU, at the
checkpoint's native BF16.

The bundle holds two engines: the acoustic and semantic sigma-VAE encoders with
their speech connectors, and a Qwen2 decoder with a TensorRT native KV cache.
The checkpoint ships no tokenizer; the build uses the checkpoint's own when it
ships `tokenizer_config.json` and otherwise the `Qwen/Qwen2.5-7B` tokenizer the
official demo passes to the processor (whose own default, Qwen2.5-1.5B, has the
same vocabulary), at revision `d149729398750b98c0af14eb82c78cfe92750796`.

## Behaviour

- Input: interleaved PCM at any sample rate (default 24 kHz), downmixed to mono,
  resampled to 24 kHz and loudness-normalized like the official processor. Audio
  of any length up to the bundle's `max_sequence_length` (default 65,536 tokens,
  about 60 minutes; a multiple of 512) is encoded exactly.
- Output: `text` is the segment contents joined by spaces; `segments` carry the
  start and end seconds of each segment. Speaker ids have no field in the shared
  result: `raw_output=true` returns the model's JSON segment list instead.
- Config: `max_output_tokens` (default: until end of text or a full cache),
  `context_info` (hotwords or background text, as the official `context_info`),
  `raw_output`. `source_language` and `--kv-cache-size` are rejected.
- The acoustic latent is the sigma-VAE mean. The official model adds random
  noise to it on every call; transformers exposes the same deterministic mode as
  `sample=False`.

```sh
trtmc transcribe vibevoice-asr.bundle --runtime-root build --input speech.wav \
  --set context_info="Speakers: Ada, Grace" --set raw_output=true
```

## Validation

The E2E compares the native runtime with the official `microsoft/VibeVoice`
code (`tests/reference-source.json`), which needs transformers 4.x while the
repository pins 5.x. Prepare the reference once, then select the E2E:

```sh
python3 families/vibevoice_asr/tests/prepare_reference.py /path/to/reference
export TRTMC_REFERENCE_SOURCE_DIR=/path/to/reference/VibeVoice
export TRTMC_VIBEVOICE_ASR_REFERENCE_PYTHON=/path/to/reference/venv/bin/python
TRTMC_BINARY=build/trtmc TRTMC_RUNTIME_ROOT=build TRTMC_NATIVE_BUILD_DIR=build \
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest \
  families/vibevoice_asr/tests/test_e2e.py --e2e-model vibevoice_asr -v
```

Both pinned Hugging Face snapshots must be cached (the checkpoint and the
tokenizer in the manifest's `hf_dependencies`). `test_e2e.py` documents the
oracles.
