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
same vocabulary), at revision `d149729398750b98c0af14eb82c78cfe92750796`. The
build reads it from the local Hugging Face cache and does not download it; fetch
it once with
`hf download Qwen/Qwen2.5-7B tokenizer.json tokenizer_config.json --revision d149729398750b98c0af14eb82c78cfe92750796`.

## Behaviour

- Input: interleaved PCM at any sample rate (default 24 kHz), downmixed to mono,
  resampled to 24 kHz and loudness-normalized like the official processor. Audio
  of any length is encoded exactly within the sequence budget below. A partial
  last frame is padded per strided layer, as the original code does; the
  transformers implementation pads the waveform instead, so its last frame can
  differ.
- Sequence budget: the bundle's `max_sequence_length` (default 65,536 tokens; a
  multiple of 512) holds the whole sequence: about 60 tokens of system and user
  text plus `context_info`, one speech token per 3,200 samples at 24 kHz (7.5
  per second, 27,000 for 60 minutes), and the generated transcript. A request
  whose prompt does not fit, or whose prompt plus `max_output_tokens` does not,
  is rejected before any audio processing, with the lengths in tokens.
  Generation stops at the end of text, at `max_output_tokens`, or, without
  `max_output_tokens`, when prompt and output fill the sequence. Size the bundle
  for the prompt plus the transcript you expect: 60 minutes leave about 38,000
  output tokens at the default, and the LibriSpeech test clip needs about 4.3
  output tokens per second of speech, JSON included.
- Output: `text` is the segment contents joined by spaces; `segments` carry the
  start and end seconds of each segment. Speaker ids have no field in the shared
  result: `raw_output=true` returns the model's JSON segment list instead. When
  generation stops at `max_output_tokens` or the sequence limit before the model
  closes the list, the transcript is partial: `raw_output=true` returns the
  decoded prefix as it is, which is not valid JSON (a character cut off at the
  end becomes U+FFFD), and the default output keeps the segments that were
  complete (with none, `text` is that prefix).
- Config: `max_output_tokens` (default: until end of text or a full sequence;
  a value must fit in the sequence with the prompt),
  `context_info` (hotwords or background text, as the official `context_info`),
  `raw_output`. `source_language` and `--kv-cache-size` are rejected.
- Tokenization follows the bundle's `tokenizer.json`, including its NFC
  normalizer and the Unicode letter and number classes of its Split pattern,
  which the runtime takes from the installed ICU library; text that is not
  valid UTF-8 is rejected.
- The acoustic latent is the sigma-VAE mean. The official model adds random
  noise to it on every call, and so does the transformers implementation; the
  E2E reference removes the noise with `vae_std = 0`.

```sh
trtmc transcribe vibevoice-asr.bundle --runtime-root build --input speech.wav \
  --set context_info="Speakers: Ada, Grace" --set raw_output=true
```

## Validation

The E2E compares the native runtime with the transformers implementation of the
model, which first shipped in transformers 5.3.0 (`tests/reference-source.json`)
while the repository pins 5.2.0. `tests/transformers_reference.py` runs it in a
child interpreter that imports transformers from a checkout of that source and
everything else from the current environment, on `microsoft/VibeVoice-ASR-HF`,
Microsoft's release of the same weights in the transformers layout (every tensor
equals the original checkpoint's), with the sigma-VAE mean:

```sh
git clone --branch v5.3.0 --depth 1 https://github.com/huggingface/transformers /path/to/transformers
export TRTMC_REFERENCE_SOURCE_DIR=/path/to/transformers
TRTMC_BINARY=build/trtmc TRTMC_RUNTIME_ROOT=build TRTMC_NATIVE_BUILD_DIR=build \
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest \
  families/vibevoice_asr/tests/test_e2e.py --e2e-model vibevoice_asr -v
```

The three pinned Hugging Face snapshots must be cached: the checkpoint and the
manifest's `hf_dependencies` (the tokenizer the build uses and the reference
weights). `TRTMC_VIBEVOICE_ASR_MODEL_DIR`, when set, must hold the pinned
checkpoint, because the reference always loads the pinned weights;
`TRTMC_VIBEVOICE_ASR_REFERENCE_PYTHON` selects another interpreter for the
reference. `test_e2e.py` documents the oracles.
