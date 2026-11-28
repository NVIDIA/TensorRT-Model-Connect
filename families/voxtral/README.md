# Voxtral speech transcription

The family builds the `VoxtralForConditionalGeneration` (Voxtral Mini 3B) checkpoint and runs it through the
`speech_transcription` Task: native CLI, C SDK and C++ SDK.

Validated checkpoint: [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
at revision `3060fe34b35ba5d44202ce9ff3c097642914f8f3` (Apache-2.0): 4,676,271,104 parameters, 8.71 GiB as stored
(BF16; the encoder positions F32). The repository ships no weights: the tests read the checkpoint from the Hugging
Face cache.

## Model

- Audio encoder: Whisper-large-v3 layout, two GELU convolutions (128 mel bins, stride 1 and 2), learned positions,
  32 pre-norm layers of width 1280 (20 heads), final LayerNorm. One 30 s chunk is 3000 mel frames and 1500 encoder
  frames.
- Projector: four consecutive encoder frames are stacked (width 5120), then `linear_1`, GELU, `linear_2` give 375
  decoder embeddings per chunk.
- Decoder: Llama with 30 layers of width 3072, 32 query and 8 key/value heads of 128, SwiGLU 8192, RoPE base 1e8,
  RMSNorm, untied output head over the 131,072-token Tekken vocabulary.

## Checkpoint format

The builder reads the Transformers files: `config.json`, `preprocessor_config.json`, `tekken.json` and the
safetensors shards named by `model.safetensors.index.json` (the Mistral-native `consolidated.safetensors` is not
read). Before anything is built, the shards must hold exactly the tensors the configuration implies, with their
shapes and a floating-point dtype (BF16, F16 or F32); convolution kernels use the PyTorch `[out, in, kernel]` layout.
Tensors are widened to float32 exactly and rounded once to the engine precision; non-finite values are refused, and
an `fp16` build refuses a weight beyond the float16 range.

Quantized exports (a `quantization` or `quantization_config` entry in `config.json`, such as MLX exports) are
refused. Voxtral Realtime (`voxtral_realtime`) is a different architecture and is not claimed. Only
Voxtral-Mini-3B-2507 is validated; another `voxtral` checkpoint whose configuration and tensors pass these checks
builds, but has not been compared with its reference.

## Task contract

- Input: PCM float32 at 16 kHz; several channels are averaged to one. Other sample rates are refused (the family
  has no resampler) and so is empty audio.
- The clip is zero-padded to whole 30 s chunks. The log-mel front end is the checkpoint's `WhisperFeatureExtractor`
  (reflect-padded STFT, periodic Hann window of 400, hop 160, its Slaney filter bank, `log10` floored at the
  clip-wide maximum minus 8, then `(x + 4) / 4`), implemented natively; the filter bank is stored in the bundle.
- Prompt (from the checkpoint's own `mistral_common` transcription encoder at build time):
  `<s>[INST][BEGIN_AUDIO]` + `[AUDIO]` x 375 per chunk + `[/INST]` + optional `lang:<code>` + `[TRANSCRIBE]`.
- `source_language`: one of `en`, `es`, `fr`, `pt`, `hi`, `de`, `nl`, `it` (the model card's languages). Without it
  the prompt carries no language tag and the model detects the language. Other codes are refused.
- Greedy decoding until `</s>` or `max_new_tokens` (Task config, I64; default: every KV-cache position the prompt
  leaves free). `token_ids` exclude the end-of-sequence token. The text is the Tekken decoding with special tokens
  ignored (each run of regular tokens decoded as UTF-8 with U+FFFD replacement) and a leading `lang:xx` tag
  removed, as the Transformers processor does.
- Timings: `setup_ms` (front end), `prefill_ms` (encoder and prompt prefill), `decode_ms`.
- Capacity: `max_sequence_length` (default 4096) fixes the KV cache and the longest clip:
  `(max_sequence_length - 9) // 375` chunks, ten 30 s chunks (300 s) at the default. Batch 1.

## Precision

`fp16` is the default; `bf16` and `fp32` are also built. Weights, activations and KV caches use the build
precision. Normalization statistics, attention scores and softmax, rotary products, GELU, the encoder's position
sum and the SwiGLU product are computed in float32 and cast back. The output projection runs in the build precision
and its logits are returned as float32. TF32 is disabled for `fp32` builds.

Measured on 1x NVIDIA B300, TensorRT 11.4, against the Transformers float32 model on the same checkpoint
(LibriSpeech test clip, 23.3 s, `en`, 383-token prompt):

| | audio embeddings: max abs error / cosine | last prompt logits: max abs error |
|---|---|---|
| engine fp32 | 1.5e-4 / 1.0000000 | 6.1e-5 |
| engine fp16 | 0.10 / 0.99994 | 0.18 (0.13 decoding one row per launch) |
| engine bf16 | 0.23 / 0.99935 | 0.19 (0.15 decoding one row per launch) |
| Transformers fp16 | 0.14 / 0.99990 | 0.082 |
| Transformers bf16 | 0.31 / 0.99939 | 0.23 |

The largest audio-embedding errors sit in the zero-padded tail of the 30 s chunk. Maxima also vary between engine
builds, because tactic selection changes the rounding: another fp16 / bf16 build of the same graphs measured 0.37
overall and 0.0092 / 0.026 on the speech rows (Transformers fp16 / bf16: 0.0035 / 0.037). On five calibration inputs
the official Transformers pipeline in fp16 and in bf16 picks the float32 argmax at every step.

## Bundle

`encoder.plan` (audio tower and projector, 1 to `max_audio_chunks` chunks), `decoder.plan` (profile 0: prefill
launches of up to 512 rows; profile 1: one row; explicit `[max_sequence_length, 1024]` key and value caches per
layer), `embed_tokens.bin` (token embeddings in the build precision, gathered on the host), `mel_filters.f32`,
`vocabulary.bin` (token bytes) and `runtime.json`.

## Tests

- `tests/test_support.py` (CPU, NumPy): checkpoint identity, configuration validation, and the checkpoint reader
  against synthetic safetensors shards (exact BF16 widening, refused layouts, dtypes and non-finite values).
- `tests/test_e2e_harness.py` (CPU): offline checkpoint resolution from the Hugging Face cache, E2E case selection,
  and the recorded checks.
- `tests/cpp/test_voxtral_frontend.cpp` (CTest `voxtral_frontend`, CPU): FFT, log-mel framing against an independent
  restatement, downmix, UTF-8 replacement, vocabulary decoding, prompt layout and `runtime.json` validation.
- `tests/test_e2e.py`: builds each manifest's bundle, runs the native CLI and both SDK consumers (identical ids and
  text required) and checks them against `tests/reference.py`: the Transformers model, loaded by Transformers from
  the same checkpoint and run in float32. `fp32` bundles must reproduce the float32 greedy ids and text; `fp16` /
  `bf16` bundles are checked teacher-forced: each native token and the final end-of-sequence token must be the
  float32 argmax or within 0.2 logit of it. At every precision the native text must equal the official processor's
  decoding of the native ids. Each case records its input audio, both transcripts and the checks for the E2E report.

```bash
python -m pip install -r families/voxtral/requirements.txt
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  PYTHONPATH=core/builder:apps/benchmark:. python -m pytest families/voxtral/tests --e2e-model voxtral -vv
```

| Manifest | Precision | Cases (premerge) |
|---|---|---|
| `voxtral-mini-3b-2507` | fp16 | recording `en` (yes); LibriSpeech + recording + LibriSpeech, 53.8 s, two chunks, automatic language (yes) |
| `voxtral-mini-3b-2507-fp32` | fp32 | LibriSpeech `en` (yes); two chunks (no); recording `en` cut at 5 tokens by `max_new_tokens` (no); ten chunks, 297.6 s, the largest encoder profile, 320 tokens up to cache position 4075 of 4096 (no) |
| `voxtral-mini-3b-2507-bf16` | bf16 | LibriSpeech, automatic language (no) |

The audio fixtures are byte-identical copies of repository fixtures; see `tests/data/README.md`.
