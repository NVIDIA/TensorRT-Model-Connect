# Voxtral speech transcription

The family builds `VoxtralForConditionalGeneration` (Voxtral Mini 3B) checkpoints exported in the MLX
affine-quantized format and runs them through the `speech_transcription` Task: native CLI, C SDK and C++ SDK.

Validated checkpoint: [`mzbac/voxtral-mini-3b-4bit-mixed`](https://huggingface.co/mzbac/voxtral-mini-3b-4bit-mixed)
at revision `d5803d1a8b3e75df4722fc721e3a7df9b57dc73d`: 4,676,271,104 parameters, 2.98 GiB as stored. It is an MLX
export of [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), which is
Apache-2.0; the export itself declares no license. The repository ships no weights: the tests read the checkpoint
from the Hugging Face cache.

## Model

- Audio encoder: Whisper-large-v3 layout, two GELU convolutions (128 mel bins, stride 1 and 2), learned positions,
  32 pre-norm layers of width 1280 (20 heads), final LayerNorm. One 30 s chunk is 3000 mel frames and 1500 encoder
  frames.
- Projector: four consecutive encoder frames are stacked (width 5120), then `linear_1`, GELU, `linear_2` give 375
  decoder embeddings per chunk.
- Decoder: Llama with 30 layers of width 3072, 32 query and 8 key/value heads of 128, SwiGLU 8192, RoPE base 1e8,
  RMSNorm, untied output head over the 131,072-token Tekken vocabulary.

## Checkpoint format

Every quantized linear stores `weight` (uint32 words holding a little-endian bit stream of 4- or 6-bit codes per
row), `scales` and `biases` (one value per row and group of 64 or 128 columns, float16 or bfloat16). The builder
reconstructs `w = scale * code + bias` in float32 and rounds it once to the engine precision; the float32 weights are
bit-identical to MLX's `mx.dequantize` with float32 scales. Convolution kernels are transposed from the MLX
`[out, kernel, in]` layout. The export's top-level `embed_tokens` copy is checked to be identical to the decoder's
table and not used. The build fails if any checkpoint tensor is left unconsumed.

Only MLX affine exports are accepted. Unquantized Transformers checkpoints such as `mistralai/Voxtral-Mini-3B-2507`
are refused because this family has not validated them; Voxtral Realtime (`voxtral_realtime`) is a different
architecture and is not claimed.

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
sum and the SwiGLU product are computed in float32 and cast back; logits are float32. TF32 is disabled for `fp32`
builds.

Measured on 1x NVIDIA B300, TensorRT 11.4, against the Transformers float32 model on the exactly dequantized
weights (LibriSpeech test clip, 23.3 s, `en`, 383-token prompt):

| | audio embeddings: max abs error / cosine | last prompt logits: max abs error |
|---|---|---|
| engine fp32 | 1.4e-4 / 1.0000000 | 8.2e-5 |
| engine fp16 | 0.30 / 0.99971 | 0.18 |
| engine bf16 | 0.40 / 0.99889 | 0.16 (0.23 decoding one row per launch) |
| Transformers fp16 | 0.098 / 0.99995 | 0.059 |
| Transformers bf16 | 0.49 / 0.99859 | 0.31 |

The largest audio-embedding errors sit in the zero-padded tail of the 30 s chunk. Maxima also vary between engine
builds, because tactic selection changes the rounding: another fp16 build of the same graph measured 0.11 overall
and 0.011 on the speech rows (bf16 0.031; Transformers fp16 / bf16 0.003 / 0.028). In every E2E case below, each
fp16 and bf16 token was the float32 model's argmax.

## Bundle

`encoder.plan` (audio tower and projector, 1 to `max_audio_chunks` chunks), `decoder.plan` (profile 0: prefill
launches of up to 512 rows; profile 1: one row; explicit `[max_sequence_length, 1024]` key and value caches per
layer), `embed_tokens.bin` (token embeddings in the build precision, gathered on the host), `mel_filters.f32`,
`vocabulary.bin` (token bytes) and `runtime.json`.

## Tests

- `tests/test_support.py` (CPU, no dependencies): checkpoint identity, configuration and quantization validation.
- `tests/cpp/test_voxtral_frontend.cpp` (CTest `voxtral_frontend`, CPU): FFT, log-mel framing against an independent
  restatement, downmix, UTF-8 replacement, vocabulary decoding and prompt layout.
- `tests/test_e2e.py`: builds each manifest's bundle, runs the native CLI and both SDK consumers (identical ids and
  text required) and checks them against `tests/reference.py`: the Transformers model with weights dequantized
  independently of the builder. `fp32` bundles must reproduce the float32 greedy ids and text; `fp16` / `bf16`
  bundles are checked teacher-forced: each native token and the final end-of-sequence token must be the float32
  argmax or within 0.2 logit of it.

```bash
python -m pip install -r families/voxtral/requirements.txt
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  PYTHONPATH=core/builder:apps/benchmark:. python -m pytest families/voxtral/tests --e2e-model voxtral -vv
```

| Manifest | Precision | Cases (premerge) |
|---|---|---|
| `voxtral-mini-3b-4bit-mixed` | fp16 | recording `en` (yes); LibriSpeech + recording + LibriSpeech, 53.8 s, two chunks, automatic language (yes) |
| `voxtral-mini-3b-4bit-mixed-fp32` | fp32 | LibriSpeech `en` (yes); two chunks (no); recording `en` cut at 5 tokens by `max_new_tokens` (no) |
| `voxtral-mini-3b-4bit-mixed-bf16` | bf16 | LibriSpeech, automatic language (no) |

The audio fixtures are byte-identical copies of repository fixtures; see `tests/data/README.md`.
