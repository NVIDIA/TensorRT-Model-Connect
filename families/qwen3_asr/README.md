# Qwen3-ASR

Speech transcription (`speech_transcription`) for Transformers-native (`-hf`) `Qwen3ASRForConditionalGeneration`
checkpoints, validated on [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) at revision
`7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`. The original releases without `-hf` (configuration under
`thinker_config`) are not claimed.

## What the bundle contains

| Section | Content |
|---|---|
| `audio_encoder.plan` | Chunked audio encoder and projector. Input `[windows, 8, 128, 100]` log-mel chunks plus the valid tokens of every chunk; output `[windows, 104, hidden]` projected rows. Attention stays inside windows of 8 chunks (104 tokens), as in the reference. |
| `decoder.plan` | Qwen3 decoder with an in-place KV cache of `max_sequence_length` positions (`IKVCacheUpdateLayer`). Profile 0 appends up to 1024 prompt tokens per launch, profile 1 one token; audio rows replace the `<|audio_pad|>` embeddings. |
| `mel_filters.bin` | Slaney filterbank, 201 x 128. |
| `vocab.bin` | Bytes of every token id, for detokenization without a BPE encoder. |
| `runtime.json` | Geometry, prompt ids, forced-language prefills, stop ids. |

The runtime computes the Whisper-style log-mel features of `Qwen3ASRFeatureExtractor`, encodes the clip (up to 8
windows, 64 s of audio, per launch), prefills the chat prompt (up to 1024 tokens per launch), decodes greedily (as
`generate(do_sample=False)`) and returns the transcript the official processor's
`return_format="transcription_only"` returns, with the generated token ids. Activation memory per launch does not
grow with `max_sequence_length`; the KV cache does (112 KiB per position in BF16).

## Use

```bash
trtmc build Qwen/Qwen3-ASR-0.6B-hf --output qwen3-asr-0.6b.bundle --max-sequence-length 4096
trtmc transcribe qwen3-asr-0.6b.bundle --input speech.wav [--source-language en] [--max-output-tokens 512]
```

- Audio: 16 kHz PCM, any channel count (channels are averaged). The family ships no resampler.
- `--source-language` forces the language as the official processor does (prefill `language <Name><asr_text>`);
  codes or names of the 30 languages the processor lists, case-insensitive. Without it the model detects the
  language.
- Config field `max_output_tokens` (default: the checkpoint's `max_new_tokens`, 512) counts the generated language
  tag as well.
- Length: the prompt (15 ids + 13 audio tokens per second of audio) plus `max_output_tokens` must fit
  `max_sequence_length` (default 4096: about 275 s of audio with 512 output tokens). Longer requests are rejected.
- Precision: `bf16` (default, the checkpoint's) or `fp32`.

## Validation

- `tests/test_engine_parity.py`: a small random model in BF16 and FP32 against the official modules (encoder with a
  partial window; decoder prefill in two launches and decode, with NaN-filled caches). The FP32 bound is 2e-5
  relative error; 3.5e-7 was measured.
- `tests/test_e2e.py`: build, native CLI, C and C++ SDK consumers and official greedy generation on a LibriSpeech
  clip; BF16 cases (automatic and forced English) and an FP32 case that must match the official token ids exactly.
- `tests/cpp/test_runtime.cpp`: the native front end, transcript parsing and prompt helpers against values produced
  by the official code.

On a GB200 with TensorRT 11.4: token ids and transcripts identical to the official model for the E2E clip (77
tokens) in BF16 and in FP32, and, against the official BF16 model, for clips of 0.3 s, 1 s, 23 s stereo, 71 s, 143 s
and 261 s, a Chinese clip, an output limit and a forced language. For the 23 s clip the decoder prefill (318
tokens) takes 11 ms and each generated token about 2 ms; the front end and encoder are reported separately as
`setup_ms`.

## Not built

- The forced-aligner head (`Qwen3ASRForTokenClassification`) and word timestamps.
- Context / hot-word system prompts (the official processor's `prompt`), batching and streaming.
