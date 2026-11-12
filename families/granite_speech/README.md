# Granite Speech

Speech transcription with IBM Granite Speech (`GraniteSpeechForConditionalGeneration`), validated on
[`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) at revision
`bd87ab862416353633ea431fe49b1614003623c5`.

| | |
|---|---|
| Task | `speech_transcription` |
| Precision | `bf16` (default, the checkpoint's precision) or `fp32` |
| Input | 16 kHz mono float PCM; other rates and channel counts are refused (no resampler) |
| Prompt | the model card's: `USER: <|audio|>can you transcribe the speech into a written format?\n ASSISTANT:` |
| Decoding | greedy, first maximum, until EOS or `max_new_tokens` (Config field, default 200 as in the model card) |
| Context | `max_sequence_length` positions (default 4,096 = `max_position_embeddings`); the prompt, one placeholder per 0.1 s of audio and `max_new_tokens` must fit, so a 4,096-position bundle accepts clips up to 387.6 s with the default 200 new tokens (407.7 s with `max_new_tokens=1`) |

## Structure

- `audio.plan` — waveform (reflect-padded by the runtime) to audio embeddings: the log-mel front end in FP32
  (strided DFT, HTK mel filters, log10, clamp at the clip maximum minus 8, frame stacking), the 16-block Conformer
  encoder (block-local attention with Shaw relative positions, depthwise convolution with folded BatchNorm, the
  mid-layer CTC self-conditioning) and the Q-Former projector (3 queries per 15-frame window).
- `decoder.plan` — the Granite 4.0 1B language model with two profiles, prefill (the whole prompt) and decode (one
  token); the runtime binds every layer's past and present keys and values onto one device cache.
- `runtime/` — `trtmc_model_granite_speech`: request checks, prompt assembly, greedy decoding and byte-level
  detokenisation (special tokens skipped, invalid UTF-8 repaired as the reference tokenizer does).

Refused, because they would load and give a different transcript: LoRA-adapter checkpoints, scaled rotary
embeddings (`rope_scaling`, non-default `rope_parameters`), tied language-model heads, a different feature extractor
or encoder, a Q-Former with text input. `source_language` is refused: the model transcribes the spoken language
without a hint.

## Use

```bash
python -m tensorrt_model_connect build ibm-granite/granite-4.0-1b-speech \
  --revision bd87ab862416353633ea431fe49b1614003623c5 -o granite-speech.bundle --precision bf16
trtmc transcribe granite-speech.bundle --input speech_16k_mono.wav --set max_new_tokens=200
```

## Tests

- CPU: `test_support.py`, `test_config.py` (geometry, refusals, inventory, sample limits), `test_constants.py`
  (front end and index algebra against the official feature extractor and modules), `test_prompt.py` (prompt ids and
  detokenisation against the official tokenizer; needs the cached checkpoint), `tests/cpp/test_runtime.cpp`.
- GPU: `test_graphs.py` (both engines against the Transformers modules on a small random model, FP32 to 1e-4) and
  `test_e2e.py` (build, `trtmc transcribe`, the C and C++ SDK consumers, and the official model in FP32 on the GPU:
  generated ids and transcript identical). Its cases cover BF16 and FP32 bundles, a clip cut at 7.3 s, the clip
  played twice, a 12-token limit and a 1,024-position bundle; on every bundle it also checks that the CLI refuses
  48 kHz and stereo audio, a too-short clip, `source_language`, `max_new_tokens=0` and a request over the capacity.

```bash
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/granite_speech/tests --e2e-model granite_speech -v
```
