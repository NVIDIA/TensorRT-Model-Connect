# Granite Speech 5 (CTC)

Speech transcription with IBM Granite Speech 5.0 TurboCTC (`GraniteSpeech5ForCTC`), validated on
[`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
at revision `b8e0718e5b6d70b2e079add8a07c7689c95b89f1`.

| | |
|---|---|
| Task | `speech_transcription` |
| Precision | `bf16` (default, the checkpoint's precision) or `fp32` |
| Input | 16 kHz float PCM; channels are averaged as the reference feature extractor does; other rates are refused (no resampler) |
| Length | one CTC frame (1,120 samples) up to `max_sequence_length` samples (default 480,000 = 30 s, at most 600 s) |
| Decoding | greedy CTC: the first largest logit per 80 ms frame, repeats collapsed, the blank (`<unk>`, id 0) dropped, then the tokenizer's decoding with special tokens skipped |

## Structure

- `encoder.plan` — waveform (the runtime appends the reference's zero sample when a clip is an odd number of whole
  hops, then reflect-pads it) to one token id per frame: the log-mel and delta front end in FP32 (strided DFT, HTK
  mel filters, log10 floored at the clip maximum minus 8, deltas, frame-pair stacking), the 16-block Conformer
  encoder (block-wise attention over 128 frames with Shaw relative positions; blocks 0 and 1 halve the frame rate
  with a stride-2 depthwise convolution next to a mean-pooled residual; the CTC posteriors of block 7 are fed back
  through `out_mid`) and the CTC head with a first-maximum argmax (a frame with a non-finite logit gets an id
  outside the vocabulary, which the runtime refuses).
- Matrix products run in the bundle's precision; normalisation, activations, the gated linear unit, the depthwise
  convolution with its batch norm, the attention scores and softmax, the residual stream and the logits run in FP32.
- `runtime/` — `trtmc_model_granite_speech5`: request checks, padding, the CTC collapse and the decoding of
  `tokenizer.json` (`▁` to a space, byte-fallback runs as UTF-8 with U+FFFD for invalid bytes, one leading space
  stripped), as `ParakeetTokenizer.decode(..., skip_special_tokens=True)` does.

Refused, because they would load and give a different transcript: an untied CTC head, a different feature extractor,
encoder or tokenizer decoder, grouped key/value heads, an even convolution kernel, repeated or out-of-range
`subsample_layers`, a pad token that differs between the model and the tokenizer, and special or added tokens that
the tokenizer configuration registers beyond `tokenizer.json`. `source_language` is refused: the model
transcribes the spoken language without a hint.

## Use

```bash
python -m tensorrt_model_connect build ibm-granite/granite-speech-5.0-470m-turboctc \
  --revision b8e0718e5b6d70b2e079add8a07c7689c95b89f1 -o granite-speech5.bundle --precision bf16
trtmc transcribe granite-speech5.bundle --input speech_16k_mono.wav
```

## Tests

- CPU: `test_support.py`, `test_config.py` (types, defaults, refusals, limits, frame counts against the official
  feature extractor), `test_frontend.py` (front-end constants and arithmetic against the official feature
  extractor), `test_vocabulary.py` (decoding against the checkpoint's `ParakeetTokenizer` and the tokenizers
  library; needs the cached checkpoint), `tests/cpp/test_runtime.cpp` and `tests/cpp/test_pipeline.cpp` (the
  runtime against a stand-in engine).
- GPU: `test_graph.py` (the front end, the Conformer blocks, the self-conditioning and the whole engine against the
  Transformers modules on a small random model, each with a negative control) and `test_e2e.py` (build,
  `trtmc transcribe`, the C and C++ SDK consumers, and the official model in FP32 on the GPU). FP32 bundles
  reproduce the ids and the transcript exactly; BF16 bundles stay within a 3 % edit distance of the FP32 reference
  on ids and characters. Every bundle also transcribes a stereo copy of its clip like the mono clip and refuses
  48 kHz audio, a clip shorter or longer than its limits, a non-finite sample, `source_language` and unknown options.

```bash
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/granite_speech5/tests --e2e-model granite_speech5 -v
```

Test audio: `ASSET_LICENSES.md`.
