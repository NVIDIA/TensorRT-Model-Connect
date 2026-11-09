# Wav2Vec2 CTC layouts

The family builds fp32 (default) or fp16 CTC graphs for group-normalized and per-frame
layer-normalized feature encoders, with or without convolution bias. Encoder
blocks retain the checkpoint's post-norm or stable pre-norm order. Stable
checkpoints apply the encoder-level normalization after the blocks.

## Precision

fp32 is the default. `precision=fp16` builds the convolutions, projections,
attention and CTC head in fp16 and keeps every normalization (feature-encoder
group or layer norm, feature-projection and encoder layer norms) in fp32; the
engine still reads fp32 audio and writes fp32 logits, so the runtime and bundle
format are unchanged. The fp32 network definition is unchanged.

Measured on TensorRT 11.4, batch 1, GPU time of one clip (ms), median of 7
interleaved rounds; B200 / B300:

| Checkpoint | Clip | fp32 | fp16 | Speedup |
|---|---|---|---|---|
| wav2vec2-base-960h | 2 s | 5.17 / 4.97 | 1.06 / 1.09 | 4.9x / 4.6x |
| wav2vec2-base-960h | 5 s | 5.61 / 5.36 | 1.17 / 1.19 | 4.8x / 4.5x |
| wav2vec2-base-960h | 30 s | 20.86 / 20.42 | 3.04 / 3.06 | 6.9x / 6.7x |
| wav2vec2-large-xlsr-53-japanese | 5 s | 6.68 / 6.42 | 1.08 / 1.18 | 6.2x / 5.4x |
| wav2vec2-large-xlsr-53-japanese | 30 s | 47.90 / 46.14 | 3.39 / 4.00 | 14.1x / 11.5x |

The plan is half the size (base 363 to 184 MiB, XLSR 1216 to 612 MiB).

Quality against the Transformers fp32 model (B200 / B300):

- wav2vec2-base-960h, 400 LibriSpeech test clips (200 clean, 200 other,
  46 min): WER against the reference text 5.70% / 5.72% for fp16, 5.68% for
  fp32 and for the Transformers model; 393 / 394 of 400 fp16 transcripts are
  identical to the Transformers transcript (fp32: 399), logit cosine at least
  0.9997.
- wav2vec2-large-xlsr-53-japanese, 200 JSUT clips (13 min): CER against the
  reference text 12.52% for fp16, fp32 and the Transformers model; 199 / 200 of
  200 fp16 transcripts identical to the Transformers transcript (fp32: 200),
  logit cosine at least 0.9996.
- No non-finite logit.

fp16 therefore does not promise the exact reference transcript that fp32 gives.
On the base checkpoint the E2E waveform has a frame whose two best tokens are
0.014 apart, and some fp16 builds change one character there, so the fp16
premerge case (`wav2vec2-xlsr-53-japanese-fp16`, exact transcript and CTC ids)
uses the XLSR checkpoint, whose closest frame on that waveform is 0.092 apart.

The existing `facebook/wav2vec2-base-960h` manifest remains unchanged. The new
`jonatasgrosman/wav2vec2-large-xlsr-53-japanese` manifest pins revision
`cf031e020336460d15a417eba710bbc5bb43be9a`: layer-normalized convolutions with bias,
24 stable encoder blocks, width 1024 and a 2341-token CTC head. The official
model card declares Apache-2.0.

The builder reads only single-file `model.safetensors`. For an older official
`pytorch_model.bin`, first run the explicit family conversion on CPU:

```sh
python -m families.wav2vec2.convert_checkpoint MODEL_DIR/pytorch_model.bin MODEL_DIR/model.safetensors
```

Conversion requires PyTorch 2.10.0 or newer, including the
[restricted-loading security fix](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
It uses `weights_only=True`,
validates a nonempty tensor dictionary and refuses to overwrite its destination.
It preserves tensor names and dtypes. The direct E2E performs this preparation
in a temporary directory without modifying the cached official checkpoint.
Sharded checkpoints and adapters are not implemented.

The checkpoint's CTC vocabulary determines the output language. A
`source_language` hint does not replace that vocabulary or select a different
language head. Input is true-length PCM at the checkpoint's declared sample
rate; no resampling or padded-batch attention mask is introduced. Rebuild the
bundle and family runtime to obtain the extended behavior; the bundle schema
and Task API are unchanged.

CPU tests compare production family dataflow with Transformers' own randomly
initialized small feature encoders and encoder blocks. TensorRT graph operations
are replaced by CPU operations for these tests. Native tests exercise the real
CTC pipeline with a fake engine and Japanese vocabulary. They do not establish
TensorRT engine correctness, official-checkpoint parity, Japanese recognition
quality or performance.

The new direct E2E uses the existing non-silent English waveform fixture and
requires an exact transcript and CTC-token match to the official Japanese
checkpoint. It fails on an empty reference transcript. This is an inference
parity workload, not a Japanese WER/CER evaluation; it has not been run locally.
