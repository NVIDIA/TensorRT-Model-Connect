# Wav2Vec2 CTC layouts

The family builds FP32 CTC graphs for group-normalized and per-frame
layer-normalized feature encoders, with or without convolution bias. Encoder
blocks retain the checkpoint's post-norm or stable pre-norm order. Stable
checkpoints apply the encoder-level normalization after the blocks.

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
