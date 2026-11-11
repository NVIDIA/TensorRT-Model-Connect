# WavLM CTC speech transcription

The family builds `WavLMForCTC` checkpoints for the existing
`speech_transcription` Task: one TensorRT engine reads the true-length waveform
and returns CTC logits; the runtime normalises the clip when the checkpoint asks
for it and decodes the logits greedily with the checkpoint's own vocabulary, as
`Wav2Vec2CTCTokenizer` decodes: repeats collapsed, the pad token dropped, the
word delimiter turned into a space and the ends of the text stripped.

## Layouts

WavLM keeps wav2vec2's convolutional feature encoder, feature projection,
weight-normed positional convolution and Transformer blocks, in both of their
layouts: the base checkpoints use a group-normalised feature encoder and
post-norm blocks, the large checkpoints a per-frame layer-normalised feature
encoder and stable pre-norm blocks with the encoder norm after the last block.

Every attention block adds a relative position bias to its scores:

- The key-minus-query offset of each pair of frames is mapped to one of
  `num_buckets` buckets: half of them per sign, the first half of those exact,
  the rest spaced logarithmically up to `max_bucket_distance`, beyond which every
  offset shares the last bucket. Only the first block has the bucket embedding;
  every block reuses its bias.
- Each block rescales that bias per head and query frame with a gate computed
  from the block's own attention input (after the pre-norm in the stable
  layout): `a * (b * c - 1) + 2`, where `a` and `b` are sigmoids of a projection
  of the head's slice of the input and `c` is a learned per-head constant.

The builder evaluates the buckets on the host, in the reference's float32
arithmetic, into a table with one row per offset from `-max_bucket_distance` to
`max_bucket_distance`; the engine clamps each offset into that range and gathers
its row, so the table does not grow with the clip. The gate's eight-way
projection, which the reference sums in two groups of four, is folded into one
two-way projection at build time.

The builder refuses, by name, every variant it does not build: adapters, a GELU
other than the exact one, an odd positional kernel, a vocabulary whose ids do
not cover the head, a tokenizer other than `Wav2Vec2CTCTokenizer` (the phoneme
tokenizer joins tokens with spaces), a tokenizer that lowercases or whose space
clean-up would change the text (`clean_up_tokenization_spaces` with `.`, `?`,
`!`, `,` or `'` in the vocabulary), and a pad token that disagrees with
`config.pad_token_id`. It reads the configuration before any weight, so a
checkpoint without a CTC head, such as the pretrained `microsoft/wavlm-large`,
resolves to this family through its `model_type` and is refused because it has
no transcription to build. Added tokens are read from `added_tokens.json`.

## Checkpoints

The direct E2E pins two LibriSpeech clean-100h fine-tunes, one per layout:

| Manifest | Checkpoint | Revision | Layout |
|---|---|---|---|
| `wavlm-libri-clean-100h-large` (fp32) and `-fp16` | `patrickvonplaten/wavlm-libri-clean-100h-large`, fine-tuned from `microsoft/wavlm-large` | `e70e3a062ec399c46008ee55d1fb52c7ba338d5c` | 24 stable blocks, width 1024, layer-normalised feature encoder, normalised input |
| `wavlm-libri-clean-100h-base-plus` (fp32) | `patrickvonplaten/wavlm-libri-clean-100h-base-plus`, fine-tuned from `microsoft/wavlm-base-plus` | `02c289c4471cd1ba4b0ff3e7c304afe395c5026a` | 12 post-norm blocks, width 768, group-normalised feature encoder, unnormalised input |

Both heads have 31 classes: the word delimiter `|` is id 0 and the CTC blank is
the tokenizer's `[PAD]`, id 28, so nothing in the runtime assumes the blank is
id 0. `microsoft/wavlm-large` and `microsoft/wavlm-base-plus` are distributed
under the license their model cards link (the UniSpeech repository, CC BY-SA
3.0); the two fine-tunes declare no license on the Hub.

The builder reads only single-file `model.safetensors`. The released WavLM
checkpoints ship `pytorch_model.bin`; convert it first on CPU:

```sh
python -m families.wavlm.convert_checkpoint MODEL_DIR/pytorch_model.bin MODEL_DIR/model.safetensors
```

Conversion requires PyTorch 2.10.0 or newer, uses `weights_only=True`,
validates a nonempty tensor dictionary, refuses to overwrite its destination and
preserves tensor names and dtypes. The direct E2E performs this preparation in a
temporary directory without modifying the cached checkpoint.

## Precision

fp32 is the default. `precision=fp16` builds the convolutions, projections,
attention, position bias and gates in fp16 and keeps every normalisation in
fp32; the engine still reads fp32 audio and writes fp32 logits, so the runtime
and bundle format are the same.

Measured on TensorRT 11.4 and one GB200, batch 1, GPU time of one clip (ms),
median of three interleaved rounds of 20 runs:

| Checkpoint | Clip | fp32 | fp16 | Speedup |
|---|---|---|---|---|
| wavlm-libri-clean-100h-large | 2 s | 5.34 | 1.37 | 3.9x |
| wavlm-libri-clean-100h-large | 5 s | 6.78 | 1.84 | 3.7x |
| wavlm-libri-clean-100h-large | 30 s | 45.85 | 14.20 | 3.2x |
| wavlm-libri-clean-100h-base-plus | 5 s | 5.70 | 1.55 | 3.7x |
| wavlm-libri-clean-100h-base-plus | 30 s | 21.06 | 6.23 | 3.4x |

The plan is half the size (large 1,208 to 606 MiB, base-plus 362 to 183 MiB).

Quality against the Transformers fp32 model on every LibriSpeech `test-clean`
clip of at most 30 s (2,611 clips, 319 min; the 9 longer clips are outside the
default profile). The engines ran from Python on the feature extractor's input
values, and both sides were decoded with the checkpoint's tokenizer; the native
normalisation and decoding are what the E2E and the C++ tests cover:

| Checkpoint | Engine | WER vs reference text | Transcripts identical to Transformers | Logit cosine (min) |
|---|---|---|---|---|
| large | Transformers fp32 | 5.80 % | - | - |
| large | fp32 | 5.80 % | 2,611 | 0.9999998 |
| large | fp16 | 5.81 % | 2,490 | 0.99981 |
| base-plus | Transformers fp32 | 7.24 % | - | - |
| base-plus | fp32 | 7.24 % | 2,611 | 0.9999998 |
| base-plus | fp16 | 7.24 % | 2,581 | 0.99989 |

No logit is non-finite. fp16 does not promise the reference transcript. On the
60 clips of 5 to 16 s with the widest top-2 gaps among those whose fp16
transcript already matched, the largest fp16 logit deviation of the large
checkpoint per clip had a median of 0.58 and reached 1.18; only one of those
clips had a smallest top-2 gap above its deviation. The E2E waveform has a frame whose two
best classes are 0.0007 apart. Over the 2,611 clips an fp16 transcript differs
from the Transformers transcript by at most two characters (large) or one
(base-plus), so the fp16 E2E case bounds the character error at 1 % instead of
requiring the identical transcript. Keeping the residual stream or the
attention in fp32 did not lower the largest relative error of the pretrained
encoder's fp16 output on nine clips (7.8 % to 8.1 % in every variant, and fp32
attention took 1.8x the time), so the fp16 graph carries neither.

The pretrained `microsoft/wavlm-large` encoder, built through this graph with
an identity head in place of the CTC head, reproduces Transformers'
`last_hidden_state` on 25 clips of up to 1,165 frames with a relative error of
at most 7.4e-5 and a per-frame cosine of at least 0.9999997 in fp32. Its fp16
output is far less exact: on the 23.3 s E2E waveform, two builds of the same
fp16 graph gave relative errors of 5.4 % and 7.8 %.

## Scope

Input is true-length mono PCM at the checkpoint's sample rate, one clip per
call, 4,000 to 480,000 samples (30 s at 16 kHz) by default; `max_sequence_length`
sets a different upper bound in samples. No resampling, batching or padded
attention mask is implemented. The `WavLMForSequenceClassification`,
`WavLMForXVector` and `WavLMForAudioFrameClassification` heads and the bare
encoder's hidden states are not built.

## Tests

- `tests/test_support.py`: identity and default Task, without dependencies.
- `tests/test_model.py`: configuration and vocabulary refusals, the bucket
  layout, the clamped bias table, the folded gate and the weight-normed kernel,
  each with a control where the rule could be broken silently.
- `tests/test_reference.py`: the family's dataflow against randomly initialised
  Transformers models on CPU (feature encoder, bias table, encoder with gate and
  bias, with controls for a gate on the wrong rows, an ungated bias and a
  transposed offset), the checkpoint conversion, and, on a GPU, real fp32 and
  fp16 engines of both layouts against the same models.
- `tests/cpp/`: the CTC decoder with a blank that is not id 0 and with
  delimiters at the edges, waveform normalisation and the pipeline's refusals,
  with a fake engine.
- `tests/test_e2e.py`: builds each manifest's bundle, runs the native
  `transcribe` command on a 23.3 s LibriSpeech test-clean utterance (6930-75918-0003,
  CC BY 4.0; 1,165 frames, beyond the 800-frame bucket range) and requires the
  exact transcript and CTC ids of the Transformers fp32 model; the fp16 case
  (`tests/thresholds/wavlm-libri-clean-100h-large-fp16.json`) allows a character
  error of at most 1 % against that transcript.
