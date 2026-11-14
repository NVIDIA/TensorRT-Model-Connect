# HuBERT CTC speech transcription

This family builds `HubertForCTC` checkpoints, such as
[`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft),
into one TensorRT engine on the existing `speech_transcription` Task: a 16 kHz
waveform goes in, CTC logits come out, and the family runtime decodes them
greedily into the transcript and its token ids.

## What is built

- Identity: `architectures` contains `HubertForCTC`. The pretrained encoder
  (`HubertModel`) and the sequence classifier share the model type but have no
  transcription head and are not claimed.
- Both layouts of the model type, read from the config with Transformers'
  `HubertConfig` defaults for omitted keys:
  - large / xlarge: layer norm on every feature-encoder convolution, pre-norm
    ("stable") encoder blocks, encoder norm after the blocks;
  - base: group norm on the first convolution only, post-norm blocks, encoder
    norm before the blocks;
  - with or without convolution bias and the feature-projection layer norm
    (`feat_proj_layer_norm`, default on).
- The weight-normed positional convolution (`v * g / ||v||`, one scale per
  kernel tap) with the reference's trim of the last frame.
- Refused by name: `conv_pos_batch_norm`, adapters (`add_adapter`,
  `adapter_attn_dim`), activations other than the exact GELU, odd positional
  kernels, tokenizer settings under which the reference decodes other text
  (`do_lower_case`, a `replace_word_delimiter_char` other than a space,
  `clean_up_tokenization_spaces`, a `tokenizer_class` other than
  `Wav2Vec2CTCTokenizer`), and any checkpoint whose tensors are not exactly the
  resolved layout's (missing, unexpected or misshaped tensors).
  `masked_spec_embed`, the training-only SpecAugment vector, is the one tensor
  that is present and not read.

## Checkpoint format

The builder reads the checkpoint's single-file `pytorch_model.bin`, the format
the official HuBERT CTC checkpoints ship, with `torch.load(...,
weights_only=True)` on the CPU. That needs PyTorch 2.10.0 or newer, which
carries the
[restricted-loading security fix](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
The positional kernel may be stored as `weight_g` / `weight_v` or, as current
torch saves it, as `parametrizations.weight.original0` / `original1`; both
load to the same kernel. A checkpoint published only as `model.safetensors`,
and sharded checkpoints, are not read.

## Input and output

- Mono or interleaved multi-channel float32 PCM at the checkpoint's declared
  rate (16 kHz); channels are averaged. No resampler: another rate is refused.
- The engine takes the clip's true sample count. The base layout's first norm
  runs over the time axis, so a padded clip is a different clip; there is no
  padding and no attention mask. The default profile accepts 4,000 to 480,000
  samples (0.25 to 30 s); `max_sequence_length` sets the longest clip, in
  samples.
- The waveform is normalised to zero mean and unit variance (the processor's
  `do_normalize`) when the checkpoint states it.
- Greedy CTC as the reference tokenizer decodes it: argmax per frame, collapse
  repeats, drop the blank (the tokenizer's pad token, which must be the
  config's `pad_token_id`), map the word delimiter to a space and strip the
  ends. Token ids keep every non-blank token. The vocabulary decides the
  language; a `source_language` hint changes nothing.

## Precision

fp32 is the default and the checkpoint's stored precision. `precision=fp16`
builds the convolutions, projections, attention and CTC head in fp16 and keeps
every normalisation in fp32; the engine still reads fp32 audio and writes fp32
logits. TF32 is disabled in both builds.

fp16 does not promise the reference transcript on every clip. Measured on all
2,620 utterances of LibriSpeech `test-clean` (5.40 h, 1.3 to 35.0 s; bundles
built with `max_sequence_length=560000`, because nine clips are longer than
the default 30 s) against Transformers 5.2.0 `HubertForCTC` in fp32 with TF32
disabled, on the same samples, TensorRT 11.4.0.106 on a B300 SXM6. Transcripts
and WER come from the native runtime (the family's C++ SDK consumer); logits
come from the same bundle's plan run through the TensorRT Python runtime on the
processor's own input values:

| | fp32 engine | fp16 engine |
|---|---|---|
| WER against the LibriSpeech text (Transformers fp32: 2.08%) | 2.08% | 2.08% (+0.006 points) |
| Transcripts identical to Transformers fp32 (CTC ids identical) | 2,620 (2,620) / 2,620 | 2,605 (2,604) / 2,620 |
| Logit cosine, mean / minimum per clip | 1.0000000 / 0.9999999995 | 0.99994 / 0.99979 |
| Frames whose argmax differs (of 970,953) | 6, all at a reference top-2 gap of at most 0.0012 | 2,685 (0.28%) |
| Non-finite logits | 0 | 0 |

The official model run in fp16 (`model.half()`) on the same clips keeps the
fp32 CTC ids on 2,609 of 2,620 clips, with logit cosines of 0.99995 (mean) and
0.99979 (minimum); like the fp16 engine, it also changes frames whose fp32
top-2 gap is as large as 4.5. fp16 accuracy depends on the tactics TensorRT
picks in a given build: two other fp16 builds of the same network kept the CTC
ids on 2,590 and 2,595 clips (mean cosine 0.99985), and five builds inspected
with the engine inspector ran 75 to 99 of their 106 GEMM and convolution
layers with fp16 accumulation and gave the same mean cosine on 300 clips
(0.99984), which is consistent with fp16 accumulation causing most of the
deviation beyond the official fp16 run.

GPU time of one clip (ms), batch 1, B300 SXM6; engines: median of 7 rounds of
20 runs; Transformers: median of 7 rounds of 5 runs:

| Clip | fp32 | fp16 | Transformers fp32 eager |
|---|---|---|---|
| 2 s | 5.04 | 1.12 | 8.11 |
| 5 s | 6.59 | 1.23 | 9.75 |
| 15 s | 18.52 | 2.23 | 20.15 |
| 30 s | 46.25 | 4.07 | 35.32 |

The fp16 plan is half the size of the fp32 plan (0.60 against 1.18 GiB).

## Validation

`tests/test_model.py` runs the family's dataflow on the CPU, with the
TensorRT helpers replaced by the matching torch operations, against
Transformers' own `HubertForCTC` on small random checkpoints of both layouts
(every parameter drawn, norms included), with controls showing that a wrong
positional trim or a dropped projection norm fails the comparison, and checks
the configuration and tensor-inventory refusals. With a GPU it also builds fp32
and fp16 engines of those checkpoints: fp32 logits match the reference within
1e-4, fp16 logits within a cosine of 0.999 with every reduction in fp32. The
C++ tests cover the decoder and the pipeline with a fake engine.

The E2E manifests build `facebook/hubert-large-ls960-ft` at revision
`ece5fabbf034c1073acae96d5401b25be96709d8` and require an exact transcript and
exact CTC ids against Transformers' fp32 `HubertForCTC`, through both the
native CLI and the C++ SDK consumer: fp32 on a 4.2 s and a 23.3 s clip, fp16 on
the 23.3 s clip, whose fp16 transcript is the reference's (as it is on 98.9 to
99.4% of `test-clean`, depending on the build).

Not covered: other HuBERT heads, batching, streaming, resampling, word or
character timestamps, language-model decoding.
