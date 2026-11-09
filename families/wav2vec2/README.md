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

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder projections of `facebook/wav2vec2-base-960h`
was built and measured on TensorRT 11.4 (B200, B300) and 11.1 (B200). It is not
shipped. The engine of this family is fp32; an fp16 build of the same graph
(convolutions, projections and attention in fp16, every normalization in fp32)
is 4.5x to 6.9x faster than fp32, and FP8 on top of that fp16 build adds only
1.01x to 1.02x on B200 and 1.05x to 1.08x on B300. The plan is 44% smaller than
the fp16 plan, but 13% of the transcripts change against the reference model and
the exact E2E transcript check fails. This section records the recipe, the method
and the numbers, so that the study does not have to be repeated. The FP8 code was
not merged; this family has no FP8 build.

### Summary and decision

TensorRT 11.4, batch 1, B200 / B300. FP8 is built on the fp16 graph.

| Item | fp32 (default) | fp16 graph | FP8 | FP8 vs fp16 |
|---|---|---|---|---|
| Latency, 5 s clip (ms) | 5.605 / 5.358 | 1.167 / 1.187 | 1.150 / 1.100 | 1.02x / 1.08x |
| Latency, 30 s clip (ms) | 20.861 / 20.418 | 3.044 / 3.058 | 2.973 / 2.873 | 1.02x / 1.06x |
| Plan size (MiB) | 362.5 | 183.6 / 185.1 | 103.4 | -44% |
| LibriSpeech test WER (Transformers fp32: 5.68%) | 5.68% | 5.70% / 5.72% | 5.58% / 5.58% | -0.1 point |
| Transcripts identical to the Transformers fp32 model | 399 of 400 | 393 / 394 of 400 | 348 of 400 | 45 / 46 more changed |
| E2E test waveform, exact transcript | yes | 1 character differs in some builds | no (4 frames change) | fails |
| Non-finite logits | 0 | 0 | 0 | none |

Decision: do not ship FP8 for wav2vec2.

- Speed: FP8 only touches the encoder GEMMs, which are a fifth to a quarter of the
  fp16 engine time. The feature encoder's first normalization (over every time
  step of the clip), the convolutions and the attention stay as they are, so the
  end-to-end gain is 1% to 8%. On B200 the gain is within 2%; the remaining gain
  is plan size, which alone is not a reason to ship.
- Quality: WER against the reference text does not get worse (5.58% against
  5.68%), but the FP8 engine changes 1.0% of the words of the reference model's
  transcripts, and the exact transcript and CTC id check of the E2E case fails on
  its test waveform.
- Against the fp32 default, FP8 in the fp32 graph is 1.3x to 1.4x faster, but the
  fp16 graph is far faster with almost no transcript change, so FP8 in the fp32
  graph is not useful either.

### Recipe that was measured

No FP8 or NVFP4 wav2vec2 checkpoint exists on the Hub and no official
quantization config exists for it. The recipe follows the self-quantization flow
of the other encoder families: quantize the transformer projections, keep the
convolutional front end, the norms and the output head.

| Choice | Decision |
|---|---|
| Quantized (72) | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `intermediate_dense`, `output_dense` of the 12 encoder layers |
| Not quantized | the 7 feature-encoder convolutions, the feature projection, the positional convolution, every norm, the attention matmuls and softmax, the CTC head |
| Weight scale | per tensor, amax / 448 of the checkpoint weight, at build time |
| Activation scale | static, per tensor: max(abs(input)) / 448 recorded with forward pre-hooks on the Transformers fp32 model over 192 LibriSpeech validation clips (96 clean, 96 other). ModelOpt `FP8_DEFAULT_CFG` records the same statistic but cannot wrap the weight-normalized positional convolution. |
| Graph | `input -> Quantize -> Dequantize -> MatMul <- Dequantize <- FP8 weight`, both Dequantize layers in the type of the graph (fp16 in the fp16 graph, fp32 in the fp32 graph); the dual-GEMM build route is turned off when TensorRT lists it |
| Scale table | one key per checkpoint: model config (width, layers, vocabulary) and the first three CTC head biases |

### Method

- Checkpoint: `facebook/wav2vec2-base-960h` at the manifest revision
  `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
- Speed: the family build path (dynamic clip length, one optimization profile up
  to 30 s), CUDA graph replay of one clip of 2, 5, 15 and 30 s, GPU time, median
  of 7 interleaved rounds (5 on TensorRT 11.1); all variants built and timed in one
  job per GPU.
- Quality: 400 LibriSpeech test clips (200 `clean`, 200 `other`, 46 min, at most
  29 s each, fixed random selection; 200 clips on TensorRT 11.1), disjoint from the
  validation clips used for calibration. Input normalized like the runtime. WER
  against the reference text, WER and character error against the transcript of
  the Transformers fp32 model, logit cosine, per-frame argmax agreement and a scan
  for non-finite logits.
- The E2E test waveform (`tests/data/speech_16k_mono.wav`) with the transcript and
  CTC ids of the Transformers fp32 model, as the premerge case checks it.

### Results

Latency (ms) and plan size; the last columns are fp16 time over FP8 time:

| GPU | TensorRT | Engine | 2 s | 5 s | 15 s | 30 s | plan MiB | vs fp16 (2 / 5 / 15 / 30 s) |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | fp32 | 5.169 | 5.605 | 10.228 | 20.861 | 362.5 | - |
| B200 | 11.4 | fp16 graph | 1.063 | 1.167 | 1.957 | 3.044 | 183.6 | - |
| B200 | 11.4 | FP8, fp16 graph | 1.051 | 1.150 | 1.923 | 2.973 | 103.4 | 1.01 / 1.02 / 1.02 / 1.02 |
| B200 | 11.4 | FP8, fp32 graph | 4.023 | 4.250 | 7.327 | 15.261 | 120.7 | 0.26 / 0.27 / 0.27 / 0.20 |
| B300 | 11.4 | fp32 | 4.970 | 5.358 | 9.913 | 20.418 | 362.6 | - |
| B300 | 11.4 | fp16 graph | 1.091 | 1.187 | 1.945 | 3.058 | 185.1 | - |
| B300 | 11.4 | FP8, fp16 graph | 1.007 | 1.100 | 1.850 | 2.873 | 103.4 | 1.08 / 1.08 / 1.05 / 1.06 |
| B300 | 11.4 | FP8, fp32 graph | 3.855 | 4.051 | 7.031 | 14.718 | 120.7 | 0.28 / 0.29 / 0.28 / 0.21 |
| B200 | 11.1 | fp32 | 5.181 | 5.595 | 10.302 | 21.075 | 362.5 | - |
| B200 | 11.1 | fp16 graph | 1.064 | 1.168 | 1.960 | 3.039 | 183.6 | - |
| B200 | 11.1 | FP8, fp16 graph | 1.031 | 1.126 | 1.892 | 2.946 | 103.8 | 1.03 / 1.04 / 1.04 / 1.03 |
| B200 | 11.1 | FP8, fp32 graph | 4.011 | 4.215 | 7.356 | 15.459 | 121.1 | 0.27 / 0.28 / 0.27 / 0.20 |

Quality, TensorRT 11.4, 400 clips (B200 and B300 give the same FP8 numbers):

| Engine | WER vs reference text | WER / CER vs Transformers text | identical transcripts | logit cosine mean / min | frame argmax agreement |
|---|---|---|---|---|---|
| Transformers fp32 | 5.68% | - | - | - | - |
| fp32 | 5.68% | 0.01% / 0.00% | 399 | 1.00000 / 0.99996 | 0.9997 |
| fp16 graph (B200 / B300) | 5.70% / 5.72% | 0.09% / 0.02% | 393 / 394 | 0.99995 / 0.99969 | 0.9982 |
| FP8, fp16 graph | 5.58% | 0.99% / 0.19% | 348 | 0.99731 / 0.99311 | 0.9778 |
| FP8, fp32 graph | 5.54% | 1.11% / 0.22% | 336 | 0.99728 / 0.99271 | 0.9774 |

TensorRT 11.1 (B200, 200 clips): WER 5.70% (fp32), 5.70% (fp16 graph), 5.65% (FP8,
fp16 graph), 5.76% (FP8, fp32 graph); 199, 197, 169 and 169 of 200 transcripts
identical to the Transformers transcript. No non-finite logit in any engine.

E2E test waveform: both FP8 engines change the argmax of 4 of 207 frames (3 on
TensorRT 11.1), logit cosine 0.9976, and the transcript and CTC ids change, so the
exact premerge check of this family fails for FP8.

Kernel time per group, 30 s clip, TensorRT 11.4, B200 / B300 (CUDA kernel time in
ms, profiler):

| Kernel group | fp16 graph | FP8, fp16 graph |
|---|---|---|
| norms with their fused pointwise work | 1.256 / 1.215 | 1.260 / 1.218 |
| convolutions | 0.705 / 0.678 | 0.713 / 0.676 |
| encoder and head GEMMs | 0.620 / 0.667 | 0.511 / 0.562 |
| fused attention | 0.479 / 0.454 | 0.455 / 0.429 |
| other | 0.109 / 0.033 | 0.108 / 0.038 |
| sum | 3.168 / 3.122 | 3.047 / 2.923 |

The GEMMs are 20% of the fp16 time at 30 s (25% at 5 s) and FP8 removes 18% of
their time. The largest single kernel is the feature encoder's first
normalization (a group norm over every time step of the clip), 1.2 ms at 30 s.

### Variants

| Variant | Result |
|---|---|
| fp32 Dequantize in the fp16 graph, TensorRT 11.4 | wrong output on B200 and B300: WER 99.96% / 100%, logit cosine 0.91; no FP8 GEMM kernel in the plan. TensorRT 11.1 builds it correctly (1.6% WER against the Transformers transcript on 50 clips). A single FP8 GEMM with the same Dequantize types is correct on both versions, so the fault needs the full graph. fp16 Dequantize (the recipe) is correct on both. |
| FP8 also on the feature-encoder convolutions and the feature projection (TensorRT 11.1, B300, scratch graph with a normalization-layer form of the group norm) | 1.08x (30 s) to 1.10x (5 s) at batch 1 against the fp16 scratch graph, against 1.05x to 1.09x for the encoder projections alone; 95 instead of 102 of 120 cropped clips identical to the Transformers transcript |
| One projection kind at a time (TensorRT 11.1, B300, scratch graph, 120 clips cropped to 5 s) | `out_proj` 115, `q/k/v` 113, `output_dense` 111, `intermediate_dense` 109 of 120 identical; no subset keeps the transcripts and still gives a measurable gain |

### NVFP4

NVFP4 of the same 72 projections was tried in a scratch graph (TensorRT 11.1,
B300, 120 clips cropped to 5 s): logit cosine 0.980, 5.1% WER against the
Transformers transcript, 71 of 120 transcripts identical. It is too lossy, and the
only measured speed gain (1.07x against the fp16 graph) needs batch 8, which this
family does not build.

### When to revisit

- When the family builds batches of clips: the GEMMs then take a larger share of
  the time.
- When the feature encoder's normalization and the convolutions get faster, so
  that the GEMMs are a larger share of the fp16 time (in the scratch graph, the
  group norm written as a normalization layer was 11% faster at 30 s but 7% slower
  at 5 s on 11.1).
- When a calibration keeps the transcripts within the fp16 drift (for example per
  channel weight scales or skipping the most sensitive projections); none of the
  measured subsets did.
