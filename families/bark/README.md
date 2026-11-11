# Bark

This family builds Suno Bark (`suno/bark-small` and `suno/bark`) for
`text_to_audio`. One bundle holds four TensorRT engines: the semantic, coarse and
fine GPT-style stages (12 layers, hidden size 768 for bark-small) and the EnCodec
decoder. The C++ runtime runs the semantic and coarse stages token by token with
a KV cache, then the fine stage in one pass and the codec; it does not use CUDA
graphs. The manifests build bark-small in fp16 and fp32 and bark large in fp32,
with tensor parallel size 1 and 4. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the transformer projections of bark-small was built and
measured on TensorRT 11.1 and (spot check) 11.4. It is not shipped: the decode
step, which is most of the generation time, is 0.99x to 1.04x of fp16 with CUDA
graphs, and audio quality drops (log-mel distance to the fp32 audio 2.9x the fp16
floor). The bundle is 22% smaller. NVFP4 is only 1.02x on B300 and gives NaN
logits in the prefill. This section records the recipe, the method and the
numbers, so that the study does not have to be repeated. The FP8 code was not
merged; this family has no FP8 build. All numbers come from a local study.

### Summary and decision

| Item | fp16 (manifest) | FP8 | FP8 vs fp16 |
|---|---|---|---|
| Decode step, CUDA graph, B200 / B300 (TensorRT 11.4) | 0.566 / 0.526 ms | 0.542 / 0.529 ms | 1.04x / 0.99x |
| Decode step, eager enqueue, B200 / B300 (TensorRT 11.4) | 0.679 / 0.665 ms | 0.567 / 0.559 ms | 1.20x / 1.19x |
| Decode step, CUDA graph, B300 (TensorRT 11.1) | 0.533 ms | 0.517 ms | 1.03x |
| Bundle size | 1006.8 MiB | 785.7 MiB | -22% |
| Whisper WER of the audio, 60 sentences (PyTorch, fp32 0.081) | 0.082 | 0.089 | +0.007 |
| Log-mel L1 distance to the fp32 audio | 0.82 | 2.38 | 2.9x |
| NaN or Inf in the checked engine logits | none | none | none |

Decision: do not ship FP8 for Bark.

- Speed: at batch 1 the decode step is latency bound. One step runs 124 small
  kernels of about 4 to 5 us each; the fp16 weights read per step (about 170 MB)
  take about 21 us at full memory bandwidth, 4% of the step. FP8 can only
  shorten that part.
- The 1.19x to 1.20x eager gain comes from fewer kernel launches (100 instead of
  124 per step), not from faster math. The same step in fp16 with a CUDA graph
  (0.53 to 0.57 ms) is faster than the FP8 eager step, without a quality loss.
- Quality: the audio of the FP8 model moves away from the fp32 audio clearly
  more than fp16 does (WER against the fp32 transcripts 0.109 vs 0.066).
- The only gain without a cost is bundle size (221 MiB less).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of Bark exists on the Hub, and no official
quantization config exists for it. The recipe is the usual transformer
one: quantize the attention and MLP projections, keep embeddings, norms, heads
and the codec in fp16. Each quantized projection is
built as `input -> Quantize -> Dequantize -> MatMul`, with the weight
`FP8 constant -> Dequantize`.

| Choice | Decision |
|---|---|
| Quantized (108 projections) | `w_q`, `w_k`, `w_v`, `w_fc1`, `w_fc2` of every layer of the semantic, coarse and fine stages (12 layers each in bark-small) |
| fp16 | attention output projections `w_o` (TensorRT 11.4 cannot compile an FP8 Quantize right after the fused attention: "element type (fp8_e4m3) differs from input 1's element type (half)"), embeddings, layer norms, LM heads, the attention itself and the KV cache, the EnCodec decoder |
| Weight scale | per tensor, amax / 448 of the weight, at build time |
| Activation scale | static, per tensor: amax / 448 of the projection input, recorded with forward pre-hooks while the Hugging Face model synthesizes 48 LibriSpeech validation sentences (all three stages); `w_q`, `w_k` and `w_v` share one scale. ModelOpt `FP8_DEFAULT_CFG` records the same statistic but cannot wrap Bark's three chained `generate` stages. Scales stored in a JSON file keyed by the model config |
| Dequantize type | fp32 with a cast back to fp16 (the measured FP8 build); fp16 Dequantize measured as a variant |
| Build option | TensorRT's fused dual-GEMM pattern is turned off when the TensorRT build lists it (it fails to compile with scalar-scale FP8) |

### Method

- Model: `suno/bark-small` at the manifest revision. bark large (24 layers,
  hidden size 1024) was not measured.
- GPUs: B300 for the main study (TensorRT 11.1.0.106); B200 and B300 for the
  spot check (TensorRT 11.4.0.106, one GPU per job, all variants built and timed
  in the same job).
- Speed: the engines of the bundle run from Python on synthetic inputs of the
  real sizes: semantic prefill (257 tokens), semantic and coarse decode step
  (one token), coarse prefill (256 tokens), fine pass (1024 positions). "CUDA
  graph" is graph replay; "eager" is `execute_async_v3` per step without a
  graph, which is what the C++ runtime does. TensorRT 11.4: median of 9
  interleaved rounds of 200 calls. TensorRT 11.1: mean of 300 graph replays
  (200 eager calls) per variant, variants in sequence. Kernel counts and times
  from the CUDA profiler.
- Audio quality: the native runtime was not run with FP8 bundles. The Hugging
  Face model was run in fp32, in fp16 (the floor) and with FP8 fake quantization
  of the same 108 projections with the same weight and activation scales (same
  math as the engines). 60 LibriSpeech test sentences (20 short, 34 medium, 6
  long; the calibration used the validation split), fixed sampling seed per
  sentence, transcribed with `openai/whisper-large-v3-turbo`. Log-mel L1 is the
  mean absolute difference of 80-band log-mel spectrograms to the fp32 audio of
  the same sentence and seed (sampling diverges, so this measures how far the
  audio moves, not an absolute quality).
- Engine agreement (TensorRT 11.4): last-row semantic logits after a prefill of
  257 real token embeddings, against the Hugging Face fp32 model; semantic decode
  step and fine pass logits against the fp16 engine. A NaN or Inf in any of these
  logits would show as an undefined cosine.

### Results

Speed, TensorRT 11.4, ms, CUDA graph / eager (x is fp16 time over variant time,
CUDA graph):

| GPU | Part | fp16 | FP8 | FP8, fp16 Dequantize | NVFP4 | x FP8 | x FP8 fp16 DQ | x NVFP4 |
|---|---|---|---|---|---|---|---|---|
| B200 | semantic decode step | 0.566 / 0.679 | 0.542 / 0.567 | 0.515 / 0.634 | 0.487 / 0.611 | 1.04 | 1.10 | 1.16 |
| B200 | coarse decode step | 0.547 / 0.683 | 0.541 / 0.567 | 0.514 / 0.631 | 0.477 / 0.607 | 1.01 | 1.07 | 1.15 |
| B200 | semantic prefill (257) | 1.697 / 2.024 | 1.585 / 1.726 | 1.544 / 1.877 | 1.638 / 2.030 | 1.07 | 1.10 | 1.04 |
| B200 | coarse prefill (256) | 0.900 / 1.273 | 0.857 / 1.048 | 0.806 / 1.165 | 0.899 / 1.318 | 1.05 | 1.12 | 1.00 |
| B200 | fine pass (1024) | 1.116 / 1.170 | 1.117 / 1.170 | 1.061 / 1.114 | 1.125 / 1.193 | 1.00 | 1.05 | 0.99 |
| B300 | semantic decode step | 0.526 / 0.665 | 0.529 / 0.559 | 0.501 / 0.628 | 0.516 / 0.640 | 0.99 | 1.05 | 1.02 |
| B300 | coarse decode step | 0.528 / 0.662 | 0.531 / 0.564 | 0.502 / 0.627 | 0.516 / 0.631 | 0.99 | 1.05 | 1.02 |
| B300 | semantic prefill (257) | 1.491 / 1.793 | 1.433 / 1.601 | 1.365 / 1.671 | 1.445 / 1.855 | 1.04 | 1.09 | 1.03 |
| B300 | coarse prefill (256) | 0.844 / 1.186 | 0.815 / 1.014 | 0.746 / 1.077 | 0.833 / 1.255 | 1.04 | 1.13 | 1.01 |
| B300 | fine pass (1024) | 1.058 / 1.101 | 1.070 / 1.108 | 1.010 / 1.048 | 1.067 / 1.154 | 0.99 | 1.05 | 0.99 |

Speed, TensorRT 11.1, B300, ms, CUDA graph (two jobs; the fp16 Dequantize
variant was measured in the second job against a new build of the FP8 engine):

| Part | fp16 | FP8 | NVFP4 | FP8 (job 2) | FP8 fp16 DQ (job 2) |
|---|---|---|---|---|---|
| semantic decode step | 0.533 | 0.517 | 0.525 | 0.523 | 0.518 |
| coarse decode step | 0.539 | 0.514 | 0.528 | 0.523 | 0.520 |
| semantic prefill (257) | 1.576 | 1.528 | 1.526 | 1.532 | 1.463 |
| fine pass (1024) | 1.086 | 1.096 | 1.097 | 1.077 | 1.014 |
| semantic decode step, eager | 0.671 | 0.551 | 0.689 | 0.558 | 0.652 |

Size (TensorRT 11.4; 11.1 within 0.5 MiB):

| Item | fp16 | FP8 | FP8 fp16 DQ | NVFP4 |
|---|---|---|---|---|
| bundle | 1006.8 MiB | 785.7 MiB | 784.4 MiB | 687.2 MiB |
| semantic / coarse / fine plans | 182 / 185 / 174 MiB | 108 / 111 / 100 MiB | 107 / 110 / 100 MiB | 75 / 78 / 68 MiB |
| EnCodec plan | 21 MiB | 21 MiB | 21 MiB | 21 MiB |

Decode step kernels, B300, TensorRT 11.1 (one semantic step, CUDA graph):

| Item | fp16 | FP8 | FP8 fp16 DQ | NVFP4 |
|---|---|---|---|---|
| kernels per step | 124 | 100 | 112 | 113 |
| GPU busy time per step | 0.659 ms | 0.548 ms | 0.662 ms | 0.658 ms |
| attention GEMV over the KV cache | 0.204 ms (31%) | not quantized | not quantized | not quantized |

The GPU busy time is larger than the graph wall time because TensorRT runs
parallel branches on more than one stream.

Audio quality (Hugging Face model, 60 sentences; WER of the Whisper transcript
against the text, short / medium / long in brackets):

| Model | WER vs text | WER vs fp32 transcripts | log-mel L1 vs fp32 | mean duration |
|---|---|---|---|---|
| fp32 | 0.081 (0.095 / 0.091 / 0.048) | 0 | 0 | 6.49 s |
| fp16 (floor) | 0.082 (0.134 / 0.080 / 0.053) | 0.066 | 0.82 | 6.46 s |
| FP8 (fake quantization) | 0.089 (0.181 / 0.075 / 0.059) | 0.109 | 2.38 | 6.40 s |

Engine agreement, TensorRT 11.4, B200 (B300 in brackets where it differs):

| Engine | semantic prefill logits vs HF fp32 (cosine, top-1) | semantic decode step vs fp16 engine | fine pass vs fp16 engine |
|---|---|---|---|
| fp16 | 1.00000 (0.99999), same | - | - |
| FP8 | 0.99651 (0.99642), same | 0.99849 (0.99875) | 0.99852 (0.99851) |
| FP8, fp16 Dequantize | 0.99723, differs | 0.99850 | 0.99853 (0.99858) |
| NVFP4 | NaN in the logits | 0.95877 (0.95907) | 0.98462 (0.98513) |

### Why no gain

- Batch 1, one token per step: each projection is a GEMV of a 768-wide row
  against a 1.2 to 4.7 MB fp16 weight, about 5 us per kernel. The time is
  launch and latency, not weight bytes, so halving the weight bytes saves
  little.
- About a third of the step is attention over the KV cache, which stays fp16.
- With fp32 Dequantize the FP8 GEMMs write fp32 and need cast kernels; fp16
  Dequantize removes them and gives the best CUDA graph result (1.05x to 1.10x),
  but adds kernels back, so the eager step gains less (1.06x to 1.07x).

### NVFP4

W4A4 NVFP4 (block-16 weights at build time, dynamic block-16 activations with
the same calibrated global scales) on the same 108 projections: the bundle is 32%
smaller, the decode step is 1.15x to 1.16x faster on B200 but only 1.02x on B300
(TensorRT 11.4), and the dynamic quantize kernels make the eager step slower than
FP8. The semantic prefill logits contain NaN on both GPUs and the decode step
logits drift (cosine 0.959 to the fp16 engine), so NVFP4 is not usable as is.

### What was not tried

| Item | Note |
|---|---|
| bark large (`suno/bark`) | not measured; by weight bytes (about 600 MB per stage, 75 us) its step is also latency bound |
| FP8 `w_o`, LM heads, FP8 KV cache | not measured |
| Audio from the C++ runtime with an FP8 bundle | not run; quality is from fake quantization in PyTorch |
| Batch above 1 | the engines are built for batch 1 |
| Per-channel weight scales, other calibration methods | not measured |
| A fix for the NVFP4 prefill NaN | not investigated |

### When to revisit

- The runtime gets CUDA graphs: the fp16 decode step then takes 0.53 to 0.57 ms
  instead of 0.66 to 0.68 ms eager (1.20x to 1.26x, no quality loss), and FP8
  would add at most 1.05x to 1.10x on top.
- Batched generation (several requests per step): the step stops being latency
  bound and FP8 GEMMs can pay.
- An FP8 recipe that keeps the audio closer to fp16 (for example per-channel
  weight scales or more layers in fp16), measured with the C++ runtime.
