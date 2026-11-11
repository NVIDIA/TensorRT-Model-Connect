# MagpieTTS

This family builds `nvidia/magpie_tts_multilingual_357m` (a NeMo `.nemo`
archive) for `audio_generation`: a text encoder engine (6 layers), an
autoregressive decoder engine (12 layers with KV cache and cross-attention,
8 codebook tokens per frame) and a NanoCodec engine that turns codes into a
22 kHz waveform. The native runtime prefills the baked speaker context, then
runs two decoder passes per frame (conditioned and unconditioned, for
classifier-free guidance). The manifests build fp32; fp16 is also supported.
The family has no quantized build.

## Context prefill

The decoder plan has two optimization profiles: profile 0 for the one-frame
decoder step and profile 1 for a block of up to 110 frames (the baked speaker
context). The native runtime runs each phase in its own execution context: it
writes the first 109 context frames into the conditioned and unconditioned KV
caches with one profile-1 call each, then runs the last context frame and all
generated frames through the profile-0 context. Before, it ran all 110 context
frames one step at a time, twice, and ran every decoder step in the profile-1
context. Tensor-parallel bundles keep the one-step path.

Measured with the native runtime, fp32 manifest bundle, TensorRT 11.4, seed 42,
median of 15 interleaved runs of each sentence (5 rounds of 3), one process per
variant and round after one warm-up request. Time is to the full waveform
(encoder, decoder and codec). Sentences: short (48 frames), medium (91 frames)
and long (295 to 309 frames, the premerge prompt).

| GPU | Sentence | Before | Batched prefill only | This change | Speedup |
|---|---|---|---|---|---|
| B200 | short | 895 ms | 335 ms | 302 ms | 2.97x |
| B200 | medium | 1132 ms | 572 ms | 509 ms | 2.22x |
| B200 | long | 2255 ms | 1770 ms | 1496 ms | 1.51x |
| B300 | short | 861 ms | 328 ms | 309 ms | 2.79x |
| B300 | medium | 1088 ms | 558 ms | 521 ms | 2.09x |
| B300 | long | 2242 ms | 1730 ms | 1580 ms | 1.42x |

"Batched prefill only" keeps the decoder step in the profile-1 context. The
context prefill (both caches) goes from 572 ms to 11 ms (B200) and from 550 ms
to 11 ms (B300). The two decoder passes of a frame go from 5.33 to 4.67 ms
(B200) and from 5.12 to 4.83 ms (B300) when the step runs in its own profile.
TensorRT 11.1 (B300): 2.84x, 2.10x and 1.42x.

Output: with the default sampling, 11 of 13 test sentences give bit-identical
audio, and all 13 whisper-large-v3-turbo transcripts are identical to the
one-step path (WER 2.35% for both). The other sentences change because a few
logits differ in the last bits and a sample flips. No non-finite samples.

## Cross-attention K and V

The cross-attention K and V of a decoder layer depend only on the encoder
output (2048 padded rows). Single-GPU bundles carry a `cross_kv.plan` that
applies each layer's memory LayerNorm and K and V projections to the encoder
output, with the same ops and dtypes as the decoder used. The runtime runs it
once per request for the text and once for the classifier-free-guidance null
text, and binds the per-layer K and V rows to the decoder's `cross_k_<i>` and
`cross_v_<i>` inputs, which now have shape [2048, 128] (xa_n_heads x
xa_d_head) in the layer's compute dtype. Before, every decoder step normalized
and projected all 2048 rows in every layer. Padded rows are zero in both
cases. Bundles without `cross_kv.plan` (built before this change, and
tensor-parallel bundles) keep the old inputs and path. The decoder with
projected inputs is stored as `decoder.cross_kv.plan` instead of
`decoder.plan`, so a runtime from before this change stops at load with
`bundle section is missing or empty: decoder.plan` instead of binding the raw
encoder output to it.

Same measurement as "Context prefill" (TensorRT 11.4, fp32 manifest bundle):

| GPU | Sentence | Before (prefill change only) | This change | Speedup | vs one-step prefill |
|---|---|---|---|---|---|
| B200 | short | 302 ms | 276 ms | 1.09x | 3.24x |
| B200 | medium | 509 ms | 464 ms | 1.10x | 2.44x |
| B200 | long | 1496 ms | 1341 ms | 1.12x | 1.68x |
| B300 | short | 309 ms | 287 ms | 1.08x | 3.00x |
| B300 | medium | 521 ms | 486 ms | 1.07x | 2.24x |

The two decoder passes of a frame go from 4.67 to 4.17 ms (B200) and from
4.83 to 4.44 ms (B300); fp16 bundles: 2.93 to 2.65 ms and 2.83 to 2.64 ms. On
B300 the long sentence ends at 295 frames instead of 309 (one sample flips),
so its time is not compared. TensorRT 11.1 (B300): 4.84 to 4.52 ms per frame.
The fp32 decoder plan is 394 MiB instead of 403 MiB, plus 9 MiB for
`cross_kv.plan`. The runtime's cross-attention buffers take 50 MiB instead of
302 MiB.

Output: all 13 whisper-large-v3-turbo transcripts are identical (WER 2.35%),
no non-finite samples. On B300, 12 of 13 sentences give bit-identical audio.
On B200, 12 of 13 have the same length and differ by at most 1.1e-3 (the
codec plan of the separately built bundle picks other kernels).

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder and decoder projections was built and
measured on TensorRT 11.4 (and 11.1) on B200 and B300. It is not shipped. FP8
needs an fp16 build, and against that fp16 build it is only 1.3% to 2.1%
faster end to end. The speedup over the fp32 manifest comes from fp16, not
from FP8. FP8 also changes 37% of the per-codebook argmax choices (fp16
changes 1.5%). The decoder plan is 38% smaller. This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

x is the time of the reference engine over the FP8 time. "Synthesis" is the
mean wall time per decoder step of a full greedy synthesis loop. The loop runs
like the runtime: sequential context prefill, two passes per frame, host
sampling, no CUDA graph.

| Item (TensorRT 11.4) | fp32 (manifest) | fp16 | FP8, fp16 Dequantize | FP8 vs fp16 |
|---|---|---|---|---|
| Synthesis, B200 / B300 | 3.305 / 3.415 ms | 2.389 / 2.453 ms | 2.339 / 2.412 ms | 1.02x / 1.02x |
| Decoder step, CUDA graph, B200 / B300 | 1.921 / 2.076 ms | 1.159 / 1.088 ms | 1.107 / 1.050 ms | 1.05x / 1.04x |
| Decoder step, eager enqueue, B200 / B300 | 2.384 / 2.456 ms | 1.538 / 1.570 ms | 1.485 / 1.527 ms | 1.04x / 1.03x |
| Decoder plan size | 403 MiB | 205 MiB | 127 MiB | -38% |
| Teacher-forced argmax agreement vs fp32 | 1 | 0.985 | 0.626 | |
| Teacher-forced logit relative error vs fp32 | 0 | 0.0017 | 0.0975 | |
| Whisper round-trip WER, 60 sentences (TensorRT 11.1, B300) | 0.0205 | 0.0141 | 0.0151 (fp32 Dequantize) | |
| Non-finite logits | 0 | 0 | 0 | |

Decision: do not ship FP8 for MagpieTTS.

- The decoder step is latency bound. At batch 1 it runs about 294 small
  kernels (about 3.5 us each), and the weights are about 2% of the step time
  at HBM bandwidth. Four concurrent decoder streams give 2.6x the throughput
  of one stream (TensorRT 11.1, B300). Smaller weights cannot make this step
  much faster.
- The FP8 GEMMs save 0.04 to 0.16 ms per step of fp16 GEMM time, but the
  Quantize and Dequantize kernels take most of it back. With fp32 Dequantize
  (the first recipe) the cast and reformat time grows from 0.27 to 0.46 ms per
  step.
- With fp32 Dequantize, eager enqueue is 1.17x faster on B300 but 0.98x on
  B200, so the synthesis gain depends on the GPU and host (1.10x on B300,
  0.98x on B200). Only fp16 Dequantize is faster on both GPUs, and only by
  about 2%.
- Quality: the whisper WER stays at the level of the fp16 and fp32 engines,
  but FP8 picks a different code in 37% of the teacher-forced codebook
  decisions. The fp16 engine differs from fp32 in 1.5%.

### Recipe that was measured

No ModelOpt FP8 or NVFP4 checkpoint exists for MagpieTTS, and there is no
official quantization config for it. The recipe is the usual one for
encoder-decoder audio models: encoder and decoder linear layers and
cross-attention in FP8; convolutions wider than 1, embeddings and heads in fp16.

| Choice | Decision |
|---|---|
| FP8 (126 projections) | encoder q, k, v (18); decoder self-attention q, k, v, cross-attention q, k, v, o, MLP fc1, fc2 (108) |
| fp16 | decoder self-attention output projection (TensorRT 11.4 cannot compile an FP8 Quantize after the fused attention), encoder output projection, kernel-size-3 encoder feed-forward, embeddings, norms, output head (768 x 16192), NanoCodec |
| Weight scale | per tensor, max(abs(weight)) / 448, quantized at build time |
| Activation scale | static, per tensor: max(abs(input)) of every projection while the fp32 engines synthesize 48 LibriSpeech validation sentences |
| Dequantize type | fp32 (first recipe) or fp16; both gave no non-finite value |
| Build | precision fp16, no tensor parallelism |

### Method

- GPUs: B200 and B300, one GPU per job. TensorRT 11.4.0.106 (target) and
  11.1.0.106. All four engines (fp32, fp16, FP8 with fp32 Dequantize, FP8
  with fp16 Dequantize) were built and timed in the same job.
- Decoder step: one step at cache position 40, median of 7 interleaved
  rounds. Each round measures eager enqueue (300 calls) and CUDA graph replay
  (500 calls). The native runtime does not use CUDA graphs, so eager enqueue
  is the closer proxy.
- Synthesis: the first 6 `test-clean` LibriSpeech sentences of 6 to 20 words,
  greedy decoding with the runtime's classifier-free guidance and alignment
  prior, 3 interleaved rounds, median. The time is divided by the number of
  decoder steps, because the engines stop at slightly different frame counts.
- Quality: teacher forcing on 8 held-out LibriSpeech test sentences (6
  `test-clean`, 2 `test-other`; about 620 frames). Both engines get the same
  inputs and the decisions of the reference engine; we compare the guided
  logits of the 8 codebooks. WER: 60 LibriSpeech test sentences (20 short, 30
  medium, 10 long) are synthesized, transcribed with
  `openai/whisper-large-v3-turbo` and scored against the text. This was run
  once, on TensorRT 11.1 on B300.
- Non-finite scan: every logit of the teacher-forced runs.

### Results

Speed (x is fp16 time over FP8 time; "dq32" and "dq16" are the fp32 and fp16
Dequantize recipes):

| GPU | TensorRT | step graph fp16 / dq32 / dq16 ms | x graph dq32 / dq16 | step eager fp16 / dq32 / dq16 ms | x eager dq32 / dq16 | synthesis ms per step fp32 / fp16 / dq32 / dq16 | x synthesis dq32 / dq16 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1.159 / 1.109 / 1.107 | 1.05 / 1.05 | 1.538 / 1.576 / 1.485 | 0.98 / 1.04 | 3.305 / 2.389 / 2.427 / 2.339 | 0.98 / 1.02 |
| B300 | 11.4 | 1.088 / 1.051 / 1.050 | 1.04 / 1.04 | 1.570 / 1.337 / 1.527 | 1.17 / 1.03 | 3.415 / 2.453 / 2.227 / 2.412 | 1.10 / 1.02 |
| B200 | 11.1 | 1.247 / 1.183 / 1.196 | 1.05 / 1.04 | 1.592 / 1.638 / 1.537 | 0.97 / 1.04 | 3.302 / 2.388 / 2.432 / 2.340 | 0.98 / 1.02 |
| B300 | 11.1 | 1.100 / 1.064 / 1.063 | 1.03 / 1.03 | 1.585 / 1.346 / 1.548 | 1.18 / 1.02 | 3.388 / 2.450 / 2.215 / 2.418 | 1.11 / 1.01 |

Plan size (MiB, same on both GPUs and both TensorRT versions within 0.3 MiB):

| Engine | fp32 | fp16 | FP8 |
|---|---|---|---|
| encoder | 391.7 | 196.4 | 186.5 |
| decoder | 403.2 | 205.1 | 127.3 to 128.3 |

Quality, teacher forced (TensorRT 11.4, B200; on B300 and 11.1 the means are
within 0.01, and the minimum cosine of FP8 with fp16 Dequantize is 0.894 on
11.1):

| Engine vs fp32 | logit cosine mean / min | relative error | per-codebook argmax agreement | non-finite frames |
|---|---|---|---|---|
| fp16 | 1.00000 / 0.99997 | 0.0017 | 0.985 | 0 |
| FP8, fp32 Dequantize | 0.99550 / 0.97199 | 0.0962 | 0.636 | 0 |
| FP8, fp16 Dequantize | 0.99529 / 0.94904 | 0.0975 | 0.626 | 0 |

Quality, whisper round trip (TensorRT 11.1, B300, 60 sentences, FP8 with fp32
Dequantize):

| Engine | WER all | short / medium / long | WER vs fp16 transcripts | log-mel L1 vs fp32 |
|---|---|---|---|---|
| fp32 | 0.0205 | 0.047 / 0.022 / 0.010 | 0.0097 | 0 |
| fp16 | 0.0141 | 0.016 / 0.017 / 0.010 | 0 | 1.47 |
| FP8 | 0.0151 | 0.008 / 0.022 / 0.010 | 0.0043 | 1.77 |

Kernel time of one decoder step (CUDA graph, TensorRT 11.4, B200; ms per step):

| Kernel group | fp16 | FP8 fp32 Dequantize | FP8 fp16 Dequantize |
|---|---|---|---|
| GEMM | 1.134 | 0.973 | 1.099 |
| cast, reformat, Quantize | 0.271 | 0.461 | 0.281 |
| other pointwise | 0.136 | 0.041 | 0.136 |
| sum (busy, kernels overlap) | 1.542 | 1.474 | 1.517 |
| kernels per step | 294 | 314 | 290 |

Subsets and other variants (TensorRT 11.1, B300, decoder step with CUDA graph;
fp16 1.091 ms):

| Variant | step ms | decoder MiB |
|---|---|---|
| FP8 all (recipe above) | 1.054 | 128 |
| FP8 MLP only | 1.078 | 152 |
| FP8 self-attention q, k, v only | 1.079 | 185 |
| FP8 cross-attention only | 1.081 | 202 |
| FP8 all but cross-attention | 1.059 | 132 |
| NVFP4 (W4A4, dynamic block-16 activations) all | 1.132 | 93 |
| NVFP4 MLP only | 1.110 | 128 |

### NVFP4

NVFP4 was built for the same projections, with dynamic activation quantization
in blocks of 16. It is slower than fp16 at batch 1 (1.132 vs 1.091 ms per step
on B300, TensorRT 11.1). The dynamic quantize kernels add 0.17 to 0.25 ms per
step, and the GEMMs are too small to gain from 4-bit weights.

### Other findings (not quantization)

Two changes would make synthesis much faster, but they are not part of this
study:

- The decoder engine has a batched prefill profile (profile 1). The runtime
  did not use it: it ran the 110 context frames one step at a time, twice
  (conditioned and unconditioned). This was 55% to 75% of the time of a short
  or medium sentence. In the study, the batched profile prefilled 110 rows in
  1.46 ms (fp16, B300). The sequential path took about 358 ms. The fp16 KV
  cache matched to a relative error of 6e-4. The runtime now uses it (see
  "Context prefill").
- The cross-attention K and V projections of the 2048 padded encoder rows were
  recomputed in every decoder step. Computing them once per sentence in a
  separate engine (0.11 ms) makes the step 13% faster (1.091 to 0.953 ms, CUDA
  graph, B300, TensorRT 11.1). Single-GPU bundles now do this (see
  "Cross-attention K and V").

With FP8, the batched prefill and the sequential prefill disagree more: KV
relative error 3.8% (fp16: 0.06%). The activation scales are static, so this is
not explained. It was not reduced to a TensorRT repro.

### What was not tried

| Item | Note |
|---|---|
| FP8 self-attention output projection, FP8 output head, FP8 KV cache | not measured |
| Batch size above 1 | the engine has no batch dimension |
| Native C++ runtime with an FP8 bundle, H100 | not measured; the Python loop above mirrors the runtime |
| Whisper WER on TensorRT 11.4 | teacher-forced metrics on 11.4 match 11.1 within 0.01 |

### When to revisit

- The runtime runs several requests in one decoder batch, so that the GEMMs
  become large enough to be bandwidth or compute bound.
- A TensorRT release fuses the activation Quantize and the output cast into
  the FP8 GEMM at M = 1, so that FP8 adds no cast kernels.
- Plan size alone (decoder 205 to 128 MiB) becomes a goal: the recipe above
  builds on 11.4 and gives no non-finite values.
