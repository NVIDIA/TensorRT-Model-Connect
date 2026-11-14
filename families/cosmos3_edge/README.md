# Cosmos3-Edge reasoning

This family owns `nvidia/Cosmos3-Edge` reasoning, separately from the Cosmos3
Diffusers generation/policy family. The documented checkpoint revision used by
its E2Es is `344d602b128d1bbdacb43b08d0a3626f46343e29`.

## Inventory and boundaries

| Checkpoint | Inputs | Output | Execution |
| --- | --- | --- | --- |
| `nvidia/Cosmos3-Edge` | Text | Text | Autoregressive |
| Same checkpoint | Image(s) and text | Text | Autoregressive |
| Same checkpoint | Timestamped video frames and text | Text | Autoregressive |
| Same checkpoint | Images, timestamped video, and text | Text | Autoregressive |

This is not a text-only stand-in for the multimodal model. The family runs the
complete vision encoder, projector, multimodal position computation and language
model. It does not implement action/policy output, diffusion generation, audio,
speculative decoding, quantized checkpoints, batching or tensor parallelism.
Those are not configurations advertised by this reasoning adapter.

## Build and installation

```sh
python -m tensorrt_model_connect build /path/to/Cosmos3-Edge \
  --family cosmos3_edge --task image_video_text_to_text \
  --precision fp16 --max-sequence-length 1024 -o cosmos3-edge.bundle
```

The family selects optional Edge-LLM 0.11.0 complete-network offload for FP16 on
Linux when a matching native SDK is discoverable through `CMAKE_PREFIX_PATH` or
the Python installation prefix. Provision the SDK using Model Connect's existing
optional Edge CMake dependency. The pin is
`95515c2f87fba8982db5a519f9022277667b3cc9`.

Python builder tools can come from a compatible wheel or source installation.
The adapter probes the current Python interpreter for Edge 0.11.0 and the SDK's
TensorRT version; otherwise it uses the SDK-provisioned interpreter. A wheel does
not replace the native SDK. Builds never install packages or cross-compile.

Absent or incompatible offload support uses the native family implementation.
An Edge preparation failure emits a warning with a persistent diagnostic log,
then retries native construction once with the original request. Publication
failures and inference errors do not silently retry on another backend.
All Edge-specific Python and C++ code is below this family's `edge_llm/` folders.
No shared builder/runtime selection or model-specific core code is required.

The native fallback constructs TensorRT graphs directly with its Python API and
executes them through Model Connect's C++ runtime; it does not use ONNX or custom
GPU kernels. `--precision fp32` explicitly selects this native path. FP16 and
BF16 native graphs also exist; full-model BF16 quality has not been qualified.
There is no native GPU-SKU whitelist. Serialized engines and optional Edge SDK
binaries still require a compatible target GPU and software stack.

The default sequence capacity is 2048. The selected capacity covers prompt and
output; over-capacity requests fail instead of silently shortening generation.
Indexed checkpoint shards may live in nested directories or a standard Hugging
Face snapshot cache. Only reasoning-indexed shards are packaged, without a
second staging copy; unrelated VAE assets are not needed. Original shard files
can contain other tensors, but only indexed reasoning weights are loaded.

## Runtime tasks

The public C++ API exposes `TextContinuation`, `ImagesTextToText`,
`VideoTextToText`, and `ImageVideoTextToText`. Use the matching task for the
modalities actually supplied. `ImageVideoTextToText` requires both image and
video inputs. Legacy text and vision-language interfaces remain available.

Images are HWC RGB uint8 or finite float32 values in `[0, 1]`. Video frames retain
the caller's order and timestamps; frames in one video must have equal dimensions.
Missing timestamps use the checkpoint processor's 24 fps default. A single-frame
video remains a video. The Edge adapter uses `ImageData.isVideo`, frame storage,
and timestamps rather than treating the video as a list of independent images.

Text continuation defaults to a raw prompt. Media tasks default to the checkpoint
chat template. Chat formatting, thinking, system prompt, output budget,
temperature, top-k, top-p and seed are family-owned controls. Native also supports
min-p and repetition penalty; the Edge adapter rejects nondefault values instead
of ignoring them. Requests do not expose tool calls or policy-generation controls.

## Validation and known accuracy issues

The independent oracle is Transformers FP32 with eager attention and TF32
disabled, using the same checkpoint, prompt, pixels, frame selection and timestamps.
The unchanged normalized text edit-distance limit is **0.15**.

The initial local validation used TensorRT 11.1 / CUDA 13.3 on SM80, capacity 1024,
batch 1, greedy decoding, chat templating enabled, thinking disabled. This is scoped evidence, not a claim
that every target GPU, sequence length or sampling configuration has passed.

| Backend / precision | Chat text (2 cases) | Image | Video | Mixed image/video |
| --- | --- | --- | --- | --- |
| Native FP32 | Exact tokens/text | Exact tokens/text | Exact tokens/text | Exact tokens/text |
| Native FP16 | Exact tokens/text | Exact tokens/text | Accuracy issue: NED 0.7077 | Accuracy issue: NED 0.7802 |
| Edge 0.11 FP16 | Exact tokens/text | Exact tokens/text | Accuracy issue: NED 0.7726 | Accuracy issue: NED 0.7625 |

All these build and inference commands succeeded. Accuracy-only failures remain
runnable and are not advertised as quality-qualified. Running the official Edge
C++ API directly with the **identical engines** reproduces all five Model Connect
Edge outputs exactly. Thus the observed Edge video discrepancies are not introduced
by argument mapping or the Model Connect adapter. Engine profile changes can alter
FP16 outputs; comparisons must use the same serialized engines.

A separate raw continuation probe (`The capital of France is`, no chat template,
24 generated tokens) exactly matches HF FP32 on native FP32, but both native and
Edge FP16 differ by NED 0.69. Direct Edge reproduces the same FP16 output. The HF
reference itself is repetitive on this unformatted prompt: reference parity is
not a claim that the raw prompt produces a useful answer. This additional
precision-sensitive failure remains runnable and is not covered by the passing
chat-text rows. Use the checkpoint chat template for instruction prompts.

The Edge visual implementation documents a four-tap learned-position interpolation
approximation on downsampled axes. This is a plausible contributor, not a proven
complete explanation. Native uses the full antialiased interpolation weights;
its FP32 path passes video while its FP16 path still has precision sensitivity.

Fixtures come from the pinned checkpoint's `assets/example_reasoning_input.png`
and `assets/example_action_id_av_0_input.mp4`. Video uses four uniformly selected
source frames, aspect-preserving resize to at most 384 pixels on the long edge,
and actual source timestamps. Text/image cases generate at most 64 tokens;
video/mixed cases at most 96. The owning test module contains the exact prompts
and preparation code.

## FP8 quantization study (not shipped)

FP8 for the language model was built and measured on TensorRT 11.4 and 11.1. It
is not shipped: the decoder is only 1.08x to 1.17x faster than fp16 and the
outputs drift far from the reference. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The FP8
code was not merged; this family still rejects `--quantization`.

### Summary and decision

| Item | fp16 (native) | FP8 | Change |
|---|---|---|---|
| Decode step, B200 / B300 | 2.726 / 2.505 ms | 2.322 / 2.320 ms | 1.17x / 1.08x |
| Prefill 1024 tokens, B200 / B300 | 12.09 / 11.69 ms | 10.38 / 10.76 ms | 1.17x / 1.09x |
| Bundle (decoder plan) | 5.43 GB (3.13 GiB) | 4.05 GB (1.84 GiB) | -25% (-41%) |
| Chat answers, KL vs HF / top-1 | 8e-6 / 100% | 0.030 / 96.9% | 3700x |
| Image and video answers, KL vs HF / top-1 | 4e-5 / 99.4% | 0.013 / 94.1% | 300x |
| 64-token greedy answers equal to HF, chat / media | 12 of 12 / 20 of 26 | 5 of 12 / 0 of 26 | |
| WikiText-2, KL vs HF / top-1 | 1.3e-4 / 99.2% | 0.244 / 70.1% | |

Decision: do not ship FP8 for Cosmos3-Edge.

- Quality: the official FP8 checkpoint gives a chat KL of 0.03 and an image and
  video KL of 0.013 against the HF model; no media answer stays equal to HF for
  64 tokens, and the FP8 and fp16 engines agree for 22 tokens on average. No
  tested subset of projections, scale set or weight source comes close to the
  fp16 floor (see ablations).
- Speed: the decode step is not weight-bandwidth bound. Removing 1.3 GB of
  weight reads saves 0.2 to 0.4 ms of 2.5 to 2.7 ms; the rest is the fixed-cost
  explicit attention, norms and launches of 394 TensorRT layers.
- The clear gain is size: the decoder plan is 41% smaller.

### Recipe that was tried

- Checkpoint: the official ModelOpt FP8 export `nvidia/Cosmos3-Edge-FP8`,
  revision `cc6b7d7144f9c08b73c38bf2cc0755397fe0bfe6` (ModelOpt 0.44, `FP8`,
  static per-tensor input scales, max calibration). Its reasoning tensors are
  the same model as the BF16 checkpoint: embeddings, norms, `lm_head`, vision
  tower and projector are bit-identical, every weight scale is amax / 448 of
  the BF16 weights, and 93% to 98% of the FP8 bytes of the projections with
  their own weight scale equal a fresh quantization of the BF16 weights.
- Quantized: `to_q`, `to_k`, `to_v`, `to_out`, `mlp.up_proj` and `mlp.down_proj`
  of all 28 layers (168 projections). The export shares one weight scale and one
  input scale across `to_q`, `to_k` and `to_v` of a layer. Its ignore list keeps
  `lm_head`, `visual*` and `projector*` in BF16; the build kept the vision tower,
  projector, embeddings, norms, attention matmuls and `lm_head` in fp16.
- Outlier rule: `layers.1.mlp.down_proj` has an input scale 25x the median of
  its kind and stayed 16-bit (FP8 values dequantized at build time).
- Graph: `activation -> Quantize -> Dequantize -> MatMul`, FP8 weight constant
  through Dequantize, both Dequantize layers in fp16, one FP8 GEMM per
  projection. `to_out` builds as FP8 on TensorRT 11.4 and 11.1 here, because the
  decoder attention is explicit matmul and softmax, not a fused attention.
  The dual-GEMM knob was turned off when TensorRT listed it.
- The checkpoint was calibrated on video-generation captions
  (`quantization_metadata.json`). On reasoning inputs the activation maxima are
  up to 2.7x larger than its scales (layer 9 `down_proj`, q/k/v of layers 0, 1,
  26 and 27).

### Method

- One SLURM job per GPU type; all variants of that job built and timed in one
  process with TensorRT 11.4.0.106, capacity 1024 (the manifest value), fp16.
  Decode: one-token step of the decoder plan replayed as a CUDA graph, cache
  filled, CUDA events, median of 11 interleaved rounds of 50 steps. Prefill:
  10 runs per round, empty cache. This times the decoder engine only; the
  family runtime and the visual engine (identical in both bundles) are not
  included.
- Quality: B200, TensorRT 11.4, reference Transformers FP32 (SDPA, TF32 off) of
  `nvidia/Cosmos3-Edge` at the manifest revision. The engines receive the HF
  input embeddings and MRoPE cosines and sines of the same sequence, so the
  visual features come from PyTorch and only the decoder is compared.
  - WikiText-2 test: 40 windows of 1024 tokens, teacher forced.
  - Chat: 12 held-out Nemotron-Post-Training-Dataset-v1 prompts (chat, code,
    math, stem) with the chat template, HF greedy answer of 64 tokens; KL and
    top-1 over the answer positions, and engine greedy decoding compared with
    HF.
  - Media: 24 ADE20K (`scene_parse_150` validation) images at up to 640 pixels
    with three describe/list/where questions, and the family's video asset
    (4 frames) with two prompts; same metrics.
- Ablations ran as fake quantization in the HF model (the same FP8 cast on the
  projection inputs and weights); it reproduces the engine: WikiText KL 0.238
  vs 0.235, chat 0.039 vs 0.030 on the same samples.

### Speed (TensorRT 11.4, capacity 1024, ms)

| GPU | Variant | Decode step | Prefill 128 | Prefill 512 | Prefill 1024 |
|---|---|---|---|---|---|
| B200 | fp16 | 2.726 | 3.130 | 6.563 | 12.092 |
| B200 | FP8 (fp16 Dequantize) | 2.322 (1.17x) | 2.808 (1.11x) | 5.698 (1.15x) | 10.379 (1.17x) |
| B200 | FP8, fp32 Dequantize | 2.424 | 2.810 | 6.146 | 11.324 |
| B200 | NVFP4 MLP, FP8 attention | 2.445 | 2.977 | 5.610 | 10.294 |
| B300 | fp16 | 2.505 | 3.102 | 6.437 | 11.689 |
| B300 | FP8 (fp16 Dequantize) | 2.320 (1.08x) | 2.723 (1.14x) | 5.977 (1.08x) | 10.757 (1.09x) |
| B300 | FP8, fp32 Dequantize | 2.351 | 2.765 | 5.933 | 10.741 |
| B300 | NVFP4 MLP, FP8 attention | 2.311 | 2.966 | 5.614 | 10.102 |

Both plans have 394 layers; 223 layers of the FP8 plan run e4m3 kernels. The
unquantized decoder graph was unchanged by the FP8 code (1894 network layers
before and after, same plan size within 3 KB).

### Quality (B200, TensorRT 11.4, against Transformers FP32)

| Variant | WikiText KL / top-1 / PPL (HF 339.7) | Chat KL / top-1 / greedy equal | Media KL / top-1 / greedy equal |
|---|---|---|---|
| fp16 | 0.00013 / 99.2% / 340.2 | 8e-6 / 100% / 12 of 12 | 4e-5 / 99.4% / 20 of 26 |
| FP8, official scales | 0.244 / 70.1% / 332.3 | 0.030 / 96.9% / 5 of 12 | 0.013 / 94.1% / 0 of 26 |
| FP8, `layers.1.mlp.down_proj` also FP8 | 0.249 / 70.6% / 340.2 | 0.037 / 95.5% / 1 of 12 | - |
| NVFP4 MLP, FP8 attention | 0.409 / 61.6% / 361.7 | 0.146 / 91.5% / 0 of 12 | - |

The model gives a perplexity of about 340 on raw WikiText even in FP32: it is a
reasoning model used with its chat template (see the raw continuation note
above). The WikiText rows measure agreement with HF, not language-model quality.

Engine ablations (second B200 job, 12 WikiText windows; fp16 floor in that
build 0.0018 / 97.7%):

| Variant | WikiText KL / top-1 | Chat KL | Media KL / top-1 |
|---|---|---|---|
| FP8, official scales and weights | 0.235 / 70.5% | 0.030 | 0.013 / 94.1% |
| FP8, fp32 Dequantize | 0.236 / 70.3% | 0.031 | 0.013 / 94.9% |
| FP8, scales recalibrated on reasoning data | 0.305 / 70.5% | 0.047 | 0.010 / 94.6% |
| FP8, own per-projection weight scales from BF16, recalibrated scales | 0.187 / 71.7% | 0.033 | 0.009 / 95.7% |
| FP8, own weight scales, official input scales | 0.168 / 72.8% | 0.032 | 0.012 / 94.7% |

Recalibration used 154 held-apart inputs (72 Nemotron v1 prompts, 32
CNN/DailyMail articles, 48 ADE20K images, 2 video prompts) with 32 generated
tokens each, maximum of the absolute input per projection. Larger scales did
not help, so clipping is not the cause.

Fake-quantization ablations (6 WikiText windows, 12 chat answers; all FP8 =
0.238 / 0.039):

| Quantized | WikiText KL | Chat KL |
|---|---|---|
| Weights only (W8A16), all projections | 0.087 | 0.011 |
| Activations only, all projections | 0.094 | 0.017 |
| Only `to_q` / `to_k` / `to_v` | 0.013 / 0.015 / 0.050 | 0.002 / 0.002 / 0.002 |
| Only `to_out` | 0.118 | 0.004 |
| Only `up_proj` / `down_proj` | 0.039 / 0.100 | 0.026 / 0.007 |
| Only layer 0 / 1 / 5 / 10 | 0.150 / 0.043 / 0.013 / 0.032 | 0.002 / 0.001 / 0.001 / 0.001 |
| Only one other layer | 0.001 to 0.004 | 0.0004 to 0.0023 |
| All except layers 0 and 1 | 0.079 | 0.017 |
| All except layers 0, 1, 5, 10 and every `to_out` | 0.035 | 0.015 |
| All except layers 0, 1 and every `down_proj` | 0.047 | 0.028 |

The error is spread over every projection kind and over the first layers; even
FP8 weights alone (W8A16) give a chat KL of 0.011. Excluding enough layers and
projections to approach the floor would leave little of the already small
speedup.

### NVFP4

Measured, not shipped. MLP projections in NVFP4 (weights from the BF16
checkpoint, input scale = FP8 input scale / 6, `layers.1.mlp.down_proj` in
16-bit) with FP8 attention: decoder plan 1.42 GiB (FP8 1.84 GiB), decode not
faster than FP8 (B200 2.445 vs 2.322 ms, B300 2.311 vs 2.320 ms), prefill 1024
1% to 6% faster, KL 1.7x (WikiText) to 4.9x (chat) worse than FP8.

### Other findings

- The default capacity (2048) gives a visual profile of 8192 patches. Its
  unquantized fp16 visual plan does not build on TensorRT 11.4 or 11.1 (no
  tactic fits the 2 GiB workspace for the fp32 attention scores); 4096 patches
  (capacity 1024) builds. This is independent of FP8.
- No NaN or Inf appeared in the FP8 logits in the quality runs (about 80
  sequences with 64 decode steps each, plus 40 WikiText windows); the fp16
  Dequantize did not overflow. A separate 200-input scan was not run.

### When to revisit

- A Cosmos3-Edge FP8 checkpoint calibrated for reasoning, or a recipe with
  finer weight or activation scales (per channel, per token) that TensorRT can
  run as FP8 GEMMs.
- A decoder whose decode step is weight-bandwidth bound (fused attention, fewer
  layers per step), which would raise the FP8 share of the time.

## Owning tests

```sh
cmake --build build --target trtmc_model_cosmos3_edge cosmos3_edge_inference_test
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
TRTMC_RUNTIME_ROOT="$PWD/build" \
  PYTHONPATH=core/builder:apps/benchmark:. \
  python -m pytest families/cosmos3_edge/tests --e2e-model cosmos3_edge -q
```

The registered positive profiles validate native FP32 across all four modalities,
and FP16 text/image behavior through the selected backend. They do not turn the
recorded FP16 video failures into passes. All profiles keep the 0.15 threshold.
With `TRTMC_BUILD_TESTS=ON`, `test_cosmos3_edge_media` checks antialiased position
weights; an Edge-enabled build also provides `test_cosmos3_edge_request` for
actual frame/timestamp/seed mapping, invalid media and capacity contracts.
