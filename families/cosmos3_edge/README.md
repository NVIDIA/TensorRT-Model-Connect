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
