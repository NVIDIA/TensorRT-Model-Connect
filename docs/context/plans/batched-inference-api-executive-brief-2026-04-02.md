# Batched Inference Review

Date: 2026-04-02

## Purpose

This note defines the public local-inference API from the user journey outward.

The goal is not to guess one perfect internal implementation. The goal is to
say, very concretely:

- what kind of user is trying to do what
- what bundle they need to build
- how they use that bundle at runtime
- which public API best matches that story
- how that API should wire into the existing `factory -> registry -> plugin -> pipeline` design

This document is documentation-only. No source-code implementation is part of
this proposal.

## One Important Boundary

The local inference library and the CLI should not have the same contract.

- `trtmc-build` produces a `.trtfb` bundle.
- `trtmc` CLI may accept files such as `png`, `jpg`, or `wav` because the CLI can
  decode files before calling the library.
- the public C++ library API should accept in-memory data, not file paths
  and not compressed file formats

That means:

- local text input should be `std::string`
- local image input should be decoded pixels or tensors
- local audio input should be decoded PCM samples or tensors
- lower-level APIs should accept tensors directly

This matches the real current load path in
[pipeline_factory.cpp](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/src/runtime/registry/pipeline_factory.cpp),
[pipeline_registry.h](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/include/trtmc/runtime/pipeline_registry.h),
[pipeline_plugin.h](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/include/trtmc/runtime/pipeline_plugin.h),
and [pipeline.h](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/include/trtmc/pipeline.h).

## Existing Wiring

Every story in this document follows the same runtime wiring:

1. user builds a `.trtfb` bundle with `trtmc-build`
2. runtime calls `trtmc::load(bundle_path)`
3. `PipelineFactory::from_bundle()` reads `config.json`
4. `runtime_strategy` selects a plugin through `PipelineRegistry`
5. the plugin builds one concrete `IPipeline`
6. the user calls that pipeline through the public C++ API

That is already a good architecture boundary.

The new API should therefore live on `IPipeline`, not in the factory and not in
the registry.

## Public API Shape

The public API should have two layers.

### Layer 1: Generic Data-Plane API

This layer is for users who already have tensors and do not want semantic help.

Core API:

- `invoke(const InvokeRequest&) -> InvokeResponse`

Expected input shape:

- named tensors
- explicit tensor names
- explicit tensor shapes
- explicit dtypes
- explicit memory kind such as host, pinned host, or device

This layer should not talk about prompts, images, masks, transcripts, or audio.

### Layer 2: Semantic API

This layer is for users who want task-level calls.

Representative APIs:

- `generate_text(...)`
- `generate_text_batch(...)`
- `generate_text_stream(...)`
- `generate_text_batch_stream(...)`
- `segment(...)`
- `segment_batch(...)`
- `embed(...)`
- `embed_batch(...)`
- `transcribe(...)`
- `transcribe_batch(...)`
- `transcribe_stream(...)`
- `transcribe_batch_stream(...)`
- `generate_audio(...)`
- `generate_audio_batch(...)`
- `generate_audio_stream(...)`
- `generate_audio_batch_stream(...)`

Rule of thumb:

- if the caller already has packed tensors, use `invoke(...)`
- if the caller thinks in terms of prompts, images, texts, masks, transcripts,
  or audio clips, use a semantic API

## End-To-End User Stories

The stories below are the actual product requirements. Internal choices such as
dynamic shape, static buckets, separate engines, or optimization profiles are
implementation details derived from these stories.

### Story 1: Application Engineer Wants One Text Answer

Who this user is:

- an application engineer building a local chat or completion feature

What they want:

- build one text-generation bundle
- load it locally
- pass one prompt string
- get one text result back

What they have at runtime:

- a prompt string
- optional generation config such as `max_new_tokens`, `temperature`, and `top_k`

What functionality they need:

- one simple semantic call
- no tensor packing
- no batch management
- no file decoding

How they build the bundle:

```bash
trtmc-build build Qwen/Qwen3-0.6B -o qwen3.trtfb
```

What that bundle means:

- it packages the engines and assets needed for text generation
- it writes `runtime_strategy` into `config.json`
- at runtime that strategy will route into the decoder plugin and then into a
  concrete text-generation pipeline

How they use the bundle at runtime:

CLI today:

```bash
./build/trtmc run qwen3.trtfb --prompt "Hello" --max-new-tokens 50 \
  --hf-python /opt/venv/bin/python
```

Local C++ API today:

```cpp
auto pipe = trtmc::load("qwen3.trtfb");
trtmc::TextResult out = pipe->generate("Hello", {.max_new_tokens = 50});
```

Preferred local C++ API design:

```cpp
auto pipe = trtmc::load("qwen3.trtfb");
trtmc::TextResult out = pipe->generate_text("Hello", {.max_new_tokens = 50});
```

Why this should be a semantic API:

- the user thinks in prompts and text, not input tensors and output tensors

How this wires into the current architecture:

- `load("qwen3.trtfb")`
- factory reads `runtime_strategy`
- registry finds the decoder plugin
- plugin creates a concrete text-generation pipeline
- pipeline implements `generate_text(...)`

What this means for bundle build:

- the builder must produce the engine family that this text pipeline expects
- if the pipeline later wants distinct prefill and decode execution families,
  that stays builder- and plugin-owned
- the public contract does not need to expose those details

### Story 2: Performance Integrator Already Has Packed Tensors

Who this user is:

- a performance-sensitive integrator
- a researcher or systems engineer already holding tensors in memory
- someone who may already preprocess on GPU and wants minimal abstraction

What they want:

- build one bundle
- load it locally
- pass already-packed tensors
- get named output tensors back

What they have at runtime:

- explicit input tensors such as:
  - `pixel_values [B, 3, H, W]`
  - `input_ids [B, T]`
  - `mel_features [B, n_mels, n_frames]`
- host or device memory
- knowledge of the model’s expected tensor names

What functionality they need:

- a tensor-centric contract
- no semantic preprocessing
- no hidden file decoding
- no requirement to start from strings, images, or audio clips

How they build the bundle:

Examples:

```bash
trtmc-build build Qwen/Qwen3-0.6B -o qwen3.trtfb
trtmc-build build models/hf/Qwen__Qwen3-0.6B -o qwen3.trtfb
```

or for a different tensor-oriented model:

```bash
trtmc-build build <model> -o model.trtfb
```

What that bundle must provide:

- a stable pipeline type
- plugin-specific knowledge of valid input and output tensors
- the execution families that pipeline plans to accept

How they use the bundle at runtime:

Target local C++ API:

```cpp
trtmc::InvokeRequest req;
req.inputs = {
    {"pixel_values", {/* tensor view */}},
};
req.requested_outputs = {"mask"};

auto pipe = trtmc::load("model.trtfb");
trtmc::InvokeResponse out = pipe->invoke(req);
```

Why this should be the lower-level API:

- this user already knows the data plane
- semantic wrappers would only add friction

What input form the library should accept:

- `TensorView` or equivalent
- explicit dtype
- explicit shape
- explicit memory kind

What input form the library should not require here:

- `png` or `jpg` paths
- compressed image bytes hidden inside semantic wrappers

How this wires into the current architecture:

- same load path as every other story
- the concrete pipeline decides whether it wants to expose `invoke(...)`
- factory and registry remain unchanged

What this means for bundle build:

- builder decisions must be driven by the tensor families the pipeline wants to
  accept
- if the pipeline wants to expose packed batch execution, the bundle must
  contain the engine family that can execute those packed batches
- whether this is done through static buckets, optimization profiles, separate
  engines, or a hybrid strategy stays plugin-owned

### Story 3: Computer Vision Engineer Wants 5 Segmentation Masks In One Call

Who this user is:

- a computer vision engineer processing many images locally

What they want:

- build one segmentation bundle
- pass 5 independent images in one call
- get 5 segmentation results back in the same logical order

What they have at runtime:

- local library case:
  - 5 decoded images already in memory
  - each image is pixels plus `height`, `width`, `channels`, and layout
- CLI case:
  - 5 image files on disk
  - the CLI can decode them before calling the library

What functionality they need:

- one semantic batch call
- one result per input item
- stable input-to-output mapping
- no need for streaming

How they build the bundle:

```bash
trtmc-build build <segformer-or-other-segmentation-model> -o segformer.trtfb
```

How they use the bundle at runtime:

CLI today for one image:

```bash
./build/trtmc segment segformer.trtfb --image input.png --output mask.png
```

Target local C++ API:

```cpp
std::vector<trtmc::ImageView> images = {/* 5 decoded images */};

auto pipe = trtmc::load("segformer.trtfb");
std::vector<trtmc::SegmentResult> out = pipe->segment_batch(images);
```

What the library contract should be:

- local API accepts decoded image data, not `png` file paths
- local API returns one `SegmentResult` per item
- item grouping is semantic, not tensor-level

Why this should be a semantic batch API:

- the user starts from separate images, not one pre-packed `pixel_values [B, 3, H, W]`

How this wires into the current architecture:

- factory reads the bundle
- registry selects the segmentation plugin
- plugin constructs `SegmentPipeline` or `SamPipeline`
- that concrete pipeline may choose to implement `segment_batch(...)`

What this means for bundle build:

- the bundle must be built for the execution families this segmentation pipeline
  plans to use for multi-image inference
- that may mean batch buckets, resolution buckets, profiles, separate engines,
  or any combination
- this is exactly the kind of decision that should remain plugin-owned

### Story 4: Retrieval Engineer Wants 100 Text Embeddings

Who this user is:

- a retrieval, ranking, or search engineer

What they want:

- build one encoder bundle
- pass many texts in one call
- get one embedding per text

What they have at runtime:

- a vector of input texts
- sometimes also query/document pairs for reranking

What functionality they need:

- semantic multi-item batching
- one embedding per text
- stable item ordering
- no streaming

How they build the bundle:

```bash
trtmc-build build <bert-or-other-encoder-model> -o bert.trtfb
```

How they use the bundle at runtime:

CLI today for one text:

```bash
./build/trtmc embed bert.trtfb --prompt "Hello"
```

Target local C++ API:

```cpp
std::vector<std::string> texts = {
    "first text",
    "second text",
    "third text",
};

auto pipe = trtmc::load("bert.trtfb");
std::vector<trtmc::EmbeddingResult> out = pipe->embed_batch(texts);
```

Why this should be a semantic batch API:

- the user has texts, not packed `input_ids`
- tokenization and padding are part of the task-level contract

How this wires into the current architecture:

- registry selects the encoder plugin
- plugin creates `EncoderPipeline`
- `EncoderPipeline` may implement `embed_batch(...)` or `encode_batch(...)`

What this means for bundle build:

- builder must create the execution families the encoder pipeline expects for
  multi-text inference
- if the pipeline wants to batch across sequence lengths, the bundle must
  support the shape families that policy requires
- the public API should not expose those internal choices

### Story 5: Speech Product Engineer Wants 8 Audio Clips Transcribed

Who this user is:

- a speech product engineer building local ASR

What they want:

- build one ASR bundle
- pass several audio clips in one call
- get one transcript per clip

What they have at runtime:

- local library case:
  - decoded mono PCM samples in memory
  - sample rate per clip
- CLI case:
  - audio files on disk such as `.wav`

What functionality they need:

- semantic multi-item batching
- sample-rate aware audio input
- one text result per clip
- no streaming in this story

How they build the bundle:

```bash
trtmc-build build <whisper-or-other-asr-model> -o whisper.trtfb
```

How they use the bundle at runtime:

CLI today for one clip:

```bash
./build/trtmc transcribe whisper.trtfb --audio sample.wav
```

Local C++ API today for one clip:

```cpp
auto pipe = trtmc::load("whisper.trtfb");
trtmc::TextResult out = pipe->transcribe(samples, num_samples, 224, sample_rate);
```

Target local C++ API:

```cpp
std::vector<trtmc::AudioView> audio_inputs = {/* 8 clips */};

auto pipe = trtmc::load("whisper.trtfb");
std::vector<trtmc::TextResult> out = pipe->transcribe_batch(audio_inputs, 224);
```

Why this should be a semantic batch API:

- the user thinks in audio clips and transcripts
- mel extraction, padding, and per-item transcript mapping are task semantics

How this wires into the current architecture:

- registry selects the whisper plugin
- plugin creates `WhisperPipeline`
- `WhisperPipeline` may implement `transcribe_batch(...)`

What this means for bundle build:

- builder must package the execution families this ASR pipeline needs for
  multi-item transcription
- if the pipeline uses different execution families for encoder and decoder
  stages, that remains a plugin-owned design

### Story 6: Generative App Engineer Wants Many Prompts In One Call

Who this user is:

- an engineer building local batch generation for prompts, prompts plus images,
  or other stateful decode workloads

What they want:

- build one generative bundle
- submit several independent requests
- get one final generated result per request

What they have at runtime:

- multiple prompts
- optionally one image per prompt for multimodal generation
- one shared generation config or a small number of per-call options

What functionality they need:

- semantic batch generation
- separate request state per item
- one final result per item
- no streaming in this story

How they build the bundle:

Example:

```bash
trtmc-build build Qwen/Qwen3-0.6B -o qwen3.trtfb
```

How they use the bundle at runtime:

CLI today for one request:

```bash
./build/trtmc run qwen3.trtfb --prompt "Hello" --max-new-tokens 50 \
  --hf-python /opt/venv/bin/python
```

Target local C++ API:

```cpp
std::vector<std::string> prompts = {
    "Write a haiku about rain.",
    "Summarize this paragraph.",
    "Explain dynamic programming simply.",
};

auto pipe = trtmc::load("qwen3.trtfb");
std::vector<trtmc::TextResult> out =
    pipe->generate_text_batch(prompts, {.max_new_tokens = 128});
```

Why this should be a semantic batch API:

- the user thinks in prompts and responses
- request state, stopping policy, and output text are all semantic concepts

How this wires into the current architecture:

- registry selects the decoder or multimodal plugin
- plugin creates `TextGenerationPipeline` or `VLPipeline`
- that pipeline may implement `generate_text_batch(...)`

What this means for bundle build:

- builder must package the execution families this stateful pipeline intends to
  use
- if the pipeline uses different families for prefill and decode, or different
  families for multimodal encode vs decode, those are plugin decisions
- the public API still remains `generate_text_batch(...)`

### Story 7: Realtime Engineer Wants Streaming Output

Who this user is:

- an engineer building interactive text, ASR, or TTS

What they want:

- build one bundle
- start one or more requests
- receive partial output as it is produced

What they have at runtime:

- text prompt, audio clip, or TTS prompt
- callback or event sink for partial output

What functionality they need:

- semantic streaming
- modality-specific stream events
- a clear distinction between one stream and many simultaneous streams

How they build the bundle:

Examples:

```bash
trtmc-build build Qwen/Qwen3-0.6B -o qwen3.trtfb
trtmc-build build <whisper-or-other-asr-model> -o whisper.trtfb
trtmc-build build <bark-or-magpie-model> -o bark.trtfb
```

How they use the bundle at runtime:

Local C++ API target for one text stream:

```cpp
auto pipe = trtmc::load("qwen3.trtfb");
pipe->generate_text_stream(
    "Write a short welcome message.",
    {.max_new_tokens = 64},
    [](const trtmc::TextStreamEvent& ev) {
        // consume token/text deltas
    });
```

Local C++ API target for many text streams:

```cpp
std::vector<std::string> prompts = {
    "First request",
    "Second request",
};

auto pipe = trtmc::load("qwen3.trtfb");
pipe->generate_text_batch_stream(
    prompts,
    {.max_new_tokens = 64},
    [](const trtmc::TextStreamEvent& ev) {
        // ev.item_index identifies which request this delta belongs to
    });
```

Existing C++ API today for one audio stream:

```cpp
auto pipe = trtmc::load("bark.trtfb");
pipe->generate_audio_streaming(
    "A calm narration",
    {},
    [](const float* samples, int32_t n, int32_t sample_rate) {
        // consume PCM chunk
    });
```

Why this should remain semantic:

- text streaming, transcript streaming, and PCM chunk streaming are not the same
- a generic tensor stream would be much harder for users to consume

How this wires into the current architecture:

- same bundle load path
- concrete pipelines implement stream-capable methods when they want to
- unsupported methods continue to throw by default

What this means for bundle build:

- builder must package the execution families the streaming pipeline expects
- streaming itself is not a builder feature
- builder only needs to support the execution families the pipeline uses while
  producing incremental output

### Story 8: Workflow Engineer Wants Overnight Offline Batches

Who this user is:

- a workflow or platform engineer running large offline jobs

What they want:

- create a large dataset job
- retry failures
- inspect progress
- store outputs durably

What functionality they need:

- job lifecycle
- persistence
- retries
- partial failure handling
- monitoring

Where this belongs:

- not in the core local inference library
- this is a serving or workflow-layer concern

Why this story still matters:

- it should not leak back into the local `IPipeline` design
- the local library should focus on synchronous and callback-style in-process inference

## Current Gaps Against These Stories

The stories above are the requirements. The current codebase only partially
covers them.

### What Works Today In Principle

- Story 1 is mostly supported today through existing single-item semantic APIs in
  [pipeline.h](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/include/trtmc/pipeline.h)
  such as `generate(...)`, `segment(...)`, `embed(...)`, `transcribe(...)`, and
  `generate_audio(...)`.
- Story 7 has a narrow existing example through
  `generate_audio_streaming(...)` in
  [pipeline.h](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/include/trtmc/pipeline.h).
- Story 8 is intentionally out of scope for the local library.

Everything else is either missing as a public contract, limited to single-item
execution, or blocked by builder/runtime seams.

### Story-By-Story Support Matrix

| Story | Best public API | Current support | Main missing capability | Primary layer that must change first |
| --- | --- | --- | --- | --- |
| Story 1: one text answer | `generate_text(...)` | Partial: existing `generate(...)` already covers the simple case | clearer semantic naming, not core functionality | `header` |
| Story 2: packed tensors | `invoke(...)` | Missing | no generic tensor-level contract, no clean pipeline-facing invoke seam | `header`, then `pipeline`, then `runtime seam` |
| Story 3: 5 segmentation images | `segment_batch(...)` | Missing | no semantic batch API, no multi-image preprocessing path, common bundles still shaped as one-item execution | `header`, `pipeline`, and `builder` |
| Story 4: 100 text embeddings | `embed_batch(...)` | Missing | no semantic batch API, tokenization/padding path is single-item, common encoder bundles still shaped as one sequence | `header`, `pipeline`, and `builder` |
| Story 5: 8 audio transcriptions | `transcribe_batch(...)` | Missing | no semantic batch API, mel and decode flow is single-item, bundle/runtime may need multi-item execution families | `header`, `pipeline`, then `builder` |
| Story 6: many generative requests | `generate_text_batch(...)` | Missing | no semantic batch generation API, no per-item batch decode flow, runtime seam cannot yet select execution families cleanly | `header`, `pipeline`, `builder`, and `runtime seam` |
| Story 7: realtime streaming | `generate_text_stream(...)`, `transcribe_stream(...)`, `generate_audio_stream(...)`, batch stream variants | Partial: `generate_audio_streaming(...)` exists for one narrow case | no general semantic streaming surface, no text/asr stream APIs, no batch stream contract | `header`, then `pipeline` |
| Story 8: overnight jobs | not part of local `IPipeline` | Intentionally unsupported | job lifecycle is not a local-library concern | `serving/workflow layer`, not local inference |

How to read this matrix:

- `header` means the user cannot even express the story cleanly today
- `pipeline` means the story has to be implemented where batching and modality
  semantics actually live
- `builder` means the bundle may not yet contain an execution family suitable
  for that story
- `runtime seam` means the bundle may contain the right family, but the runtime
  cannot yet select or drive it cleanly

### Gap 1: The Public Library Contract Is Mostly Single-Item

Why this matters:

- Stories 2 through 7 all need either generic tensor invocation, semantic
  batching, semantic streaming, or batch-streaming.

Current state:

- [pipeline.h](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/include/trtmc/pipeline.h)
  exposes task-specific single-item calls.
- there is no public `invoke(...)`
- there is no public `segment_batch(...)`, `embed_batch(...)`,
  `transcribe_batch(...)`, or `generate_text_batch(...)`
- there is no public generic text-stream or ASR-stream API

Resulting gap:

- users can only express Story 1 directly today
- Stories 2 through 7 have no clear public contract even before implementation begins

### Gap 2: Preprocessing Is Still Single-Item In Key Places

Why this matters:

- Stories 3, 4, and 5 start from several independent semantic items, not one
  packed tensor.

Current state:

- vision preprocessing explicitly says only single-image input is supported in
  [image_preprocessor.h](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/src/runtime/domains/multimodal/image_preprocessor.h)
- the current library surface also takes one text, one image, or one audio clip
  at a time

Resulting gap:

- even if a model engine could execute a batch tensor, the library still lacks
  the semantic pack path needed for Story 3, Story 4, and Story 5

### Gap 3: Concrete Pipelines Are Mostly Written As Single-Item Flows

Why this matters:

- Stories 3 through 7 ultimately depend on concrete pipelines, not on the
  factory or registry

Current state:

- [segment_pipeline.cpp](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/src/runtime/pipelines/segment_pipeline.cpp)
  implements `segment(const float*, int32_t, int32_t)` and feeds one image
  through `model_->forward(...)`
- [encoder_pipeline.cpp](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/src/runtime/pipelines/encoder_pipeline.cpp)
  tokenizes one text and `encode_ids(...)` builds one `input_ids` tensor and one
  `attention_mask`
- [whisper_pipeline.cpp](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/src/runtime/pipelines/whisper_pipeline.cpp)
  transcribes one audio clip end to end: resample, mel extraction, encoder,
  cross-attention setup, decoder

Resulting gap:

- the current pipelines do not yet provide the semantic batching behavior that
  Stories 3 through 6 need
- this is the right place to add that behavior later, because batching policy
  should remain pipeline-owned

### Gap 4: Many Builders Still Materialize Single-Item Execution Families

Why this matters:

- if a story needs multi-item local inference, the bundle must contain an
  execution family that the pipeline can actually use for that story

Current state:

- [segformer.py](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/tensorrt_model_connect/tensorrt_model_connect/families/segformer.py)
  builds `pixel_values` as `(1, 3, H, W)`
- [sam.py](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/tensorrt_model_connect/tensorrt_model_connect/families/sam.py)
  builds the image encoder input as `(1, 3, image_size, image_size)`
- [modernbert.py](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/tensorrt_model_connect/tensorrt_model_connect/families/modernbert.py)
  builds `input_ids` and `attention_mask` as `(max_seq,)`
- [deberta.py](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/tensorrt_model_connect/tensorrt_model_connect/families/deberta.py)
  builds encoder inputs as `(max_seq_length,)`

Resulting gap:

- Story 3 and Story 4 are not just missing API surface; many common builder
  paths are still shaped like one-item execution
- the missing capability is not “dynamic shape everywhere”
- the missing capability is that each pipeline must be able to ask the builder
  for the execution families required by the stories it wants to support

### Gap 5: The Runtime Seam Does Not Yet Expose Execution-Family Selection Well Enough

Why this matters:

- Stories 2 through 7 eventually need some plugin-owned way to select and drive
  the right execution family at runtime
- sometimes that may be static buckets
- sometimes that may be multiple engines
- sometimes that may be optimization profiles

Current state:

- [trt_module.cpp](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/src/runtime/core/trt_module.cpp)
  already detects dynamic shapes and queries optimization profiles
- but `TrtModule::forward()` still behaves as a host-centric seam:
  upload inputs, enqueue, sync, download outputs
- `allocate_buffers()` and dynamic-shape setup are tied to profile 0
- there is no public or pipeline-facing way to say:
  - use execution family A for this call
  - use profile 1 instead of profile 0
  - keep outputs on device for the next stage

Concrete evidence that this matters:

- [magpie_tts.py](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/tensorrt_model_connect/tensorrt_model_connect/families/magpie_tts.py)
  already builds two optimization profiles for one decoder engine
- [magpie_pipeline.cpp](/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect/src/runtime/pipelines/magpie_pipeline.cpp)
  explicitly says batched prefill wants profile 1 but `TrtModule` does not yet
  expose that path

Resulting gap:

- dynamic shape is not the product requirement
- but profile-aware execution-family selection is a real runtime gap today
- without that seam, some pipelines will be forced into sequential fallback
  even when the builder already prepared a better execution family

## What Dynamic Shape Actually Means In This Proposal

Dynamic shape should not be treated as a top-level API feature.

The correct chain is:

- user story
- required execution families
- builder strategy
- runtime selection mechanism

That means:

- some stories may be satisfied with static single-purpose engines
- some stories may be best served by static buckets
- some stories may benefit from optimization profiles
- some stories may need multiple engines rather than one dynamic engine

So the real missing capability is not “support dynamic shape”.

The real missing capability is:

- let each plugin and pipeline choose the execution families it wants
- let the builder encode those families in the bundle
- let the runtime select among those families cleanly

That is the gap that Story 2 through Story 7 expose.

## What These Stories Mean For The Header

The header should stay small.

The lower-level generic layer should expose:

- `invoke(const InvokeRequest&)`

The higher-level semantic layer should expose the task calls users actually ask
for:

- single-item semantic calls such as `generate_text(...)`, `segment(...)`,
  `embed(...)`, `transcribe(...)`, and `generate_audio(...)`
- semantic batch calls such as `segment_batch(...)`, `embed_batch(...)`,
  `transcribe_batch(...)`, and `generate_text_batch(...)`
- semantic stream calls such as `generate_text_stream(...)`,
  `transcribe_stream(...)`, and `generate_audio_stream(...)`
- semantic batch-stream calls such as `generate_text_batch_stream(...)`,
  `transcribe_batch_stream(...)`, and `generate_audio_batch_stream(...)`

The rule is simple:

- generic layer is data-oriented
- semantic layer is task-oriented
- plugins and pipelines decide which methods they actually implement

## How This Wires Cleanly Into The Existing Architecture

The architecture should remain:

- thin factory
- thin registry
- plugin-owned construction
- pipeline-owned inference behavior

That means:

- `PipelineFactory` still only loads the bundle and resolves `runtime_strategy`
- `PipelineRegistry` still only maps strategy to plugin
- `IPipelinePlugin::create()` still only creates the concrete pipeline
- the concrete pipeline decides whether it implements:
  - `invoke(...)`
  - semantic single-item APIs
  - semantic batch APIs
  - semantic stream APIs

This keeps scalability where it belongs:

- public contract is shared
- implementation policy stays local to each plugin and pipeline

## What This Means For Builder Design

Builder design should also stay simple at the contract level.

The requirement is not:

- every model must use dynamic shape

The requirement is:

- each bundle must contain the execution families its pipeline plans to use for
  the user stories it chooses to support

Examples:

- a segmentation pipeline that wants multi-image batch inference needs a bundle
  that supports the batch and resolution families it plans to execute
- an encoder pipeline that wants many-text embedding needs a bundle that
  supports the sequence-length families it plans to execute
- a text-generation pipeline that wants batch generation or streaming needs a
  bundle that supports the execution families that pipeline uses for those flows

How that is achieved is intentionally not part of the public API contract:

- static shapes
- static buckets
- optimization profiles
- separate engines
- hybrid strategies

Those are builder and plugin decisions, not user-facing API concepts.

## Bottom Line

The user stories point to a simple API shape:

- one generic tensor-level API for advanced callers: `invoke(...)`
- semantic single-item APIs for ordinary tasks
- semantic batch APIs for multi-item tasks
- semantic stream APIs for realtime tasks

And they point to a simple wiring model:

- build a bundle with `trtmc-build`
- load that bundle with `trtmc::load(...)`
- let `runtime_strategy` choose the plugin
- let the concrete pipeline own which APIs it implements

That keeps the contract clear for users and keeps execution policy where it
scales best: inside the plugin and pipeline.
