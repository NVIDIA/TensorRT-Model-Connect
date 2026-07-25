---
title: C++ and C ABI
---

The public C++ API is centered on `include/trtmc/pipeline.h`.

## Load a bundle

```cpp
#include <trtmc/pipeline.h>

#include <iostream>

int main() {
    auto pipe = trtmc::load("/tmp/qwen3.trtfb", "/opt/venv/bin/python");
    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = 20;
    auto out = pipe->generate("The capital of France is", cfg);
    std::cout << out.text << "\n";
}
```

For full control, use `LoadOptions`:

```cpp
trtmc::LoadOptions options;
options.hf_python = "/opt/venv/bin/python";
options.runtime_cache_path = "/tmp/trtmc-rtx.cache";
options.cuda_graphs = true;
options.kv_cache_size_bytes = 512ULL * 1000ULL * 1000ULL;
options.backend_search_paths = {"/opt/trtmc/backends"};
options.model_plugin_search_paths = {"/opt/trtmc/models"};
options.set_tokens = {"runtime.prefer_gpu_greedy=true"};

auto pipe = trtmc::load("/tmp/model.trtfb", options);
```

`load()` supports both bundle shapes. A native bundle uses
`runtime_strategy`, the model-plugin index, and a backend DSO. A bundle with
`optimized_runtime.json` instead loads its exact embedded implementation DSO;
`model_plugin_search_paths` and `backend_search_paths` do not select that
implementation.

## Concurrent requests

Do not call request methods concurrently on one `IPipeline`. Pipeline
instances own mutable execution-context, stream, cache/state, and
adapter-binding data; the public interface does not promise per-instance
thread safety.

For native bundles, use independent instances or `PipelinePool`:

```cpp
#include <trtmc/runtime/pipeline_factory.h>
#include <trtmc/runtime/pipeline_pool.h>

auto pool = trtmc::PipelineFactory::from_bundle_pool(
    "/tmp/native-model.trtfb", 4);

// Each worker acquires one exclusive, move-only lane for one in-flight request.
auto lease = pool->acquire();
auto result = lease->generate("Hello");
```

`acquire()` waits for an available lane; `try_acquire()` reports exhaustion
without waiting. Destroying or moving over a lease releases its lane.
`PipelinePool` keeps mutable execution state isolated per lane and coordinates
adapter maintenance across lanes.

`from_bundle_pool()` does not support optimized-runtime bundles and throws
before loading their implementation DSO. The delegated runtime owns its own
batching and scheduling, so load those bundles with `trtmc::load()` or
`PipelineFactory::from_bundle()` and follow that provider's concurrency
contract.

## Result types

| Type | Returned by |
| --- | --- |
| `TextResult` | `generate()`, `transcribe()`, `transcribe_streaming()` |
| `ImageResult` | `generate_image()` |
| `AudioResult` | `generate_audio()`, `speak()` |
| `EmbeddingResult` | `embed()`, `encode()`, `solve()` |
| `SegmentResult` | `segment()` |
| `PromptedSegmentationResult` | `segment_prompted()`, `segment_prompted_text()` |
| `ClassificationResult` | `classify()` |
| `TextEmbedding` | `encode_text()` for diffusion text encoders |

`rerank()` returns a `float`, and `detect()` returns serialized detection JSON
as `std::string`. `generate_image_batch()` returns
`std::vector<ImageResult>`.

## GenerateConfig

`GenerateConfig` controls decoding and generation:

```cpp
trtmc::GenerateConfig cfg;
cfg.max_new_tokens = 128;
cfg.temperature = 0.7f;
cfg.top_k = 50;
cfg.top_p = 0.9f;
cfg.min_p = 0.0f;
cfg.seed = 1234;
cfg.guidance_scale = 3.5f;
cfg.num_steps = 28;
cfg.use_chat_template = true;
cfg.enable_thinking = false;
```

## Streaming transcription

```cpp
trtmc::TranscriptionStreamConfig cfg;
cfg.input_sample_rate = 16000;
cfg.att_context_left = 70;
cfg.att_context_right = 13;
cfg.emit_partial_results = true;

auto stream = pipe->create_transcription_stream(cfg);
auto partial = stream->accept_audio(samples, num_samples, false);
auto final = stream->finish();
```

## Offline transcription

`TranscriptionConfig` carries per-request offline ASR controls:

```cpp
trtmc::TranscriptionConfig cfg;
cfg.input_sample_rate = 16000;
cfg.max_output_tokens = 80;
cfg.beam_size = 2;
cfg.source_language = "en";
cfg.target_language = "fr";
cfg.task = trtmc::TranscriptionTask::kTranslate;
cfg.punctuation = true;
cfg.timestamps = true;
cfg.max_input_duration_seconds = 300.0F;
cfg.segment_duration_seconds = 20.0F;

auto result = pipe->transcribe(samples, num_samples, cfg);
for (const auto& segment : result.segments) {
    std::cout << segment.start_seconds << "\t" << segment.end_seconds
              << "\t" << segment.text << "\n";
}
```

`transcribe_batch(const std::vector<TranscriptionRequest>&)` preserves each
request's samples and config. The default implementation is sequential and
returns results in request order. Canary overrides it with native batches of up
to 16 encoder inputs and a 32-lane decoder, including batched beam search. The
legacy max-token/sample-rate overload is still supported.

## C ABI

The current C-linkage subset is a starting point for C++ shims and FFI
experiments:

```cpp
TrtmcPipelineOptions opts{};
opts.hf_python = "/opt/venv/bin/python";

trtmc::IPipeline* pipe = trtmc_create_pipeline_ex("/tmp/model.trtfb", &opts);
if (pipe == nullptr) {
    const char* err = trtmc_last_error();
    // Report err and stop.
}
const char* version = trtmc_version();
int has_trt = trtmc_has_trt();
```

The C-linkage surface currently exposes pipeline creation, error/version
queries, batched image generation through `trtmc_generate_batch()`, and
per-image cleanup through `trtmc_image_result_free()`. The caller owns the
output array, and must free each successful result's pixel buffer.

There is no exported pipeline-destroy function. Creation returns an
`IPipeline*`, and the public header uses C++ types even for its C-linkage
declarations. Do not expose that handle as a complete pure-C or
foreign-language ownership contract; wrap it in C++ or add a matching destroy
entry point first.

`TrtmcPipelineOptions::hf_python`, `runtime_cache`, and `cuda_graphs` are
consumed during creation. The current implementation does not consume the
legacy `max_new_tokens` or `image_path` fields; generation settings belong on
the request API.
