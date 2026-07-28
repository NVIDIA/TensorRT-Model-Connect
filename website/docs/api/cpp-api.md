---
title: C++ API and C-Linkage Subset
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
adapter maintenance across lanes. `size()` returns the fixed lane count and
`available()` returns the currently unleased count.

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

Additional `IPipeline` capability and metadata methods are:

| Method | Contract |
| --- | --- |
| `default_max_new_tokens()` | Runtime-owned default used when a caller does not supply a positive request limit. |
| `supports_image_generation()` | Reports whether image-generation entry points are implemented. |
| `generate_audio_streaming()` | Streams generated PCM chunks through an `AudioChunkCallback`. |
| `model_id()` | Returns the loaded model identifier. |
| `pipeline_type()` | Returns the concrete runtime pipeline type used in capability errors and diagnostics. |

`ImageResult::pixels` is frame-major, interleaved float32 data in
`[T, H, W, C]` order with values in `[0, 1]`. Its length is
`num_frames * height * width * channels`; a single image has
`num_frames == 1`.

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

The complete field inventory is:

| Fields | Contract |
| --- | --- |
| `max_new_tokens`, `num_samples` | Output limit and non-autoregressive sample count. |
| `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `eos_token_id` | Token sampling and termination controls. |
| `guidance_scale`, `cfg_scale`, `num_steps`, `sde_gamma` | Diffusion, flow-matching, and conditional-guidance controls; negative sentinel values select model defaults where supported. |
| `initial_latents`, `condition_latents`, `condition_mask`, `sampling_steps`, `sde_noises` | Optional packed raw-state inputs. Shapes remain model-owned and must match the selected bundle contract. |
| `negative_prompt`, `height`, `width` | Text-to-image negative prompt and output-size overrides. Empty or non-positive values select bundle defaults. |
| `text_generation_mode`, `block_length`, `confidence_threshold` | Text-diffusion or speculative mode, block length, and confidence threshold. |
| `tail_frames` | Additional speech-to-speech frames after the input. |
| `use_chat_template`, `enable_thinking` | Tokenizer chat-template and reasoning-mode selection. |
| `stop_on_boxed_answer`, `stop_check_interval` | Optional boxed-answer stopping behavior and polling interval. |
| `lora_adapter_id` | Loaded dynamic adapter ID. Empty selects the base model. |

## Dynamic LoRA lifecycle

Check `supports_lora_adapters()` before maintenance. A LoRA-capable
`IPipeline` exposes `load_lora_adapter(adapter_id, adapter_path)`,
`unload_lora_adapter(adapter_id)`, and `loaded_lora_adapters()`.
`GenerateConfig::lora_adapter_id` selects one registered adapter for a request;
an empty value clears adapter bindings and uses the base model.

```cpp
if (!pipe->supports_lora_adapters()) {
    throw std::runtime_error("bundle was not built for dynamic LoRA");
}

pipe->load_lora_adapter("product-style", "/tmp/my-peft-adapter");
trtmc::GenerateConfig cfg;
cfg.lora_adapter_id = "product-style";
auto result = pipe->generate("Describe the image.", image, height, width, cfg);
pipe->unload_lora_adapter("product-style");
```

Qwen-VL accepts a standard PEFT directory containing `adapter_config.json`
and `adapter_model.safetensors`. Loading fails when the engine has no dynamic
LoRA inputs, the ID is empty, the directory or files are invalid, the PEFT
mode is unsupported, tensors do not match the engine targets/shapes/dtypes, or
the adapter rank exceeds the engine capacity. Selecting or unloading an
unknown ID throws. Loading the same ID replaces its cached weights; unloading
an active ID first clears the current binding. A request that already acquired
adapter weights keeps shared ownership until it finishes, while subsequent
selection of an unloaded ID fails.

One `IPipeline` still must not execute concurrent requests. For multiple
lanes, perform adapter maintenance through `PipelinePool`:

```cpp
if (!pool->supports_lora_adapters()) {
    throw std::runtime_error("one or more lanes do not support dynamic LoRA");
}
pool->load_lora_adapter("product-style", "/tmp/my-peft-adapter");
auto ids = pool->loaded_lora_adapters();
pool->unload_lora_adapter("product-style");
```

Pool maintenance blocks new `acquire()` calls and waits for all outstanding
leases to return before touching adapters. Loading applies the ID to every
lane, skips lanes that already contain it, and rolls back newly loaded lanes
if a later lane fails. `supports_lora_adapters()` is true only when every lane
supports the feature. Unloading removes the ID from every lane and throws when
none contains it. `loaded_lora_adapters()` also waits for the maintenance
barrier and returns the shared registry view.

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

## C-linkage C++ subset

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

For `num_prompts > 0`, a non-null `out_results` must point to a writable array
of at least `num_prompts` entries. Release any pixel buffers from an earlier
call before reusing that array: `trtmc_generate_batch()` zero-initializes every
entry before validating the remaining arguments. On success, release each
returned buffer with `trtmc_image_result_free()`. On any nonzero return, every
entry remains zero-initialized, `pixels` is null, and there is no allocation to
release. `trtmc_image_result_free()` is a no-op for such a zero-initialized
entry and sets a released `pixels` pointer back to null.

There is no exported pipeline-destroy function. Creation returns an
`IPipeline*`, and the public header uses C++ types such as `std::uint64_t` even
for its C-linkage declarations. This is not a C-compatible header or a
complete stable C ABI. Do not expose that handle as a pure-C or
foreign-language ownership contract; wrap it in C++ or first design an opaque
C handle with a matching destroy entry point.

`TrtmcPipelineOptions::hf_python`, `runtime_cache`, and `cuda_graphs` are
consumed during creation. The current implementation does not consume the
legacy `max_new_tokens` or `image_path` fields; generation settings belong on
the request API.
