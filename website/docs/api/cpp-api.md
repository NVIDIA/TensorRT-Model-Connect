---
title: C++ API and C-Linkage Subset
---

The public C++ API is centered on `include/trtmc/pipeline.h`.

:::note C-linkage status

The C++ API is the primary native interface. The shared pipeline C-linkage
header exposes a useful C++-compatible subset for shims and FFI experiments,
but it is not yet a complete stable pure-C ownership API. Separately versioned
model headers, such as SAM2 and SAM2-HOI video, define their own ownership and
compatibility contracts.

:::

## Load a bundle

```cpp
#include <trtmc/pipeline.h>

#include <iostream>

int main() {
    auto pipe = trtmc::load("/tmp/qwen3.bundle", "/opt/venv/bin/python");
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
options.config_path = "/etc/trtmc/runtime.json";
options.backend_search_paths = {"/opt/trtmc/backends"};
options.model_plugin_search_paths = {"/opt/trtmc/models"};
options.set_tokens = {"runtime.prefer_gpu_greedy=true"};

auto pipe = trtmc::load("/tmp/model.bundle", options);
```

`load()` supports both bundle shapes. A native bundle uses
`runtime_strategy`, the model-plugin index, and a backend DSO. A bundle with
`optimized_runtime.json` instead loads its exact embedded implementation DSO;
`model_plugin_search_paths` and `backend_search_paths` do not select that
implementation.

`kv_cache_size_bytes` is a route-specific override, not a safe generic default.
Set it only for a decoder bundle built with runtime-resizable KV support, for
example:

```cpp
options.kv_cache_size_bytes = 512ULL * 1000ULL * 1000ULL;
```

Leave it at zero for full-context native-KV Qwen3/Llama bundles. Those bundles
own one fixed physical capacity and reject every nonzero override.

`config_path` is the library equivalent of runtime `--config` and accepts a
schema-driven JSON file on the C++ path. `set_tokens` supplies repeatable
`namespace.field=value` session overrides; those values win over the file.
The current Qwen Edge-LLM optimized implementation rejects both runtime config
surfaces instead of silently ignoring them.

## SAM2-HOI video C ABI

SAM2-HOI owns a versioned, fixed five-JPEG API in
`include/trtmc/models/sam2_hoi_video.h`. The symbols are exported by
`libtrtmc_model_sam2_hoi.so`; they are not methods on the shared `IPipeline`
interface and there is no generic `trtmc` video-tracking command.

```cpp
#include <trtmc/models/sam2_hoi_video.h>

#include <cstdio>

int main() {
    TrtmcSam2HoiVideoSession* session =
        trtmc_sam2_hoi_video_create_from_bundle_v1(
            "/tmp/sam2-hoi-tracking.bundle",
            "/opt/trtmc/models",
            "/opt/trtmc/backends");
    if (session == nullptr) {
        std::fprintf(stderr, "%s\n", trtmc_sam2_hoi_video_last_error());
        return 1;
    }

    TrtmcSam2HoiVideoRunResultV1 result{};
    const int32_t status = trtmc_sam2_hoi_video_run_jpeg_files_v1(
        session,
        "000000.jpg", "000001.jpg", "000002.jpg", "000003.jpg", "000004.jpg",
        "/tmp/tracking.json", "/tmp/tracking-masks",
        &result, sizeof(result));
    if (status != TRTMC_SAM2_HOI_VIDEO_STATUS_OK) {
        std::fprintf(stderr, "%s\n", trtmc_sam2_hoi_video_last_error());
    }
    trtmc_sam2_hoi_video_session_destroy(session);
    return status == TRTMC_SAM2_HOI_VIDEO_STATUS_OK ? 0 : 1;
}
```

The caller owns the bundle and session. Supply exactly five nonempty JPEG file
paths in temporal order. Both output paths must be nonempty to materialize the
schema-version-1 tracking JSON and per-frame uint8 NumPy masks; pass two empty
strings for the benchmark discard path. Exactly one empty output path is an
invalid argument. Successful version-1 calls report a 64-byte scalar result
with the produced frame count; the current E2E profile requires that count to
be five. No output allocation crosses the ABI.

Calls on one session must be serialized. An argument or ABI preflight failure
does not poison the session, while a failure after JPEG processing begins does.
Always read `trtmc_sam2_hoi_video_last_error()` on a nonzero status and destroy
the session, including after failure.

## Inspect a bundle without loading it

`include/trtmc/bundle.h` exposes metadata inspection independently of
`IPipeline` construction:

```cpp
#include <trtmc/bundle.h>

#include <iostream>

int main() {
    const std::string path = "/tmp/model.bundle";
    if (!trtmc::IsBundle(path)) {
        return 1;
    }

    trtmc::BundleInfo info = trtmc::InspectBundle(path);
    std::cout << info.model_id << "\n"
              << info.family << "\n"
              << info.runtime_strategy << "\n";

    for (const trtmc::BundleSectionInfo& section : info.sections) {
        std::cout << section.name << " " << section.offset << " "
                  << section.size << "\n";
    }
}
```

`IsBundle()` checks the `.bundle` magic bytes; it is not a full compatibility
or engine-load proof. `InspectBundle()` returns `BundleInfo`, including model,
precision, TensorRT/ABI, shape, tokenizer, runtime-strategy, and section
metadata. Each `BundleSectionInfo` contains the section name, byte offset, and
size. `BundleInfo::max_batch_size` is a `MaxBatchSize` envelope with separate
`dit`, `text_encoder`, and `vae` caps; absent JSON values currently default to
one.

See [Bundle Format](../architecture/bundle-format.md) for layout and
compatibility semantics.

## Tokenizer API

`include/trtmc/tokenizer.h` defines the public `ITokenizer` interface:

```cpp
std::vector<int32_t> encode(const std::string& text) const;
std::string decode(const std::vector<int32_t>& ids) const;
int32_t id_for_token(std::string_view token) const;
std::string token_for_id(int32_t id) const;
```

The factories return an owning `std::unique_ptr<ITokenizer>`:

| Factory | Input contract |
| --- | --- |
| `CreateVocabTokenizer()` | An ordered token vocabulary. |
| `CreateIpaTokenizer()` | Phoneme dictionary, heteronyms, vocabulary, and config byte buffers. |
| `CreateBpeTokenizer()` | A tokenizer JSON buffer; special tokens are off by default. |
| `CreateWordPieceTokenizer()` | A tokenizer JSON buffer; special tokens are on by default. |
| `CreateUnigramTokenizer()` | A tokenizer JSON buffer; special tokens are on by default. |

These factories expose tokenizer mechanics; they do not choose the correct
tokenizer or special-token policy for a bundle. Prefer `trtmc::load()` when the
model-owned runtime should resolve and validate its packaged tokenizer assets.

## Audio and image I/O helpers

`include/trtmc/trtmc_io.hpp` exposes convenience helpers in `trtmc::io`:

| Helper | Contract |
| --- | --- |
| `write_wav(AudioResult, path)` | Writes mono IEEE float32 WAV. |
| `read_wav(path)` | Reads float32 or int16 WAV, downmixes channels to mono, and returns `AudioResult`. |
| `read_image(path)` | Returns `LoadedImage` with float RGB HWC pixels in `[0, 1]`; decode failure can return an empty value. |
| `save_png(path, pixels, width, height)` | Writes float RGB HWC pixels after clamping and uint8 conversion. |
| `save_png(ImageResult, path)` | Writes the first frame only and expects at least `H*W*3` values. |
| `decode_image(path, h, w)` | Legacy wrapper; prefer `read_image()`. |

The header provides WAV implementations inline; image decoding and PNG writing
link through `trtmc_core`. The `ImageResult` overload follows the current
three-channel HWC producer convention. It does not resolve the public
`ImageResult` layout inconsistency described under [Result types](#result-types)
and should not be used as evidence that every model emits the same strides.

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
    "/tmp/native-model.bundle", 4);

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
| `TranscriptionStreamResult` | `ITranscriptionStream::accept_audio()` and `finish()` |
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

The public result fields are:

| Type | Fields and shape |
| --- | --- |
| `TranscriptionSegment` | `start_seconds`, `end_seconds`, `text`, and segment `token_ids`. |
| `TextResult` | `text`, output `token_ids`, provider-populated `setup_ms`, `prefill_ms`, and `decode_ms`, plus timestamped `segments` when requested and supported. |
| `TranscriptionStreamResult` | Current `text` and `token_ids`, `is_final`, the current accepted-chunk counter in `chunk_index`, cumulative `accepted_samples`, and the configured input rate reported in `sample_rate`. |
| `ImageResult` | `pixels`, `height`, `width`, `channels`, and `num_frames`; the unresolved layout caveat is below. |
| `AudioResult` | Mono float32 `samples` in `[-1, 1]`, `num_samples`, and `sample_rate`. |
| `EmbeddingResult` | Flat `data` and embedding `dim`. |
| `SegmentResult` | Class-index `mask` in `[H, W]`, `height`, and `width`. |
| `PromptedSegmentationResult` | Logit `masks` in `[num_masks, H, W]`, `iou_scores`, absolute-pixel `boxes` in `[num_masks, 4]`, `num_masks`, `height`, and `width`. |
| `ClassificationResult` | Class `logits`, `top_class`, and `top_score`. |
| `TextEmbedding` | Flat `data` and its explicit `shape`. |

Additional `IPipeline` capability and metadata methods are:

| Method | Contract |
| --- | --- |
| `default_max_new_tokens()` | Runtime-owned default used when a caller does not supply a positive request limit. |
| `supports_image_generation()` | Reports whether image-generation entry points are implemented. |
| `generate_audio_streaming()` | Streams generated PCM chunks through an `AudioChunkCallback`. |
| `model_id()` | Returns the loaded model identifier. |
| `pipeline_type()` | Returns the concrete runtime pipeline type used in capability errors and diagnostics. |

`ImageResult::pixels` always has
`num_frames * height * width * channels` float32 values in `[0, 1]`, and a
single image has `num_frames == 1`. The indexing contract is currently
inconsistent in the codebase: the public header comments describe channel-first
`[C, H, W]` data (and the C-linkage buffer as flattened `C*H*W`), while the
current Flux, Wan, and SANA-WM producers write interleaved `[H, W, C]` or
`[T, H, W, C]` data. Until a code change selects and enforces one public
layout, portable callers must not infer strides from the header or this manual;
verify the selected pipeline's implementation/evidence and normalize the buffer
at the application boundary.

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
| `source_language_token_id`, `forced_bos_token_id` | Request-level M2M-100/NLLB language framing. Both default to `-1` (disabled); enabled values must be non-negative. The source token is appended after source EOS, and forced BOS becomes the decoder's first token. |
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

`TranscriptionStreamConfig` fields are:

| Field | Contract |
| --- | --- |
| `input_sample_rate` | Source PCM sample rate; the stream validates/converts it for the selected model. |
| `max_new_tokens` | Per-stream decoding limit. |
| `att_context_left`, `att_context_right` | Cache-aware encoder context measured in 80 ms frames. Common right-context values 0, 1, 6, and 13 correspond to 80, 160, 560, and 1120 ms chunks. |
| `use_cache` | Reuse encoder attention/convolution caches between chunks. |
| `use_feature_cache` | Reuse mel/pre-encoder overlap between chunks. |
| `emit_partial_results` | Permit non-final text results from `accept_audio()`. |
| `online_normalization` | Request online feature normalization when the selected checkpoint supports it. |
| `pad_and_drop_preencoded` | Select the padded/drop-preencoded first-chunk path instead of requiring a separate first-step encoder plan. |
| `language` | Prompt-dictionary tag such as `en-US`, `es-ES`, or `auto`; empty selects prompt index 0, and bundles without a prompt kernel ignore it. |

The current Nemotron streaming RNNT path requires both caches. It rejects
`online_normalization`, requires a matching right-context encoder section, and,
when `pad_and_drop_preencoded` is false, requires a first-step encoder section.

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

trtmc::IPipeline* pipe = trtmc_create_pipeline_ex("/tmp/model.bundle", &opts);
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

`trtmc_create_pipeline(bundle_path, flags)` is the legacy convenience entry
point. The current implementation ignores `flags`, constructs default
`TrtmcPipelineOptions`, and delegates to `trtmc_create_pipeline_ex()`.
`trtmc_pipeline_t` is only an alias for `trtmc::IPipeline*`; despite its name it
is not a separately opaque C handle.

`trtmc_image_result_t` contains the allocated `pixels` pointer, `height`,
`width`, `channels`, `num_frames`, and `num_pixels`. The last field is the total
number of floats across all frames and dimensions; it is the safe allocation
length even while the public indexing order remains unresolved.

For `num_prompts > 0`, a non-null `out_results` must point to a writable array
of at least `num_prompts` entries. Release any pixel buffers from an earlier
call before reusing that array. On success, release each returned buffer with
`trtmc_image_result_free()`. The function does not promise to initialize the
array on every error path, so callers should initialize their array before the
call and treat entries as owned results only after a successful return.
`trtmc_image_result_free()` sets a released `pixels` pointer back to null.

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

{/* Collaborative review anchor: batch 2. */}
