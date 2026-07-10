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
options.set_tokens = {"runtime.max_batch_size=1"};

auto pipe = trtmc::load("/tmp/model.trtfb", options);
```

## Result types

| Type | Returned by |
| --- | --- |
| `TextResult` | `generate()`, `transcribe()`, `transcribe_streaming()` |
| `ImageResult` | `generate_image()` |
| `AudioResult` | `generate_audio()`, `speak()` |
| `EmbeddingResult` | `embed()`, `encode()`, `solve()` |
| `SegmentResult` | `segment()` |
| `TextEmbedding` | `encode_text()` for diffusion text encoders |

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
cfg.beam_size = 4;
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
returns results in request order. The legacy max-token/sample-rate overload is
still supported.

## C ABI

The C ABI is for FFI and backward-compatible integrations:

```cpp
TrtmcPipelineOptions opts{};
opts.max_new_tokens = 50;
opts.hf_python = "/opt/venv/bin/python";

trtmc::IPipeline* pipe = trtmc_create_pipeline_ex("/tmp/model.trtfb", &opts);
const char* err = trtmc_last_error();
const char* version = trtmc_version();
int has_trt = trtmc_has_trt();
```

The C ABI currently exposes creation and query entry points. The returned pointer is produced by the C++ runtime; FFI users should wrap it in a C++ ownership shim or add a matching destroy function before exposing it across a pure-C or foreign-language boundary.
