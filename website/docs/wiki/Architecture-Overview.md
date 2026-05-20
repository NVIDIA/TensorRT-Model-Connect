# Architecture Overview

## Document Control

| Field | Value |
|-------|-------|
| Document ID | ARCH-001 |
| Version | 2.0 |
| Status | NORMATIVE |
| Classification | ISO 26262-6 Section 7 -- Software Architectural Design |
| Author | Safety Architecture Team (TensorRT-Model-Connect Team) |
| Reviewer | Independent Review Required (TBD — assign before merge) |
| Review Status | Pending independent review |
| Last Updated | 2026-03-30 |
| Supersedes | ARCH-001 v1.0 (aspirational architecture document) |

This document describes the software architecture of the tensorrt-model-connect system as implemented in the codebase. All file paths referenced in this document have been verified to exist. Aspirational or planned changes are explicitly marked as **PLANNED** in Section 11.

---

## 1. Scope and Purpose

This document defines the software architectural design for the tensorrt-model-connect system: a two-phase inference platform that converts HuggingFace models into optimized TensorRT bundles (Python build phase) and runs autoregressive or single-pass inference from those bundles (C++ runtime phase).

The scope covers:

- The Python builder package (`tensorrt_model_connect/`) and its plugin-based family architecture.
- The C++ runtime and its manifest-registered plugin-based pipeline dispatch.
- The `.trtfb` bundle format that bridges the two phases.
- Core runtime abstractions: `TrtModule`, `KvCache`, `RecurrentState`.
- All 14 concrete pipeline implementations and 25 runtime strategies dispatched via a manifest-registered plugin registry.

---

## 2. System Architecture

The system operates in two strictly separated phases:

| Phase | Language | Entry Point | Input | Output |
|-------|----------|-------------|-------|--------|
| **Build** | Python | `./build/trtmc build` / `tensorrt_model_connect.build()` | HF repo ID or local directory | `.trtfb` bundle |
| **Run** | C++ | `trtmc run` / `trtmc::load()` / C ABI | `.trtfb` bundle | Task-specific results |

The bundle is the sole interface between the two phases. The C++ runtime never reads HuggingFace model directories directly. All model-specific architectural decisions (attention type, normalization, activation functions, weight layout) are baked into the TRT engine plan at build time.

```
  HuggingFace Model
        |
        v
  [Python Builder]  -- plugin dispatch, graph construction, engine compilation
        |
        v
    .trtfb Bundle   -- self-describing: engine plan(s) + tokenizer + config JSON
        |
        v
   [C++ Runtime]    -- registry-based plugin dispatch, pipeline assembly, autoregressive loop
        |
        v
  Task-Specific Output (text, audio, image, segmentation, embedding)
```

---

## 3. Python Builder Architecture

The Python builder is a fully plugin-based system. Adding a new model family requires only a single Python file with zero edits to shared code.

### 3.1 Package Structure

- **Package root**: `tensorrt_model_connect/tensorrt_model_connect/`
- **Entry points**: `build_cli.py` (builder CLI), `__init__.py` (Python API), `__main__.py`

### 3.2 Orchestration Flow

The orchestrator in `tensorrt_model_connect/tensorrt_model_connect/engine_builder.py` executes:

1. **Resolve model** -- download from HuggingFace or use local directory.
2. **Parse config** -- `config.py` reads `config.json` into `ModelConfig`.
3. **Find plugin** -- `families/__init__.py` calls `find_plugin(model_type)`.
4. **Load weights** -- plugin's `load_weights()` calls `checkpoint_mapper.py` to load safetensors into a `WeightDict`, applying family-specific transforms.
5. **Build engine(s)** -- plugin's `build_engine()` constructs TRT networks. For VL models, `build_vision_engine()` builds a second engine. For diffusion models, `build_components()` builds text encoder(s), denoiser, and VAE decoder.
6. **Write bundle** -- `bundle_writer.py` packages engine plan(s), tokenizer files, and config JSON into a `.trtfb` file.

### 3.3 Family Plugin Protocol

Defined in `tensorrt_model_connect/tensorrt_model_connect/families/base.py` as a Python `Protocol`:

```python
class FamilyPlugin(Protocol):
    name: str
    def matches(self, model_type: str) -> bool: ...
    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict: ...
    def build_engine(self, config: ModelConfig, weights: WeightDict,
                     max_cache_length: int, *, verbose: bool = False) -> bytes: ...
    # Optional: build_vision_engine(), get_vl_config(),
    #           build_components(), get_diffusion_config()
```

### 3.4 Plugin Auto-Discovery

`tensorrt_model_connect/tensorrt_model_connect/families/__init__.py` uses `pkgutil.iter_modules()` to scan all `.py` files in the families directory. Any module exposing a `plugin` attribute is registered automatically. There are currently 67 Python files in the families directory (65 plugins + `base.py` protocol + `__init__.py` auto-discovery module). Batch onboarding can use the autopilot system (`scripts/autopilot/autorun.py`), which launches configurable agent CLI sessions that follow `AGENTS.md` and the repo-local skills.

Key discovery functions:
- `find_plugin(model_type)` -- matches standard models by HF `model_type`.
- `find_diffusion_plugin(pipeline_class)` -- matches diffusion models by diffusers pipeline class name.

### 3.5 Three-Layer Graph Construction

The TRT graph building is layered:

| Layer | File | Responsibility |
|-------|------|----------------|
| 1. Atomic ops | `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` | Tensor-in/tensor-out ops: RoPE, RMSNorm, attention, ALiBi, conv, etc. |
| 2. Composable blocks | `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py` | Multi-op blocks: SwiGLU MLP, GELU MLP, attention block, apply_norm |
| 3. Standard decoder | `tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py` | Full decoder engine: embedding, N transformer layers, LM head |

Most decoder families call into `standard_decoder_builder.py`. Specialized architectures (Mamba, Whisper, Bark, diffusion) build custom graphs in their plugin or dedicated builder modules.

### 3.6 Specialized Builders

Beyond the standard decoder, the build package contains dedicated engine builders:

| Builder | File | Purpose |
|---------|------|---------|
| Vision (Qwen VL) | `qwen_vl_vision_builder.py` | Qwen2.5-VL (3D RoPE) and Qwen3-VL (DeepStack) vision encoders |
| Vision (InternViT) | `internvit_vision_builder.py` | InternVL vision encoder |
| Vision (ONNX) | `onnx_vision_builder.py` | ONNX-traced vision encoders |
| Vision (Phi4) | `phi4mm_vision_builder.py` | Phi-4 multimodal vision encoder |
| Vision (CLIP) | `clip_encoder_builder.py` | CLIP text/image encoder |
| Encoder | `encoder_builder.py` | BERT/encoder-only models |
| Encoder (Mistral) | `mistral_encoder_builder.py` | Mistral encoder for embedding/reranking |
| Encoder (Qwen3) | `qwen3_encoder_builder.py` | Qwen3 text encoder |
| T5 Encoder | `t5_encoder_builder.py` | T5 text encoder for diffusion |
| Causal VAE 3D | `causal_vae_3d_builder.py` | Wan2.1 causal 3D VAE decoder |
| VAE 2D | `vae_2d_builder.py` | 2D VAE decoder |
| FLUX DiT | `flux_dit_builder.py` | FLUX denoiser (DiT) |
| FLUX2 DiT | `flux2_dit_builder.py` | FLUX.2 denoiser variant |
| FLUX VAE | `flux_vae_builder.py` | FLUX VAE decoder |
| Z-Image DiT | `z_image_dit_builder.py` | Z-Image denoiser (DiT) |
| Standard DiT | `standard_dit_builder.py` | Generic DiT builder |
| EnCodec | `encodec_builder.py` | EnCodec audio codec |
| NanoCodec | `nanocodec_builder.py` | NanoCodec audio codec |

### 3.7 Debug Runner

`tensorrt_model_connect/tensorrt_model_connect/debug_runner.py` provides pure-Python TRT inference runners that mirror C++ runtime behavior:

- `TrtRunner` -- standard decoder with device-resident KV cache.
- `MambaTrtRunner` -- SSM with device-resident conv + SSM state.
- `VLTrtRunner` -- vision encoder + text decoder with image preprocessing.

These are used by diff-testing tools (`tools/diff_logits.py`, `tools/diff_layers.py`, `tools/diff_vl.py`) and the E2E test harness.

Additionally, `tensorrt_model_connect/tensorrt_model_connect/diffusion_runner.py` provides a `DiffusionRunner` for pure-Python TRT inference of diffusion models (text encoding, denoising loop, VAE decode), following the same pattern as the decoder `TrtRunner`.

### 3.8 Scheduler Package

`tensorrt_model_connect/tensorrt_model_connect/schedulers/` contains Python diffusion noise schedulers used during build-time validation and debug runner inference. Currently implements `flow_match_euler.py` (Flow Matching Euler Discrete), matching the C++ `FlowMatchEulerScheduler`.

---

## 4. C++ Runtime Architecture

### 4.1 Public API

The public API consists of three headers:

| Header | Contents |
|--------|----------|
| `include/trtmc/pipeline.h` | `IPipeline` (abstract base with 14 virtual methods), result types (`TextResult`, `ImageResult`, `AudioResult`, `EmbeddingResult`, `SegmentResult`, `TextEmbedding`), `GenerateConfig`, factory function `trtmc::load()`, C ABI functions |
| `include/trtmc/bundle.h` | `BundleInfo` struct, `InspectBundle()`, `IsBundle()` |
| `include/trtmc/tokenizer.h` | `ITokenizer` interface, factory functions for `VocabTokenizer`, `BpeTokenizer`, `WordPieceTokenizer`, `UnigramTokenizer`, `IpaTokenizer` |

`IPipeline` defines 14 virtual methods spanning all modalities:

- `generate()` (text-only and text+image overloads)
- `encode_text()`, `generate_image()` (two overloads)
- `generate_audio()`, `transcribe()`, `speak()`
- `embed()`, `rerank()`, `segment()`, `encode()`
- `solve()`, `detect()`

Each method has a default implementation that throws `std::runtime_error`, so pipeline classes only override the methods they support.

### 4.2 C ABI Entry

- **File**: `src/cabi/api/trtmc_c.cpp` (~108 LOC)
- Exposes `trtmc_create_pipeline_ex()`, `trtmc_last_error()`, `trtmc_version()`, `trtmc_has_trt()`.
- Delegates immediately to `PipelineFactory::from_bundle()`.

### 4.3 Pipeline Factory (Registry-Based Dispatch)

- **Header**: `include/trtmc/runtime/pipeline_factory.h`
- **Implementation**: `src/runtime/registry/pipeline_factory.cpp` (~124 LOC)
- **Registry**: `include/trtmc/runtime/pipeline_registry.h`, `src/runtime/registry/pipeline_registry.cpp`
- **Plugin interface**: `include/trtmc/runtime/pipeline_plugin.h`, `src/runtime/registry/pipeline_plugin.cpp`

The factory is the single entry point for creating pipelines from bundles. It performs:

1. `ReadBundleFile()` -- parse the `.trtfb` file.
2. Extract `config.json` section from the bundle.
3. `extract_json_string()` -- read the `runtime_strategy` field.
4. `normalize_legacy_strategy()` -- rewrite legacy ambiguous strategy strings (e.g., `"diffusion"` to `"diffusion_wan"`, `"text_to_audio"` to `"text_to_audio_bark"`).
5. `PipelineRegistry::instance().lookup(strategy)` -- look up the manifest-registered plugin.
6. `parse_base_config()` -- parse universal config fields into `BaseConfig`.
7. `plugin->create(ctx)` -- delegate pipeline construction to the plugin.

**Manifest-registered plugin architecture**: Each model runtime folder in `src/runtime/models/` defines a plugin class implementing `IPipelinePlugin` and exposes a registrar function via `REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST`:

```cpp
// Inside namespace trtmc in each model plugin .cpp:
REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_decoder_plugin, DecoderPlugin,
                                         "decoder_kv_cache", "decoder_moe");
```

`PipelineRegistry` is a singleton mapping strategy strings to `IPipelinePlugin*` instances. Adding a new strategy requires only a new model runtime folder plus one manifest entry -- no edits to `pipeline_factory.cpp` or any central dispatch logic.

The plugin manifest `cmake/trtmc_pipeline_plugins.cmake` lists each plugin source and registrar function. Model plugin files register 25 strategy strings:

| Model Plugin | Strategies Registered |
|-------------|----------------------|
| `text_generation/plugin.cpp` | `decoder_kv_cache`, `decoder_moe` |
| `recurrent/ssm_plugin.cpp` | `ssm_recurrent` |
| `recurrent/rwkv_plugin.cpp` | `rwkv_recurrent` |
| `recurrent/hybrid_plugin.cpp` | `hybrid_mamba_attention` |
| `encoder/plugin.cpp` | `encoder_only`, `embedding`, `reranking`, `neural_operator` |
| `vision_language/plugin.cpp` | `vision_language` |
| `segmentation/plugin.cpp` | `segmentation`, `prompted_segmentation` |
| `encoder/object_detection_plugin.cpp` | `object_detection` |
| `whisper/plugin.cpp` | `speech_to_text` |
| `bark/plugin.cpp` | `text_to_audio_bark` |
| `magpie/plugin.cpp` | `text_to_audio_magpie` |
| `speech/plugin.cpp` | `speech_to_speech` |
| `omni/plugin.cpp` | `omni_multimodal` |
| `t5/plugin.cpp` | `text_to_text` |
| `marian/plugin.cpp` | `marian_translation` |
| `seq2seq/plugin.cpp` | `seq2seq_encoder_decoder` |
| `flux/plugin.cpp` | `diffusion_flux` |
| `wan/plugin.cpp` | `diffusion_wan`, `diffusion_pixart` |
| `z_image/plugin.cpp` | `diffusion_zimage` |
| `cmake/trtmc_pipeline_plugins.cmake` | source/anchor manifest for generated linker retention |

### 4.4 Configuration

- **Header**: `include/trtmc/runtime/pipeline_plugin.h`
- **Implementation**: `src/runtime/registry/pipeline_plugin.cpp`

`BaseConfig` is a universal struct with ~14 fields that every pipeline needs (vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, head_dim, max_cache_length, bos/eos IDs, runtime_strategy, precision, tokenizer flags). Parsed by `parse_base_config()`. Each plugin parses its own strategy-specific fields directly from the raw JSON config text, keeping plugin-specific knowledge out of shared code.

### 4.5 Bundle Section Lookup

- **Header**: `src/bundle/bundle_view.h`
- **Implementation**: `src/bundle/bundle_view.cpp`

Provides `find_section(bundle, name)` for exact-name lookup and `find_sections_by_prefix(bundle, prefix)` for multi-section queries. Plugins use these lightweight helpers to extract their own engine plans, tokenizer data, and config sections from the `BundleFile`.

---

## 5. Bundle Format

- **Header**: `src/bundle/bundle_format.h`
- **Implementation**: `src/bundle/bundle_format.cpp`
- **Writer**: `tensorrt_model_connect/tensorrt_model_connect/bundle_writer.py`

### 5.1 Binary Layout

```
Bytes 0-7:    Magic "TRTFB\x00\x01\x00"
Bytes 8-15:   uint64_t json_header_length (little-endian)
Bytes 16..N:  JSON metadata header (UTF-8)
Bytes N..EOF: Binary sections referenced by offset in the header
```

### 5.2 Sections

The JSON header declares named sections with byte offsets and sizes. Common sections include:

| Section | Present In | Contents |
|---------|-----------|----------|
| `engine_plan` | All | Primary TRT engine plan bytes |
| `config.json` | All | Model config (runtime_strategy, dimensions, tokenizer settings) |
| `tokenizer_dir` | Most | HF tokenizer files (tokenizer.json, vocab, merges) |
| `vision_engine_plan` | VL, SAM | Vision encoder TRT engine plan |
| `denoiser_plan` | Diffusion | DiT/UNet TRT engine plan |
| `vae_decoder_plan` | Diffusion | VAE decoder TRT engine plan |
| `text_encoder_N` | Diffusion | Text encoder TRT engine plan(s) |
| `preprocessor_weights` | Diffusion | Preprocessing weight tensors (timestep embedder, patchify, etc.) |
| `preprocessor_config` | VL | Image preprocessing configuration |
| `semantic_engine_plan` | Bark | Semantic model engine |
| `coarse_engine_plan` | Bark | Coarse acoustic model engine |
| `fine_engine_plan` | Bark | Fine acoustic model engine |
| `codec_engine_plan` | Bark | EnCodec decoder engine |
| `talker_engine_plan` | Omni | Talker decoder engine |
| `code2wav_engine_plan` | Omni | Code-to-waveform engine |

### 5.3 Self-Describing Config

The bundle's `config.json` section carries all build-time decisions. The C++ runtime reads `runtime_strategy` to select the pipeline type, `max_cache_length` for cache sizing, `tokenizer_add_special_tokens` for tokenizer behavior, and modality-specific fields (vision dimensions, audio parameters, diffusion scheduler config). No external configuration files are needed at runtime.

---

## 6. Runtime Strategy Dispatch

### 6.1 Complete Dispatch Flow

```
trtmc::load(bundle_path)
  -> PipelineFactory::from_bundle()           [src/runtime/registry/pipeline_factory.cpp]
    -> ReadBundleFile()                       [src/bundle/bundle_format.cpp]
    -> extract config.json section from bundle
    -> extract_json_string("runtime_strategy") [src/utils/json_helpers.cpp]
    -> normalize_legacy_strategy()            [src/runtime/registry/pipeline_factory.cpp]
    -> PipelineRegistry::instance().lookup()  [src/runtime/registry/pipeline_registry.cpp]
    -> parse_base_config()                    [src/runtime/registry/pipeline_plugin.cpp]
    -> plugin->create(PipelineContext)        [src/runtime/models/<owner>/plugin.cpp]
      -> find_section(bundle, "engine_plan")  [src/bundle/bundle_view.cpp]
      -> load_trt_module_from_plan()          -- deserialize engine, create TrtModule
      -> create_tokenizer_from_bundle()       -- try BPE/WordPiece/Unigram from bundle
      -> create KvCache / RecurrentState      -- if applicable
      -> construct concrete Pipeline class
  -> return unique_ptr<IPipeline>
```

### 6.2 Runtime Strategy to Pipeline Class Mapping

All 25 registered strategy strings and their pipeline mappings:

| `runtime_strategy` | Model Plugin | Pipeline Class | State Management |
|--------------------|-------------|----------------|------------------|
| `decoder_kv_cache` | `text_generation/plugin.cpp` | `TextGenerationPipeline` | `KvCache` |
| `decoder_moe` | `text_generation/plugin.cpp` | `TextGenerationPipeline` | `KvCache` |
| `ssm_recurrent` | `recurrent/ssm_plugin.cpp` | `RecurrentPipeline` | `RecurrentState` (conv + ssm) |
| `rwkv_recurrent` | `recurrent/rwkv_plugin.cpp` | `RecurrentPipeline` | `RecurrentState` (5 state vectors) |
| `hybrid_mamba_attention` | `recurrent/hybrid_plugin.cpp` | `RecurrentPipeline` | `KvCache` + `RecurrentState` (hybrid) |
| `encoder_only` | `encoder/plugin.cpp` | `EncoderPipeline` | None (single pass) |
| `embedding` | `encoder/plugin.cpp` | `EncoderPipeline` | None (single pass) |
| `reranking` | `encoder/plugin.cpp` | `EncoderPipeline` | None (single pass) |
| `neural_operator` | `encoder/plugin.cpp` | `EncoderPipeline` | None (single pass) |
| `vision_language` | `vision_language/plugin.cpp` | `VLPipeline` | `KvCache` + vision `TrtModule` |
| `segmentation` | `segmentation/plugin.cpp` | `SegmentPipeline` | None (single pass) |
| `prompted_segmentation` | `segmentation/plugin.cpp` | `SamPipeline` | None (two-pass: encoder + decoder) |
| `object_detection` | `encoder/object_detection_plugin.cpp` | `EncoderPipeline` | None (single pass) |
| `diffusion_flux` | `flux/plugin.cpp` | `FluxPipeline` | None (iterative denoising) |
| `diffusion_wan` | `wan/plugin.cpp` | `WanPipeline` | None (iterative denoising) |
| `diffusion_pixart` | `wan/plugin.cpp` | `WanPipeline` | None (iterative denoising) |
| `diffusion_zimage` | `z_image/plugin.cpp` | `ZImagePipeline` | None (iterative denoising) |
| `text_to_text` | `t5/plugin.cpp` | (inline T5Pipeline) | `KvCache` + encoder `TrtModule` |
| `marian_translation` | `marian/plugin.cpp` | (inline MarianPipeline) | `KvCache` + encoder `TrtModule` |
| `seq2seq_encoder_decoder` | `seq2seq/plugin.cpp` | (inline Seq2SeqPipeline) | `KvCache` + encoder `TrtModule` |
| `speech_to_text` | `whisper/plugin.cpp` | `WhisperPipeline` | Legacy `WhisperBackend` |
| `text_to_audio_bark` | `bark/plugin.cpp` | `BarkPipeline` | Legacy `BarkBackend` |
| `text_to_audio_magpie` | `magpie/plugin.cpp` | `MagpiePipeline` | Legacy `MagpieTTSBackend` |
| `speech_to_speech` | `speech/plugin.cpp` | `SpeechPipeline` | Legacy `SpeechToSpeechBackend` |
| `omni_multimodal` | `omni/plugin.cpp` | `OmniPipeline` | `TrtModule` + `KvCache` (new runtime) |

Note: Legacy bundles may carry ambiguous strategy strings (`"diffusion"`, `"text_to_audio"`). `pipeline_factory.cpp` normalizes these at load time using `normalize_legacy_strategy()`, which inspects the `diffusion_backend_type` and `magpie_tts` config fields to select the correct unambiguous strategy.

---

## 7. Concrete Pipeline Implementations

Pipeline header/implementation pairs live beside their plugins in `src/runtime/models/<owner>/`. All implement `IPipeline`:

### 7.1 Text Generation

| Class | Header | Composition |
|-------|--------|-------------|
| `TextGenerationPipeline` | `text_generation/pipeline.h` | `TrtModule` (decoder) + `KvCache` + `ITokenizer` |
| `RecurrentPipeline` | `recurrent/pipeline.h` | `TrtModule` + `IStateManager` + `ITokenizer` |

`TextGenerationPipeline` serves all standard decoder and MoE models. The model-specific architecture (GQA, RoPE, SwiGLU, etc.) is baked into the TRT engine.

`RecurrentPipeline` uses an `IStateManager` abstraction (defined in `recurrent_pipeline.h`) with two implementations:
- `RecurrentStateManager` -- wraps `RecurrentState` for pure-recurrent models (Mamba, RWKV). Does not produce attention masks.
- `HybridStateManager` -- wraps both `KvCache` + `RecurrentState` for hybrid Mamba-attention models. Produces attention masks via the KvCache.

### 7.2 Vision-Language

| Class | Header | Composition |
|-------|--------|-------------|
| `VLPipeline` | `vl_pipeline.h` | `TrtModule` (text decoder) + `TrtModule` (vision encoder, optional) + `KvCache` + `ImagePreprocessor` + `ITokenizer` |

### 7.3 Encoder / Perception

| Class | Header | Composition |
|-------|--------|-------------|
| `EncoderPipeline` | `encoder_pipeline.h` | `TrtModule` + `ITokenizer`; mode string selects behavior |
| `SegmentPipeline` | `segment_pipeline.h` | `TrtModule` (single-pass segmentation) |
| `SamPipeline` | `sam_pipeline.h` | `TrtModule` (image encoder) + `TrtModule` (mask decoder) |

### 7.4 Diffusion

| Class | Header | Composition |
|-------|--------|-------------|
| `WanPipeline` | `wan_pipeline.h` | `TrtModule` (T5 encoder) + `TrtModule` (denoiser) + `TrtModule` (VAE) |
| `FluxPipeline` | `flux_pipeline.h` | `TrtModule`(s) (T5 + CLIP) + `TrtModule` (denoiser) + `TrtModule` (VAE) |
| `ZImagePipeline` | `z_image_pipeline.h` | `TrtModule` (text encoder) + `TrtModule` (denoiser) + `TrtModule` (VAE) |

### 7.5 Audio

| Class | Header | Composition |
|-------|--------|-------------|
| `WhisperPipeline` | `whisper_pipeline.h` | Legacy `WhisperBackend` + mel filterbank + `ITokenizer` |
| `BarkPipeline` | `bark_pipeline.h` | Legacy `BarkBackend` + `ITokenizer` |
| `MagpiePipeline` | `magpie_pipeline.h` | Legacy `MagpieTTSBackend` + `ITokenizer` |
| `SpeechPipeline` | `speech_pipeline.h` | Legacy `SpeechToSpeechBackend` |
| `OmniPipeline` | `omni_pipeline.h` | `TrtModule` (thinker) + `KvCache` + `TrtModule` (talker) + `KvCache` + `TrtModule` (code2wav) + `ITokenizer` |

Note: Whisper, Bark, Magpie, and Speech pipelines delegate to legacy backend classes in `src/runtime/domains/audio/`. `OmniPipeline` is fully migrated to the `TrtModule` + `KvCache` composition pattern.

---

## 8. Core Abstraction Inventory

### 8.1 TrtModule

- **Header**: `include/trtmc/runtime/trt_module.h`
- **Implementation**: `src/runtime/backend/trt_module_impl.cpp`

The `model.forward()` abstraction for TensorRT engines. Wraps an engine + execution context. Provides:

- `forward(TensorMap)` -- CPU-to-GPU-to-CPU synchronous execution.
- `forward_device(DeviceTensorMap)` -- GPU-only execution, no host transfers.
- `forward_async()` / `sync()` -- asynchronous execution.
- `bind_external()` -- allows `KvCache` to bind cache device pointers directly.
- `device_ptr()` -- direct device buffer access.
- `keep_alive()` -- opaque resource ownership (engine, stream lifetime).

### 8.2 KvCache

- **Header**: `include/trtmc/runtime/kv_cache.h`
- **Implementation**: `src/runtime/core/kv_cache.cpp`

Autoregressive KV cache state manager. HF equivalent: `DynamicCache` / `past_key_values`. Manages:

- Per-layer K/V device tensors of shape `[max_length, kv_dim]`.
- Position tracking and causal attention mask construction.
- `bind_to(TrtModule)` -- binds `cache_k_{i}`, `cache_v_{i}` (inputs) and `present_k_{i}`, `present_v_{i}` (outputs).
- `advance()` -- copies present K/V into cache, advances position.
- `reset()` -- zeros all buffers for a new sequence.

### 8.3 RecurrentState

- **Header**: `include/trtmc/runtime/recurrent_state.h`
- **Implementation**: `src/runtime/core/recurrent_state.cpp`

Generic config-driven SSM/RWKV state manager. Replaces the original separate `MambaStepState` and `RwkvStepState` with a single class parameterized by `TensorSpec` vectors:

- Mamba: `{{"conv_state", {d_inner*conv_kernel}}, {"ssm_state", {state_size*d_inner}}}`
- RWKV: 5 state vectors per layer: `attn_state`, `ff_state`, `num_state`, `den_state`, `max_state`.

### 8.4 IScheduler

- **Header**: `include/trtmc/runtime/scheduler.h`
- **Implementation**: `src/runtime/core/flow_match_euler_scheduler.cpp`

Diffusion noise scheduler interface. HF equivalent: `SchedulerMixin` / `FlowMatchEulerDiscreteScheduler`. Provides:

- `set_timesteps(num_steps)` -- configure the timestep schedule.
- `timesteps()` / `sigmas()` -- access the schedule.
- `step()` -- single scheduler step: update latents in-place.

One concrete implementation: `FlowMatchEulerScheduler` (used by FLUX, Wan, Z-Image). Factory function `create_scheduler(name, shift)` instantiates by name.

### 8.5 ITokenizer

- **Header**: `include/trtmc/tokenizer.h`
- **Implementations**: `src/tokenizer/`

Five concrete implementations (all native C++, no Python subprocess dependency):

| Class | File | Mechanism |
|-------|------|-----------|
| `VocabTokenizer` | `vocab_tokenizer.cpp` | Word-to-ID lookup from vocabulary list |
| `BpeTokenizer` | `bpe_tokenizer.cpp` | Native BPE tokenizer (parses HF `tokenizer.json`) |
| `WordPieceTokenizer` | `wordpiece_tokenizer.cpp` | Native WordPiece tokenizer (BERT-style, parses HF `tokenizer.json`) |
| `UnigramTokenizer` | `unigram_tokenizer.cpp` | Native Unigram tokenizer (parses HF `tokenizer.json`) |
| `IpaTokenizer` | `ipa_tokenizer.cpp` | Phoneme tokenizer for MagpieTTS |

All implementations live in `src/tokenizer/`. The public API is `trtmc/tokenizer.h`, which exports factory functions `CreateVocabTokenizer()`, `CreateBpeTokenizer()`, `CreateWordPieceTokenizer()`, `CreateUnigramTokenizer()`, and `CreateIpaTokenizer()`. Plugin helpers in `plugin_helpers.h` provide `create_tokenizer_from_bundle()` which auto-detects the tokenizer type from the bundle's `tokenizer.json` section, trying BPE, WordPiece, and Unigram in order.

---

## 9. Backend Executor Organization

Backend executor code lives in `src/runtime/core/` and `src/runtime/domains/` organized by modality:

| Directory | Contents |
|-----------|----------|
| `src/runtime/core/` | Shared infrastructure: `TrtModule`, `KvCache`, `RecurrentState`, `DeviceTensor`, TRT common utilities, decode runtime (argmax, mask building), engine lifecycle |
| `src/runtime/domains/audio/` | Whisper, Bark, MagpieTTS, Speech-to-Speech, Omni backends and supporting types |
| `src/runtime/domains/diffusion/` | Diffusion denoising step seam, generation plan, scheduler helpers, preprocessor weights, math utilities |
| `src/runtime/domains/encoder/` | Encoder, embedding, and reranking backends |
| `src/runtime/domains/multimodal/` | VL backend, vision engine, image preprocessor (4 strategies: `qwen_merge_group`, `simple_chw`, `center_crop_chw`, `aspect_preserve_chw`) |
| `src/runtime/domains/perception/` | Segmentation, SAM, detection, neural operator backends |
| `src/runtime/domains/recurrent/` | Mamba, RWKV, hybrid backends and their decode runtimes and step states |

---

## 10. Known Architectural Debt

### 10.1 (Resolved) FastPathModelConfig God Struct

The former `FastPathModelConfig` monolithic struct has been replaced by `BaseConfig` (~14 universal fields) in `include/trtmc/runtime/pipeline_plugin.h`. Each plugin now parses its own strategy-specific fields directly from the raw JSON config, keeping plugin-specific knowledge localized. This item is resolved.

### 10.2 (Resolved) Centralized Pipeline Factory

The former centralized `pipeline_factory.cpp` (~700 LOC) has been replaced by a registry-based dispatch (`src/runtime/registry/pipeline_factory.cpp`, ~124 LOC). Each strategy plugin exposes a registrar function listed in `cmake/trtmc_pipeline_plugins.cmake`; the generated registrar source calls those functions explicitly. Adding a new strategy requires a new `.cpp` file and one manifest entry -- no edits to the factory or any central dispatch logic. This item is resolved.

### 10.3 Legacy Audio Backends

Whisper, Bark, Magpie, and Speech pipelines delegate to legacy backend classes (`WhisperBackend`, `BarkBackend`, `MagpieTTSBackend`, `SpeechToSpeechBackend` in `src/runtime/domains/audio/`) that predate the `TrtModule` + `KvCache` composition pattern. `OmniPipeline` is the only audio pipeline that has been migrated to the new pattern.

**Impact**: The legacy backends duplicate patterns (engine loading, cache management, decode loops) that are now handled generically by `TrtModule` and `KvCache`.

### 10.4 Perception Backends vs Pipeline Classes

The perception backends (`DetectionBackend`, `NeuralOperatorBackend`, `SegmentationBackend`, `SamBackend` in `src/runtime/domains/perception/`) coexist with the pipeline-level classes (`SegmentPipeline` in `src/runtime/models/segmentation/segment_pipeline.h`, `SamPipeline` in `src/runtime/models/segmentation/sam_pipeline.h`). The relationship between backend and pipeline is not always clear. `EncoderPipeline` handles `object_detection` and `neural_operator` strategies but the corresponding dedicated pipeline classes do not exist at the `IPipeline` level.

---

## 11. Planned Evolution

Some items below have been implemented (marked accordingly); the remaining items are still under consideration.

### 11.1 (Implemented) Plugin Registry for C++ Runtime

**STATUS: IMPLEMENTED.** The centralized `pipeline_factory.cpp` dispatch has been replaced by a `PipelineRegistry` singleton populated by generated manifest registrar calls. See Section 4.3 for details.

### 11.2 (Implemented) BaseConfig Decomposition

**STATUS: IMPLEMENTED.** `FastPathModelConfig` has been replaced by a lightweight `BaseConfig` (~14 universal fields) plus per-plugin JSON parsing. Each plugin reads its own strategy-specific config fields from the raw JSON text. See Section 4.4 for details.

### 11.3 Legacy Audio Migration

Migrate remaining audio backends (Whisper, Bark, Magpie, Speech) to the `TrtModule` + `KvCache` composition pattern, following the `OmniPipeline` precedent.

### 11.4 Service-Oriented Runtime

Decompose pipeline implementations into service interfaces (`ITextService`, `IAudioService`, etc.) behind a `PipelineRouter` that translates `IPipeline` calls into service requests. This would separate task orchestration from TRT execution.

---

## Appendix A: File Path Reference

All paths below are relative to the repository root and have been verified to exist.

### Python Builder
| Path | Purpose |
|------|---------|
| `tensorrt_model_connect/tensorrt_model_connect/engine_builder.py` | Build orchestrator |
| `tensorrt_model_connect/tensorrt_model_connect/config.py` | HF config.json parser |
| `tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py` | Safetensors weight loader |
| `tensorrt_model_connect/tensorrt_model_connect/bundle_writer.py` | Bundle file writer |
| `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` | Atomic TRT graph ops |
| `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py` | Composable graph blocks |
| `tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py` | Standard decoder engine builder |
| `tensorrt_model_connect/tensorrt_model_connect/debug_runner.py` | Python TRT inference runners (decoder, Mamba, VL) |
| `tensorrt_model_connect/tensorrt_model_connect/diffusion_runner.py` | Python TRT diffusion runner |
| `tensorrt_model_connect/tensorrt_model_connect/pipeline.py` | Subprocess wrapper around C++ trtmc binary |
| `tensorrt_model_connect/tensorrt_model_connect/schedulers/` | Python diffusion schedulers (flow_match_euler) |
| `tensorrt_model_connect/tensorrt_model_connect/families/__init__.py` | Plugin auto-discovery |
| `tensorrt_model_connect/tensorrt_model_connect/families/base.py` | FamilyPlugin protocol |
| `tensorrt_model_connect/tensorrt_model_connect/build_cli.py` | Python builder CLI entry point |

### C++ Runtime -- Public API
| Path | Purpose |
|------|---------|
| `include/trtmc/pipeline.h` | IPipeline interface, result types, C ABI |
| `include/trtmc/bundle.h` | BundleInfo, InspectBundle |
| `include/trtmc/tokenizer.h` | ITokenizer interface |
| `include/trtmc/runtime/trt_module.h` | TrtModule abstraction |
| `include/trtmc/runtime/kv_cache.h` | KvCache state manager |
| `include/trtmc/runtime/recurrent_state.h` | RecurrentState manager |
| `include/trtmc/runtime/scheduler.h` | IScheduler interface, FlowMatchEulerScheduler |
| `include/trtmc/runtime/tensor.h` | Tensor, TensorMap, TensorInfo types |
| `include/trtmc/runtime/device_tensor.h` | DeviceTensor, DeviceTensorMap types |
| `include/trtmc/runtime/pipeline_factory.h` | PipelineFactory |
| `include/trtmc/runtime/pipeline_registry.h` | PipelineRegistry singleton, manifest registration macro |
| `include/trtmc/runtime/pipeline_plugin.h` | IPipelinePlugin interface, BaseConfig, PipelineContext |
| `include/trtmc/runtime/tokenizer_interface.h` | ITokenizer abstract interface (re-exported by `tokenizer.h`) |
| `include/trtmc/runtime/domains/audio/speech_decode_stop_policy.h` | Speech decode stop policy for audio pipelines |
| `include/trtmc/runtime/domains/audio/subprocess_runner.h` | Subprocess runner utility for tokenizer bridge |
| `include/trtmc/runtime/domains/multimodal/image_transform_helper.h` | Image transformation utilities for VL preprocessing |

### C++ Runtime -- Registry and Plugins
| Path | Purpose |
|------|---------|
| `src/runtime/registry/pipeline_factory.cpp` | Registry-based factory dispatch (~124 LOC) |
| `src/runtime/registry/pipeline_registry.cpp` | PipelineRegistry singleton |
| `src/runtime/registry/pipeline_plugin.cpp` | BaseConfig parsing (parse_base_config) |
| `src/runtime/models/text_generation/plugin.cpp` | decoder_kv_cache, decoder_moe |
| `src/runtime/models/encoder/plugin.cpp` | encoder_only, embedding, reranking, neural_operator |
| `src/runtime/models/recurrent/ssm_plugin.cpp` | ssm_recurrent |
| `src/runtime/models/recurrent/rwkv_plugin.cpp` | rwkv_recurrent |
| `src/runtime/models/recurrent/hybrid_plugin.cpp` | hybrid_mamba_attention |
| `src/runtime/models/vision_language/plugin.cpp` | vision_language |
| `src/runtime/models/segmentation/plugin.cpp` | segmentation, prompted_segmentation |
| `src/runtime/models/encoder/object_detection_plugin.cpp` | object_detection |
| `src/runtime/models/whisper/plugin.cpp` | speech_to_text |
| `src/runtime/models/bark/plugin.cpp` | text_to_audio_bark |
| `src/runtime/models/magpie/plugin.cpp` | text_to_audio_magpie |
| `src/runtime/models/speech/plugin.cpp` | speech_to_speech |
| `src/runtime/models/omni/plugin.cpp` | omni_multimodal |
| `src/runtime/models/t5/plugin.cpp` | text_to_text |
| `src/runtime/models/marian/plugin.cpp` | marian_translation |
| `src/runtime/models/seq2seq/plugin.cpp` | seq2seq_encoder_decoder |
| `src/runtime/models/flux/plugin.cpp` | diffusion_flux |
| `src/runtime/models/wan/plugin.cpp` | diffusion_wan, diffusion_pixart |
| `src/runtime/models/z_image/plugin.cpp` | diffusion_zimage |
| `cmake/trtmc_pipeline_plugins.cmake` | Plugin source/anchor manifest |
| `src/runtime/plugins/shared/plugin_helpers.h` | Shared plugin helpers (TrtModule loading, tokenizer, KV) |

### C++ Runtime -- Implementation
| Path | Purpose |
|------|---------|
| `src/cabi/api/trtmc_c.cpp` | C ABI entry point |
| `src/bundle/bundle_format.h` | Bundle magic, section types |
| `src/bundle/bundle_format.cpp` | Bundle reader |
| `src/bundle/bundle_view.h` | find_section(), find_sections_by_prefix() |
| `src/runtime/backend/trt_module_impl.cpp` | TrtModule implementation |
| `src/runtime/core/kv_cache.cpp` | KvCache implementation |
| `src/runtime/core/recurrent_state.cpp` | RecurrentState implementation |
| `src/runtime/core/flow_match_euler_scheduler.cpp` | FlowMatchEulerScheduler implementation |
| `src/runtime/models/text_generation/pipeline.h` | TextGenerationPipeline |
| `src/runtime/models/recurrent/pipeline.h` | RecurrentPipeline, IStateManager |
| `src/runtime/models/vision_language/pipeline.h` | VLPipeline |
| `src/runtime/models/encoder/pipeline.h` | EncoderPipeline |
| `src/runtime/models/segmentation/segment_pipeline.h` | SegmentPipeline |
| `src/runtime/models/segmentation/sam_pipeline.h` | SamPipeline |
| `src/runtime/models/flux/pipeline.h` | FluxPipeline |
| `src/runtime/models/wan/pipeline.h` | WanPipeline |
| `src/runtime/models/z_image/pipeline.h` | ZImagePipeline |
| `src/runtime/models/whisper/pipeline.h` | WhisperPipeline |
| `src/runtime/models/bark/pipeline.h` | BarkPipeline |
| `src/runtime/models/magpie/pipeline.h` | MagpiePipeline |
| `src/runtime/models/speech/pipeline.h` | SpeechPipeline |
| `src/runtime/models/omni/pipeline.h` | OmniPipeline |
| `src/tokenizer/vocab_tokenizer.cpp` | VocabTokenizer |
| `src/tokenizer/bpe_tokenizer.cpp` | BpeTokenizer (native C++) |
| `src/tokenizer/wordpiece_tokenizer.cpp` | WordPieceTokenizer (native C++) |
| `src/tokenizer/unigram_tokenizer.cpp` | UnigramTokenizer (native C++) |
| `src/tokenizer/ipa_tokenizer.cpp` | IpaTokenizer |
