# Pipeline Deep Dive

| Field | Value |
|-------|-------|
| Document ID | PDD-001 |
| ISO 26262-6 clause | 7.4.5 (Software unit design and implementation) |
| Applicable to | tensorrt-model-connect C++ runtime |
| Revision | 2.0 |
| Date | 2026-03-12 |
| Status | Living document -- reflects code as of this revision date |
| Author | Safety Architecture Team (TensorRT-Model-Connect Team) |
| Reviewer | Independent Review Required (TBD — assign before merge) |
| Review Status | Pending independent review |

---

This page is a detailed walkthrough of the real pipeline creation and request
handling code.  Every class, function, enum, and file path referenced here
exists in the codebase.  There is no `PipelineRouter`, `PipelineServices`,
`BuildContext`, or `StrategyBuilder`.

---

## 1. Entry Point: `trtmc_create_pipeline_ex()`

**File:** `src/cabi/api/trtmc_c.cpp` (108 LOC)

This is the C ABI entry point. Its responsibilities are deliberately narrow:

1. Validate `bundle_path` is non-null and non-empty.
2. Call `trtmc::IsBundle(path)` to verify the file has `.trtfb` magic bytes.
3. Extract `hf_python` from `TrtmcPipelineOptions` if provided.
4. Delegate entirely to `trtmc::PipelineFactory::from_bundle(path, hf_python)`.
5. On success: log timing, return `pipeline.release()` (raw pointer transfer).
6. On exception: store error in `thread_local g_last_error`, return `nullptr`.

The backward-compatible `trtmc_create_pipeline(bundle_path, flags)` is a thin
wrapper that calls `trtmc_create_pipeline_ex` with default options.

Additional C ABI functions:
- `trtmc_last_error()` -- returns the thread-local error string.
- `trtmc_version()` -- returns `TRTMC_VERSION_STRING`.
- `trtmc_has_trt()` -- returns 1 if compiled with TRT, 0 otherwise.

**Note:** `trtmc_create_pipeline_ex` does NOT own any modality-specific logic.
All construction logic is in `pipeline_factory.cpp`.

---

## 2. Pipeline Factory: `PipelineFactory::from_bundle()`

**File:** `src/runtime/registry/pipeline_factory.cpp` (~124 LOC)
**Header:** `include/trtmc/runtime/pipeline_factory.h`

This single static method is the entire pipeline assembly path:

```text
PipelineFactory::from_bundle(bundle_path, hf_python)
  |
  +-> ReadBundleFile(bundle_path)                 [src/bundle/bundle_format.cpp]
  |     Returns BundleFile{info, sections[]}
  |
  +-> Extract config.json section text
  |
  +-> extract_json_string(config_text, "runtime_strategy")
  |     Defaults to "decoder_kv_cache" if empty/absent
  |
  +-> normalize_legacy_strategy(strategy, config_text)
  |     Rewrites legacy ambiguous strings:
  |       "text_to_audio" -> "text_to_audio_bark" or "text_to_audio_magpie"
  |       "diffusion"     -> "diffusion_wan" / "diffusion_flux" / "diffusion_zimage" / "diffusion_pixart"
  |
  +-> PipelineRegistry::instance().lookup(strategy)
  |     Returns IPipelinePlugin* from the registry singleton
  |
  +-> parse_base_config(config_text, max_cache_length)
  |     Returns BaseConfig (~10 universal fields)
  |
  +-> plugin->create(PipelineContext{bundle, base_cfg, config_json, hf_python, bundle_path})
        Each plugin parses its own strategy-specific config from raw JSON,
        extracts bundle sections, loads engines, creates tokenizers/caches,
        and returns the fully constructed pipeline.
        Returns unique_ptr<IPipeline>
```

A free function `trtmc::load()` is also provided as a convenience alias:
```cpp
std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python)
{
    return PipelineFactory::from_bundle(bundle_path, hf_python);
}
```

---

## 3. Plugin Registry: Strategy Dispatch

**Defined in:** `include/trtmc/runtime/pipeline_registry.h`, `src/runtime/registry/pipeline_registry.cpp`

The runtime uses a **registry-based plugin pattern** for strategy dispatch.
There is no enum, no switch/case, and no centralized factory function per
strategy family.

### PipelineRegistry (singleton)

```cpp
class PipelineRegistry {
public:
    static PipelineRegistry& instance();
    void register_plugin(const std::string& strategy, IPipelinePlugin* plugin);
    IPipelinePlugin* lookup(const std::string& strategy) const;
    std::vector<std::string> registered_strategies() const;
};
```

### IPipelinePlugin (interface)

**Defined in:** `include/trtmc/runtime/pipeline_plugin.h`

```cpp
class IPipelinePlugin {
public:
    virtual ~IPipelinePlugin() = default;
    virtual std::unique_ptr<IPipeline> create(const PipelineContext& ctx) = 0;
};
```

Each plugin receives a `PipelineContext` containing the `BundleFile`, `BaseConfig`,
raw JSON text, `hf_python` path, and `bundle_path`. The plugin is responsible for
parsing its own strategy-specific config fields from the raw JSON, extracting
needed sections from the bundle, loading TRT engines, creating tokenizers and
caches, and returning a fully constructed pipeline.

### Manifest Registration

Plugins expose a registrar function via the registry macro:

```cpp
// In src/runtime/models/<family>/plugin.cpp, inside namespace trtmc:
REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_decoder_plugin, DecoderPlugin,
                                         "decoder_kv_cache", "decoder_moe");
```

The `REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST` macro in `pipeline_registry.h`
defines the function expected by the manifest-generated registrar source.

`cmake/trtmc_pipeline_plugins.cmake` generates the source that calls every
listed registrar explicitly.

### 25 Registered Strategies (model-owned plugin files)

| Strategy | Model runtime plugin | Pipeline class |
|----------|------------|----------------|
| `decoder_kv_cache` | `<family>/plugin.cpp` | `TextGenerationPipeline` |
| `decoder_moe` | `<family>/plugin.cpp` | `TextGenerationPipeline` |
| `mamba_ssm_recurrent` | `mamba/plugin.cpp` | `RecurrentPipeline` |
| `rwkv_recurrent` | `rwkv/plugin.cpp` | `RecurrentPipeline` |
| `hybrid_mamba_attention` | `nemotron_h/plugin.cpp`, `qwen3_5/plugin.cpp` | `RecurrentPipeline` |
| `encoder_only` | `encoder/plugin.cpp` | `EncoderPipeline` |
| `embedding` | `encoder/plugin.cpp` | `EncoderPipeline` |
| `reranking` | `encoder/plugin.cpp` | `EncoderPipeline` |
| `neural_operator` | `encoder/plugin.cpp` | `EncoderPipeline` |
| `vision_language` | `vision_language/plugin.cpp` | `VLPipeline` |
| `segmentation` | `segmentation/plugin.cpp` | `SegmentPipeline` |
| `prompted_segmentation` | `segmentation/plugin.cpp` | `SamPipeline` |
| `object_detection` | `encoder/object_detection_plugin.cpp` | `EncoderPipeline` |
| `speech_to_text` | `whisper/plugin.cpp` | `WhisperPipeline` |
| `text_to_audio_bark` | `bark/plugin.cpp` | `BarkPipeline` |
| `text_to_audio_magpie` | `magpie/plugin.cpp` | `MagpiePipeline` |
| `speech_to_speech` | `speech/plugin.cpp` | `SpeechPipeline` |
| `omni_multimodal` | `omni/plugin.cpp` | `OmniPipeline` |
| `t5_text_to_text` | `t5/plugin.cpp` | `T5Pipeline` |
| `marian_translation` | `marian/plugin.cpp` | `MarianPipeline` |
| `bart_seq2seq_encoder_decoder` | `bart/plugin.cpp` | `BartPipeline` |
| `m2m_100_seq2seq_encoder_decoder` | `m2m_100/plugin.cpp` | `M2M100Pipeline` |
| `diffusion_flux` | `flux/plugin.cpp` | `FluxPipeline` |
| `diffusion_wan` | `wan/plugin.cpp` | `WanPipeline` |
| `diffusion_pixart` | `wan/plugin.cpp` | `WanPipeline` |
| `diffusion_zimage` | `z_image/plugin.cpp` | `ZImagePipeline` |

Runtime plugin and pipeline files live together under `src/runtime/models/`.
`T5Pipeline`, `MarianPipeline`, and `Seq2SeqPipeline` remain inline in their
respective model plugin files.

If `runtime_strategy` is empty (old bundles), it defaults to `"decoder_kv_cache"`.
If no plugin is registered for the strategy string, `PipelineFactory::from_bundle()`
throws `std::runtime_error` listing all available strategies.

---

## 4. Per-Plugin Pipeline Construction

Each model runtime folder in `src/runtime/models/` is self-contained. Its
plugin implements `IPipelinePlugin::create()`, parses its own strategy-specific
config from the raw JSON, extracts the bundle sections it needs via
`find_section(ctx.bundle, "section_name")`, loads TRT engines, creates
tokenizers and caches, and returns a fully constructed pipeline.

Each runtime model folder owns the helper code it needs:
- `plugin_helpers.h/cpp` -- `load_trt_module_from_plan()`, `create_tokenizer_from_bundle()`, `compute_kv_dim()`, `cache_dtype_from_precision()`, `find_section()`
- `diffusion_helpers.h/cpp` -- model-local diffusion loading and family config parsing
- `audio_helpers.h/cpp` -- model-local audio bundle loading helpers

### 4.1 Decoder Plugins

**`src/runtime/models/<family>/plugin.cpp`** -- handles that family's decoder strategy keys, for example `<family>_decoder_kv_cache` and `<family>_decoder_moe`:
- `load_trt_module_from_plan()` -- deserialize the engine plan into a `TrtModule`.
- `create_tokenizer_from_bundle()` -- extract tokenizer files, create `HfPythonTokenizer`.
- `KvCache(num_layers, max_cache_length, kv_dim, stream)`.
- Returns that family's local `TextGenerationPipeline(decoder, cache, config, stream, tokenizer, model_id)` implementation.

**`src/runtime/models/mamba/plugin.cpp`** -- handles `mamba_ssm_recurrent`:
- `MambaRecurrentState` with specs: `conv_state` [d_inner * conv_kernel], `ssm_state` [state_size * d_inner].
- `MambaRecurrentState(state)`.
- Returns `RecurrentPipeline(decoder, state_mgr, config, stream, "MambaPipeline", tokenizer, model_id)`.

**`src/runtime/models/rwkv/plugin.cpp`** -- handles `rwkv_recurrent`:
- `RwkvRecurrentState` with 5 specs per layer: `attn_state`, `ff_state`, `num_state`, `den_state`, `max_state` (each [hidden_size]).
- Returns `RecurrentPipeline(..., "RwkvPipeline", ...)`.

**`nemotron_h/plugin.cpp` and `qwen3_5/plugin.cpp`** -- handle hybrid Mamba-attention strategies:
- `KvCache` for attention layers (num_attention_layers).
- family-owned recurrent state for Mamba layers (num_mamba_layers) with `conv_state` and `ssm_state`.
- Family-owned hybrid state (`NemotronHHybridState` or `Qwen35HybridState`) implements `IInferenceState` by delegating to both.
- Returns `RecurrentPipeline(..., "HybridPipeline", ...)`.

**Key files:**
- `src/runtime/models/<family>/pipeline.h` -- `TextGenerationPipeline`
- `src/runtime/models/<recurrent-family>/pipeline.h` -- family-owned `RecurrentPipeline`
- `src/runtime/models/<family>/inference_state.h` -- `IInferenceState` interface
- `src/runtime/models/mamba/recurrent_state.h` -- `MambaRecurrentState`
- `src/runtime/models/rwkv/recurrent_state.h` -- `RwkvRecurrentState`
- `src/runtime/models/nemotron_h/hybrid_state.h` -- `NemotronHHybridState` (KvCache + NemotronHRecurrentState)
- `src/runtime/models/qwen3_5/hybrid_state.h` -- `Qwen35HybridState` (KvCache + Qwen35RecurrentState)
- `src/runtime/models/<family>/kv_cache.h` -- `KvCache`

### 4.2 Vision and Perception Plugins

**`vl_plugin.cpp`** -- handles `vision_language`:
1. Load text decoder TrtModule from `engine_plan`.
2. Load optional vision encoder TrtModule from `vision_engine_plan` (soft failure).
3. `KvCache` for text decoder.
4. Parse VL preprocessing config from bundle.
5. Returns `VLPipeline(text_decoder, vision_encoder, cache, vlc, vl_preprocess, stream, tokenizer, model_id)`.

**`segmentation_plugin.cpp`** -- handles `segmentation` and `prompted_segmentation`:
- `segmentation` -> `SegmentPipeline(module, model_id)`.
- `prompted_segmentation` -> `SamPipeline(image_encoder, mask_decoder, model_id)`.

**`object_detection_plugin.cpp`** -- handles `object_detection`:
- Returns `EncoderPipeline` configured for detection mode.

**Key files:**
- `src/runtime/models/<vl-family>/pipeline.h` -- family-owned `VLPipeline`, `VLConfig`
- `src/runtime/models/segformer/segment_pipeline.h` -- `SegmentPipeline`
- `src/runtime/models/sam/sam_pipeline.h` -- `SamPipeline`
- `src/runtime/models/sam3/sam3_pipeline.h` -- `Sam3Pipeline`
- `src/runtime/models/<vl-family>/image_preprocessor.h` -- family-owned `VLPreprocessConfig`, preprocessing strategies

### 4.3 Diffusion Plugins

**`flux_plugin.cpp`** -- handles `diffusion_flux`:
- Loads denoiser, VAE, and text encoder TrtModules.
- Also extracts a CLIP tokenizer from bundle (dual tokenizer).
- Returns `FluxPipeline(text_encoders[], denoiser, vae, config, weights, tokenizer, clip_tokenizer, model_id)`.

**`wan_plugin.cpp`** -- handles `diffusion_wan` and `diffusion_pixart`:
- Returns `WanPipeline(text_encoder, denoiser, vae, config, weights, tokenizer, model_id)`.

**`zimage_plugin.cpp`** -- handles `diffusion_zimage`:
- Parses `ZImagePreprocessorWeights`.
- Returns `ZImagePipeline(text_encoder, denoiser, vae, config, weights, z_weights, tokenizer, model_id, hf_python, bundle_path)`.

**Key files:**
- `src/runtime/models/flux/pipeline.h`, `wan_pipeline.h`, `z_image_pipeline.h` -- `FluxPipeline`, `WanPipeline`, `ZImagePipeline`
- `src/runtime/models/wan/pipeline.cpp`, `flux_pipeline.cpp`, `z_image_pipeline.cpp` -- implementations
- `src/runtime/models/<diffusion-model>/<family>_diffusion_types.h` -- family-owned config, preprocessor weights, and result types
- `src/runtime/models/<diffusion-model>/diffusion_helpers.h` -- model-local diffusion loading

### 4.4 Audio Plugins

**`whisper_plugin.cpp`** -- handles `speech_to_text`:
- Creates `WhisperPipeline` from bundle sections.

**`bark_plugin.cpp`** -- handles `text_to_audio_bark`:
- Creates `BarkPipeline` with multi-engine (semantic, coarse, fine, codec).

**`magpie_plugin.cpp`** -- handles `text_to_audio_magpie`:
- Creates `MagpiePipeline` from bundle sections.

**`speech_plugin.cpp`** -- handles `speech_to_speech`:
- Creates `SpeechPipeline` from bundle sections.

**`omni_plugin.cpp`** -- handles `omni_multimodal`:
- Loads thinker TrtModule + KvCache, optional talker TrtModule + KvCache, optional code2wav TrtModule.
- Returns `OmniPipeline(thinker, thinker_cache, talker, talker_cache, code2wav, config, stream, tokenizer, model_id)`.

**Key files:**
- `src/runtime/models/whisper/pipeline.h`, `src/runtime/models/bark/pipeline.h`, `src/runtime/models/magpie/pipeline.h`, `src/runtime/models/personaplex/pipeline.h`, `src/runtime/models/qwen3_omni/pipeline.h`
- `src/runtime/models/<audio-model>/audio_helpers.h` -- model-local audio loading helpers
- `src/runtime/models/<audio-family>/` -- model-owned audio config types and seam headers

### 4.5 Encoder and Seq2Seq Plugins

**`encoder_plugin.cpp`** -- handles `encoder_only`, `embedding`, `reranking`, `neural_operator`:
- Returns `EncoderPipeline(module, strategy, tokenizer, model_id)`.

**`t5/plugin.cpp`** -- handles `t5_text_to_text`:
- Defines `T5Pipeline` inline (encoder-decoder seq2seq).

**`marian_plugin.cpp`** -- handles `marian_translation`:
- Defines `MarianPipeline` inline (encoder-decoder machine translation).

**`bart/plugin.cpp`** -- handles `bart_seq2seq_encoder_decoder`:
- Defines `BartPipeline` inline (BART encoder-decoder).

**`m2m_100/plugin.cpp`** -- handles `m2m_100_seq2seq_encoder_decoder`:
- Defines `M2M100Pipeline` inline (M2M-100/NLLB encoder-decoder).

**Key files:**
- `src/runtime/models/<encoder-family>/pipeline.h` -- `EncoderPipeline`
- `src/runtime/models/<encoder-family>/pipeline.cpp` -- implementations

---

## 5. TrtModule: The Forward Pass Abstraction

**Header:** `include/trtmc/runtime/trt_module.h`
**Implementation:** `src/runtime/backend/trt_module_impl.cpp`

`TrtModule` wraps a TRT `ICudaEngine` + `IExecutionContext`. It pre-allocates
all device buffers at construction and provides three forward pass modes:

### 5.1 `forward(const TensorMap& inputs) -> TensorMap`
Synchronous CPU-to-CPU path:
1. **H2D upload:** For each input in the TensorMap, `cudaMemcpyAsync(HostToDevice)` into pre-allocated device buffers.
2. **Execute:** `ctx_->enqueueV3(stream_)`.
3. **Sync:** `cudaStreamSynchronize(stream_)`.
4. **D2H download:** For each output buffer, `cudaMemcpy(DeviceToHost)` into pre-allocated host staging buffers.
5. Returns a TensorMap where each `Tensor.data` points to the host staging buffer.

This is the primary path used by `TextGenerationPipeline::run_step()`.

### 5.2 `forward_device(const DeviceTensorMap& inputs) -> DeviceTensorMap`
GPU-to-GPU path (no host copies):
1. **D2D copy:** For each input DeviceTensor, `cudaMemcpyAsync(DeviceToDevice)` if the pointer differs from the internal buffer.
2. **Execute + sync.**
3. Returns references to internal output buffers (no copy).

### 5.3 `forward_async(inputs)` + `sync()`
Split async path:
1. `forward_async()` -- H2D upload + enqueueV3 (no sync).
2. `sync()` -- `cudaStreamSynchronize()`.
3. Caller then reads outputs via forward or device_ptr().

### 5.4 `bind_external(name, device_ptr)`
Replaces the pre-allocated buffer for a tensor with an external pointer:
1. Frees the old buffer (if owned).
2. Sets `is_external = true` (so destructor does not free it).
3. Calls `ctx_->setTensorAddress(name, external_device_ptr)`.

This is how `KvCache::bind_to()` and each family-owned recurrent state's
`bind_to()` inject their state buffers into the TrtModule's execution context.

### 5.5 `keep_alive(shared_ptr<void>)`
Stores opaque resource ownership (engine, stream) so they outlive the module's
execution context. Called by `pipeline_factory.cpp` after engine deserialization.

---

## 6. KvCache Lifecycle

**Header:** `src/runtime/models/<family>/kv_cache.h`
**Implementation:** `src/runtime/models/<family>/kv_cache.cpp`

### 6.1 Construction

```cpp
KvCache(num_layers, max_length, kv_dim, stream)
```
Allocates per-layer device buffers:
- `cache_k_[i]`: DeviceTensor shape `[max_length, kv_dim]` (persistent K cache).
- `cache_v_[i]`: DeviceTensor shape `[max_length, kv_dim]` (persistent V cache).
- `present_k_[i]`: DeviceTensor shape `[1, kv_dim]` (single-step K output).
- `present_v_[i]`: DeviceTensor shape `[1, kv_dim]` (single-step V output).

Calls `reset()` to zero all buffers and set position to 0.

### 6.2 `bind_to(TrtModule& module)`

For each layer `i`:
```cpp
module.bind_external("cache_k_" + i, cache_k_[i].data());
module.bind_external("cache_v_" + i, cache_v_[i].data());
module.bind_external("present_k_" + i, present_k_[i].data());
module.bind_external("present_v_" + i, present_v_[i].data());
```
After this call, the TRT engine reads from `cache_k/v` and writes to `present_k/v`
as part of its execution.

### 6.3 `build_attention_mask(vector<float>& mask)`

Builds a causal mask of size `max_length + 1`:
- Positions `[0, position)` = `0.0f` (visible/attended).
- Positions `[position, max_length)` = `-1e4f` (masked/future).
- Position `max_length` = `0.0f` (current token slot -- always visible).

### 6.4 `advance()`

After each decoder step:
1. D2D async copy: `present_k_[i] -> cache_k_[i][position_, :]` for all layers.
2. D2D async copy: `present_v_[i] -> cache_v_[i][position_, :]` for all layers.
3. `position_++`.
4. If `position_ >= max_length_`, clamp to `max_length_ - 1` (sliding window behavior).

### 6.5 `reset()`

Zeros all `cache_k_`, `cache_v_`, `present_k_`, `present_v_` buffers via `cudaMemsetAsync`.
Sets `position_ = 0`. Synchronizes the stream.

---

## 7. Family-Owned Recurrent State Lifecycle

**Headers:** `src/runtime/models/<recurrent-family>/recurrent_state.h`
**Implementations:** `src/runtime/models/<recurrent-family>/recurrent_state.cpp`

Each recurrent family owns its recurrent tensor state manager. The old shared
runtime recurrent-state header/source files are retired.

### 7.1 Construction

```cpp
<Family>RecurrentState(num_layers, specs, stream)
```
Where `specs` is a vector of `TensorSpec{name, shape, output_prefix}`. For each spec
and each layer, allocates two DeviceTensors:
- `state_[spec_idx][layer_idx]` -- persistent state (input to the engine).
- `present_[spec_idx][layer_idx]` -- single-step output from the engine.

### 7.2 `bind_to(TrtModule& module)`

For each spec and layer `i`:
```cpp
module.bind_external("{spec.name}_{i}", state_[spec_idx][i].data());
module.bind_external("{spec.output_prefix}_{i}", present_[spec_idx][i].data());
```

### 7.3 `advance()`

D2D async copy: `present_[spec][layer] -> state_[spec][layer]` for all specs and layers.

### 7.4 `reset()`

Zeros all state and present buffers.

---

## 8. IInferenceState: Unifying KvCache and Family-Owned Recurrent State

**Defined in:** `src/runtime/models/<family>/inference_state.h`

```cpp
class IInferenceState {
public:
    virtual ~IInferenceState() = default;
    virtual void reset() = 0;
    virtual void bind_to(TrtModule& module) = 0;
    virtual void prepare_step(TensorMap& inputs) = 0;
    virtual void advance() = 0;
    virtual int32_t position() const = 0;
    virtual bool ok() const = 0;
    virtual bool has_mask() const = 0;
};
```

Three concrete implementations:

| Class | Header | Used by | Mask? |
|-------|--------|---------|-------|
| `KvCache` | `src/runtime/models/<family>/kv_cache.h` | Standard decoders, VL | Yes |
| `MambaRecurrentState` | `src/runtime/models/mamba/recurrent_state.h` | Mamba | No |
| `RwkvRecurrentState` | `src/runtime/models/rwkv/recurrent_state.h` | RWKV | No |
| `NemotronHHybridState` | `src/runtime/models/nemotron_h/hybrid_state.h` | Nemotron-H | Yes (delegates to KvCache) |
| `Qwen35HybridState` | `src/runtime/models/qwen3_5/hybrid_state.h` | Qwen3.5 | Yes (delegates to KvCache) |

All three implement `IInferenceState`. Pipelines program against the
interface -- `TextGenerationPipeline` and `RecurrentPipeline` both accept
`unique_ptr<IInferenceState>`.

---

## 9. IPipeline: The Public API

**Header:** `include/trtmc/pipeline.h`

`IPipeline` is a pure virtual interface with default implementations that throw
`std::runtime_error` for unsupported operations. Each pipeline type overrides
only the methods it supports:

| Pipeline class | Overrides |
|---------------|-----------|
| `TextGenerationPipeline` | `generate(prompt, cfg)` |
| `RecurrentPipeline` | `generate(prompt, cfg)` |
| `VLPipeline` | `generate(prompt, cfg)`, `generate(prompt, image, h, w, cfg)` |
| `FluxPipeline` | `generate_image(prompt, cfg)` |
| `WanPipeline` | `generate_image(prompt, cfg)` |
| `ZImagePipeline` | `generate_image(prompt, cfg)` |
| `WhisperPipeline` | `transcribe(audio, n, max_tokens, sample_rate)` |
| `BarkPipeline` | `generate_audio(prompt, cfg)` |
| `MagpiePipeline` | `generate_audio(prompt, cfg)` |
| `SpeechPipeline` | `speak(audio_in, n, cfg, sample_rate)` |
| `OmniPipeline` | `generate_audio(prompt, cfg)` |
| `EncoderPipeline` | `embed(text)`, `encode(text)`, `rerank(query, doc)` |
| `SegmentPipeline` | `segment(pixels, h, w)` |
| `SamPipeline` | `segment(pixels, h, w)` |

All pipelines also implement `model_id()` and `pipeline_type()` (pure virtual).

---

## 10. Bundle Sections: Accessed via `find_section()`

In the plugin-based architecture, each plugin accesses bundle sections directly
via the `find_section(bundle, "section_name")` helper from
the owning model's `plugin_helpers.h`. This returns a `const BundleSection*`
(non-owning pointer into the `BundleFile::sections` vector).

Common section names used by plugins:

| Section name pattern | Used by |
|---------------------|---------|
| `engine_plan` | Most plugins (main TRT engine) |
| `vision_engine_plan` | `vl_plugin.cpp` (vision encoder) |
| `config.json` | Parsed by `pipeline_factory.cpp` for `BaseConfig` |
| `preprocessor_config.json` | `vl_plugin.cpp` (image preprocessing) |
| `tokenizer.json`, `tokenizer_config.json` | All plugins with tokenizers |
| `denoiser_plan` | Diffusion plugins |
| `vae_decoder_plan` | Diffusion plugins |
| `text_encoder_N_plan` | Diffusion plugins |
| `coarse_engine_plan`, `fine_engine_plan`, `codec_engine_plan` | `bark_plugin.cpp` |
| `talker_engine_plan`, `code2wav_engine_plan` | `omni_plugin.cpp` |
| `mel_filterbank` | `whisper_plugin.cpp` |

The `BundleFile` must outlive any use of section pointers.

---

## 11. BaseConfig: Universal Bundle Metadata

**Header:** `include/trtmc/runtime/pipeline_plugin.h`
**Implementation:** `src/runtime/registry/pipeline_plugin.cpp`

`BaseConfig` holds the ~10 universal fields that every pipeline needs:

```cpp
struct BaseConfig {
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{1};
    int32_t num_heads{1};
    int32_t num_kv_heads{1};
    int32_t head_dim{0};
    int32_t attention_size{0};
    int32_t max_cache_length{32};
    int32_t id_bos{-1};
    int32_t id_eos{-1};
    std::string runtime_strategy{"decoder_kv_cache"};
    std::string precision{"fp32"};
    bool tokenizer_add_special_tokens{false};
    bool tokenizer_add_special_tokens_present{false};
};
```

Parsed by `parse_base_config()` from the `config.json` section text. Each plugin
then parses its own strategy-specific fields directly from `ctx.config_json`
(the raw JSON text). This avoids the need for a monolithic config struct --
each plugin reads only the fields it requires.

---

## 12. Known Limitations

1. **No pipeline reuse.** Each `trtmc_create_pipeline_ex()` call deserializes
   engines from scratch. There is no caching of deserialized engines across
   pipeline instances.

2. **Batching is family-specific.** Most KvCache and family-owned recurrent
   states manage one sequence at a time. Canary offline transcription supports
   up to 16 encoder inputs and 32 decoder lanes per engine execution.

3. **Synchronous execution.** All pipeline `generate()` methods are fully
   synchronous and block until completion. There is no async/streaming API.

---

## 13. File Reference

| Component | File |
|-----------|------|
| C ABI entry point | `src/cabi/api/trtmc_c.cpp` |
| Pipeline factory | `src/runtime/registry/pipeline_factory.cpp` |
| Pipeline factory header | `include/trtmc/runtime/pipeline_factory.h` |
| Pipeline registry | `include/trtmc/runtime/pipeline_registry.h`, `src/runtime/registry/pipeline_registry.cpp` |
| Plugin interface + BaseConfig | `include/trtmc/runtime/pipeline_plugin.h`, `src/runtime/registry/pipeline_plugin.cpp` |
| Model-local plugin helpers | `src/runtime/models/<model>/plugin_helpers.h`, `.cpp` |
| Plugin source/anchor manifest | `cmake/trtmc_pipeline_plugins.cmake` |
| IPipeline interface | `include/trtmc/pipeline.h` |
| TextGenerationPipeline | `src/runtime/models/<family>/pipeline.h`, `.cpp` |
| RecurrentPipeline | `src/runtime/models/<recurrent-family>/pipeline.h`, `.cpp` |
| VLPipeline | `src/runtime/models/<vl-family>/pipeline.h`, `.cpp` |
| Diffusion pipelines | `src/runtime/models/flux/pipeline.h`, `src/runtime/models/wan/pipeline.h`, `src/runtime/models/z_image/pipeline.h`, `src/runtime/models/pixart/pipeline.h`, `.cpp` |
| Audio pipelines | `src/runtime/models/whisper/pipeline.h`, `src/runtime/models/bark/pipeline.h`, `src/runtime/models/magpie/pipeline.h`, `src/runtime/models/personaplex/pipeline.h`, `src/runtime/models/qwen3_omni/pipeline.h`, `.cpp` |
| Segment/SAM pipelines | `src/runtime/models/segformer/segment_pipeline.h`, `src/runtime/models/sam/sam_pipeline.h`, `src/runtime/models/sam3/sam3_pipeline.h`, `.cpp` |
| Encoder pipelines | `src/runtime/models/<encoder-family>/pipeline.h`, `.cpp` |
| TrtModule | `include/trtmc/runtime/trt_module.h`, `src/runtime/backend/trt_module_impl.cpp` |
| KvCache | `src/runtime/models/<family>/kv_cache.h`, `src/runtime/models/<family>/kv_cache.cpp` |
| Family recurrent state | `src/runtime/models/mamba/recurrent_state.h`, `src/runtime/models/rwkv/recurrent_state.h`, `src/runtime/models/nemotron_h/recurrent_state.h`, `src/runtime/models/qwen3_5/recurrent_state.h`, `src/runtime/models/qwen3_omni/recurrent_state.h` |
| Bundle format | `src/bundle/bundle_format.h`, `.cpp` |
| Image preprocessor | `src/runtime/models/<vl-family>/image_preprocessor.h`, `.cpp` |
| Diffusion types | `src/runtime/models/<diffusion-model>/<family>_diffusion_types.h` |
| Diffusion helpers | `src/runtime/models/<diffusion-model>/diffusion_helpers.h`, `.cpp` |
| Audio helpers | `src/runtime/models/<audio-model>/audio_helpers.h`, `.cpp` |
| Qwen Image scheduler | `src/runtime/models/qwen_image/qwen_image_scheduler.cpp` |
| Tokenizer interface | `include/trtmc/runtime/tokenizer_interface.h` |
| HF Python tokenizer | `src/tokenizer/bpe_tokenizer.cpp` |
| Vocab tokenizer | `src/tokenizer/vocab_tokenizer.cpp` |
