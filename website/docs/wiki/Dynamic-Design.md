# Dynamic Design

| Field | Value |
|-------|-------|
| Document ID | DD-001 |
| ISO 26262-6 clause | 7 (Dynamic aspects of the software architectural design) |
| Applicable to | tensorrt-model-connect (Python build + C++ runtime) |
| Revision | 2.0 |
| Date | 2026-03-12 |
| Status | Living document -- reflects code as of this revision date |
| Author | Safety Architecture Team (TensorRT-Model-Connect Team) |
| Reviewer | Independent Review Required (TBD — assign before merge) |
| Review Status | Pending independent review |

---

This page documents the actual runtime flows as implemented in source code today.
Every participant, function name, and file path references real code.
There is no `PipelineRouter`, `PipelineServices`, `BuildContext`, or `StrategyBuilder` in the codebase.

---

## 1. Bundle Build Flow (Python)

Entry point: `python/tensorrt_model_connect/engine_builder.py` function `build()`.

```mermaid
sequenceDiagram
    participant User
    participant CLI as build_cli.py _cmd_build()
    participant EB as engine_builder.py build()
    participant Resolve as _resolve_model()
    participant Config as ModelConfig.from_dir()
    participant Families as families/__init__.py find_plugin()
    participant Plugin as FamilyPlugin
    participant Writer as bundle_writer.py write_bundle()

    User->>CLI: ./build/trtmc build <model> -o model.trtfb
    CLI->>EB: build(model_id_or_path, output_path, max_cache_length)
    EB->>Resolve: _resolve_model(model_id_or_path)
    Note over Resolve: Local dir with config.json? Return directly.<br/>HF repo ID? snapshot_download().<br/>.nemo archive? Extract to synthetic HF dir.
    Resolve-->>EB: model_dir (local path)
    EB->>EB: Check for model_index.json (diffusers format)
    alt Standard model (config.json)
        EB->>Config: ModelConfig.from_dir(model_dir)
        Config-->>EB: config (model_type, hidden_size, etc.)
        EB->>Families: find_plugin(config.model_type)
        Families-->>EB: plugin (FamilyPlugin instance)
        EB->>Plugin: plugin.load_weights(model_dir, config)
        Plugin-->>EB: weights (WeightDict)
        EB->>Plugin: plugin.build_engine(config, weights, max_cache_length)
        Plugin-->>EB: engine_plan (bytes)
        opt Vision models (Qwen-VL, InternVL)
            EB->>Plugin: plugin.build_vision_engine(model_dir, config, weights)
            Plugin-->>EB: vision_plan (bytes)
        end
        opt Multi-engine models (Bark, PersonaPlex)
            EB->>Plugin: plugin.build_extra_engines(config, weights, max_cache_length)
            Plugin-->>EB: extra_engines dict (name -> plan bytes)
        end
        EB->>Writer: write_bundle(output_path, info, sections)
    else Diffusion model (model_index.json)
        EB->>EB: _build_diffusion_bundle()
        EB->>Families: find_diffusion_plugin(pipeline_class)
        Families-->>EB: plugin
        EB->>Plugin: plugin.load_weights(model_dir, config)
        EB->>Plugin: plugin.build_components(model_dir, config, weights)
        Plugin-->>EB: components dict (text_encoders, denoiser, vae_decoder, preprocessor_weights)
        EB->>Writer: write_bundle(output_path, info, sections)
    end
    Writer-->>User: model.trtfb on disk
```

**Key files:**
- `python/tensorrt_model_connect/build_cli.py` -- Python builder CLI dispatch
- `python/tensorrt_model_connect/engine_builder.py` -- `build()`, `build_bundle()`, `_build_diffusion_bundle()`
- `python/tensorrt_model_connect/config.py` -- `ModelConfig.from_dir()`
- `python/tensorrt_model_connect/families/__init__.py` -- `find_plugin()`, `find_diffusion_plugin()`
- `python/tensorrt_model_connect/families/base.py` -- `FamilyPlugin` protocol
- `python/tensorrt_model_connect/bundle_writer.py` -- `write_bundle()`

---

## 2. Runtime Pipeline Creation Flow (C++)

Entry point: `trtmc_create_pipeline_ex()` in `src/cabi/api/trtmc_c.cpp`.
Factory logic lives in `PipelineFactory::from_bundle()` in `src/runtime/registry/pipeline_factory.cpp`.
Plugin dispatch uses `PipelineRegistry` in `src/runtime/registry/pipeline_registry.cpp`.

```mermaid
sequenceDiagram
    participant User
    participant CABI as trtmc_c.cpp<br/>trtmc_create_pipeline_ex()
    participant Factory as PipelineFactory::from_bundle()<br/>(pipeline_factory.cpp)
    participant Bundle as bundle_format.cpp<br/>ReadBundleFile()
    participant Config as pipeline_plugin.cpp<br/>parse_base_config()
    participant Registry as PipelineRegistry<br/>::instance().lookup()
    participant Plugin as IPipelinePlugin<br/>::create()
    participant Pipeline as Concrete IPipeline

    User->>CABI: trtmc_create_pipeline_ex(bundle_path, options)
    CABI->>CABI: validate bundle_path \!= null/empty
    CABI->>CABI: IsBundle(path) -- check magic bytes
    CABI->>Factory: PipelineFactory::from_bundle(path, hf_python)
    Factory->>Bundle: ReadBundleFile(bundle_path)
    Bundle-->>Factory: BundleFile (info + sections vector)
    Factory->>Factory: Extract config.json section text
    Factory->>Factory: extract_json_string("runtime_strategy")
    Factory->>Factory: normalize_legacy_strategy(strategy, config_text)
    Note over Factory: Rewrites legacy strings:<br/>"text_to_audio" -> bark/magpie<br/>"diffusion" -> wan/flux/zimage/pixart
    Factory->>Registry: PipelineRegistry::instance().lookup(strategy)
    Registry-->>Factory: IPipelinePlugin* (or nullptr -> error)
    Factory->>Config: parse_base_config(config_text, max_cache_length)
    Config-->>Factory: BaseConfig (~10 universal fields)
    Factory->>Plugin: plugin->create(PipelineContext{bundle, config, json, hf_python, path})
    Note over Plugin: Plugin parses strategy-specific config from raw JSON,<br/>extracts bundle sections, loads TRT engines,<br/>creates tokenizers/caches, returns pipeline
    Plugin->>Pipeline: Construct concrete pipeline
    Pipeline-->>Plugin: unique_ptr<IPipeline>
    Plugin-->>Factory: unique_ptr<IPipeline>
    Factory-->>CABI: unique_ptr<IPipeline>
    CABI-->>User: IPipeline* (released raw pointer)
```

**Key files:**
- `src/cabi/api/trtmc_c.cpp` -- C ABI entry point
- `src/runtime/registry/pipeline_factory.cpp` -- `PipelineFactory::from_bundle()` (~124 LOC)
- `src/runtime/registry/pipeline_registry.cpp` -- `PipelineRegistry` singleton
- `src/runtime/registry/pipeline_plugin.cpp` -- `parse_base_config()`
- `include/trtmc/runtime/pipeline_factory.h` -- `PipelineFactory` class declaration
- `include/trtmc/runtime/pipeline_registry.h` -- `PipelineRegistry`, manifest registration macro
- `include/trtmc/runtime/pipeline_plugin.h` -- `IPipelinePlugin`, `BaseConfig`, `PipelineContext`
- `src/runtime/plugins/` -- 20 manifest-registered plugin files (25 strategies total)
- `src/bundle/bundle_format.cpp` -- `ReadBundleFile()`, `HasBundleMagic()`

---

## 3. Text Generation Request Flow

Entry point: `TextGenerationPipeline::generate()` in `src/runtime/models/<family>/pipeline.cpp`.

```mermaid
sequenceDiagram
    participant User
    participant TGP as TextGenerationPipeline
    participant Tok as ITokenizer
    participant Cache as KvCache
    participant Module as TrtModule
    participant GPU as CUDA / TRT Engine

    User->>TGP: generate(prompt, cfg)
    TGP->>Tok: encode(prompt)
    Tok-->>TGP: input_ids (vector<int32_t>)
    TGP->>Cache: reset()
    Note over Cache: Zero all cache_k/v and present_k/v buffers.<br/>Set position = 0.
    TGP->>Cache: bind_to(*decoder_)
    Note over Cache: For each layer i:<br/>  bind_external("cache_k_i", ...)<br/>  bind_external("cache_v_i", ...)<br/>  bind_external("present_k_i", ...)<br/>  bind_external("present_v_i", ...)
    loop Prefill: for each token in input_ids[0..N-2]
        TGP->>TGP: run_step(token_id, logits)
        TGP->>Cache: build_attention_mask(mask)
        Note over Cache: mask[0..position-1] = 0.0 (visible)<br/>mask[position..max_len-1] = -1e4 (masked)<br/>mask[max_len] = 0.0 (current token slot)
        TGP->>Module: forward({token_id, position_id, attention_mask})
        Module->>GPU: H2D inputs -> enqueueV3 -> D2H outputs
        GPU-->>Module: logits tensor
        Module-->>TGP: TensorMap with "logits"
        TGP->>Cache: advance()
        Note over Cache: D2D copy: present_k/v -> cache_k/v[position]<br/>position++
    end
    TGP->>TGP: run_step(input_ids.back(), logits)
    loop Decode: for step in 0..max_new_tokens
        TGP->>TGP: next_token = argmax(logits)
        TGP->>TGP: output.push_back(next_token)
        alt next_token == eos_token_id
            Note over TGP: Break
        else
            TGP->>TGP: run_step(next_token, logits)
        end
    end
    TGP->>Tok: decode(new_tokens)
    Tok-->>TGP: text string
    TGP-->>User: TextResult{text, token_ids}
```

**Key files:**
- `src/runtime/models/<family>/pipeline.cpp` -- `generate()`, `generate_from_ids()`, `run_step()`
- `src/runtime/models/<family>/pipeline.h` -- class declaration
- `src/runtime/core/kv_cache.cpp` -- `bind_to()`, `advance()`, `build_attention_mask()`, `reset()`
- `src/runtime/backend/trt_module_impl.cpp` -- `forward()`, `forward_async()`, `bind_external()`

---

## 4. Recurrent (Mamba/RWKV/Hybrid) Generation Flow

Same generate loop structure as text generation, but using `IInferenceState` abstraction.

```mermaid
sequenceDiagram
    participant User
    participant RP as RecurrentPipeline
    participant SM as IInferenceState
    participant Module as TrtModule

    User->>RP: generate(prompt, cfg)
    RP->>RP: tokenizer_->encode(prompt)
    RP->>SM: reset()
    RP->>SM: bind_to(*decoder_)
    Note over SM: Family recurrent state: binds conv_state/ssm_state per layer<br/>Family-owned hybrid state: binds KvCache + family recurrent tensors
    loop Prefill + Decode
        RP->>SM: build_mask(mask)
        Note over SM: Family recurrent state: no-op (no mask needed)<br/>Family-owned hybrid state: delegates to KvCache::build_attention_mask()
        RP->>Module: forward({token_id, [mask], [position_id]})
        Module-->>RP: logits
        RP->>SM: advance()
        Note over SM: Family recurrent state: D2D present->state, position++<br/>Family-owned hybrid state: KvCache::advance() + family recurrent advance()
    end
    RP-->>User: TextResult{text, token_ids}
```

**Key files:**
- `src/runtime/models/<recurrent-family>/pipeline.h` -- family-owned `RecurrentPipeline`
- `include/trtmc/runtime/inference_state.h` -- `IInferenceState` interface
- `src/runtime/models/<recurrent-family>/pipeline.cpp` -- `generate()`, `run_step()`
- `src/runtime/models/<recurrent-family>/recurrent_state.h` -- family-owned recurrent state class

---

## 5. Vision-Language Generation Flow

Entry point: `VLPipeline::generate()` in each model-owned
`src/runtime/models/<vl-family>/pipeline.cpp`.

```mermaid
sequenceDiagram
    participant User
    participant VLP as VLPipeline
    participant Tok as ITokenizer
    participant VisionMod as TrtModule (vision)
    participant TextMod as TrtModule (text decoder)
    participant Cache as KvCache

    User->>VLP: generate(prompt, image_pixels, h, w, cfg)
    VLP->>Tok: encode(prompt with image tokens)
    Tok-->>VLP: input_ids (contains image_token_id placeholders)
    opt Vision encoder present
        VLP->>VLP: preprocess image pixels per VLPreprocessConfig
        VLP->>VisionMod: forward({pixel_values})
        VisionMod-->>VLP: image_features [num_patches, feature_dim]
    end
    VLP->>Cache: reset()
    VLP->>Cache: bind_to(*text_decoder_)
    loop VL prefill: for each input token
        alt token == image_token_id AND features available
            VLP->>TextMod: run_text_step_with_embed(token, vision_feature, use_embed=1.0)
        else
            VLP->>TextMod: run_text_step(token, logits)
        end
        VLP->>Cache: advance()
    end
    loop Decode: autoregressive token generation
        VLP->>TextMod: run_text_step(next_token, logits)
        VLP->>VLP: argmax(logits)
        VLP->>Cache: advance()
    end
    VLP->>Tok: decode(new_tokens)
    VLP-->>User: TextResult{text, token_ids}
```

**Key files:**
- `src/runtime/models/<vl-family>/pipeline.h` -- `VLPipeline`, `VLConfig`
- `src/runtime/models/<vl-family>/pipeline.cpp` -- `generate()`, `generate_vl_from_ids()`, `run_vision_encoder()`
- `src/runtime/models/<vl-family>/image_preprocessor.h` -- `VLPreprocessConfig`, preprocessing strategies

---

## 6. Diffusion Generation Flow (WanPipeline)

Entry point: `WanPipeline::generate_image()` in `src/runtime/models/wan/pipeline.cpp`.

```mermaid
sequenceDiagram
    participant User
    participant WP as WanPipeline
    participant Tok as ITokenizer
    participant T5 as TrtModule (T5 text encoder)
    participant Denoiser as TrtModule (DiT denoiser)
    participant VAE as TrtModule (VAE decoder)
    participant Sched as Flow-Match Euler Scheduler

    User->>WP: generate_image(prompt, cfg)
    WP->>Tok: encode(prompt)
    Tok-->>WP: input_ids
    WP->>T5: forward({input_ids})
    T5-->>WP: text_embeddings [seq_len, text_dim]
    WP->>WP: project_text() via family-owned preprocessor weights
    WP->>WP: Initialize random latent noise [z_dim, t_lat, h_lat, w_lat]
    WP->>WP: patchify(latents) -> patches [num_patches, patch_dim]
    WP->>WP: compute_3d_rope(nt, nh, nw) -> cos, sin
    WP->>Sched: compute timestep schedule (num_inference_steps)
    loop Denoising: for each timestep
        WP->>WP: compute_timestep_embedding(timestep) -> temb_6d, time_embed
        WP->>Denoiser: forward({hidden, temb_6d, time_embed, encoder_hidden, cos, sin})
        Denoiser-->>WP: predicted noise
        WP->>Sched: euler step: latents += dt * noise_pred
    end
    WP->>WP: unpatchify(patches) -> latents
    WP->>VAE: forward({latents}) per temporal frame
    VAE-->>WP: decoded pixels
    WP-->>User: ImageResult{pixels, height, width, num_frames}
```

**Key files:**
- `src/runtime/models/flux/pipeline.h`, `wan_pipeline.h`, `z_image_pipeline.h` -- `WanPipeline`, `FluxPipeline`, `ZImagePipeline`
- `src/runtime/models/wan/pipeline.cpp` -- `generate_image()`, denoising loop, VAE decode
- `src/runtime/models/flux/pipeline.cpp` -- FLUX-specific RoPE, timestep embedding, dual tokenizer
- `src/runtime/models/z_image/pipeline.cpp` -- Z-Image patchify, caption projection
- `src/runtime/models/qwen_image/qwen_image_scheduler.cpp` -- Qwen Image-owned scheduler implementation

---

## 7. Audio Generation Flow (Bark)

Entry point: `BarkPipeline::generate_audio()` in individual pipeline `.cpp` files.

```mermaid
sequenceDiagram
    participant User
    participant BP as BarkPipeline
    participant Backend as BarkBackend
    participant Tok as ITokenizer

    User->>BP: generate_audio(prompt, cfg)
    BP->>Tok: encode(prompt)
    Tok-->>BP: input_ids
    BP->>Backend: generate(input_ids, cfg)
    Note over Backend: Stage 1: Semantic model generates semantic tokens<br/>Stage 2: Coarse model generates coarse audio tokens<br/>Stage 3: Fine model generates fine audio tokens<br/>Stage 4: EnCodec decodes tokens to waveform
    Backend-->>BP: AudioResult{samples, num_samples, sample_rate}
    BP-->>User: AudioResult
```

**Key files:**
- `src/runtime/models/bark/pipeline.h/cpp` -- `BarkPipeline`
- `src/runtime/models/whisper/pipeline.h/cpp`, `magpie_pipeline.h/cpp`, `speech_pipeline.h/cpp`, `omni_pipeline.h/cpp`
- `src/runtime/models/bark/plugin.cpp`, `whisper_plugin.cpp`, `magpie_plugin.cpp`, `speech_plugin.cpp`, `omni_plugin.cpp`
- `src/runtime/domains/audio/bark_config.h`, `bark_generation_plan.h`

---

## 8. Omni Multimodal Flow (TrtModule-based)

Entry point: `OmniPipeline::generate_audio()` in individual pipeline `.cpp` files.

```mermaid
sequenceDiagram
    participant User
    participant OP as OmniPipeline
    participant Thinker as TrtModule + KvCache
    participant Talker as TrtModule + KvCache
    participant Code2Wav as TrtModule

    User->>OP: generate_audio(prompt, cfg)
    OP->>OP: tokenizer_->encode(prompt) -> input_ids
    OP->>Thinker: run_thinker(input_ids, max_tokens)
    Note over Thinker: Autoregressive decode with KvCache.<br/>Collects text tokens + hidden states.
    Thinker-->>OP: text_tokens + hidden_states
    OP->>Talker: run_talker(hidden_states, num_tokens)
    Note over Talker: Converts hidden states to RVQ codec tokens<br/>using talker decoder with its own KvCache.
    Talker-->>OP: codec_tokens [n_codebooks x n_frames]
    OP->>Code2Wav: run_code2wav(codec_tokens, n_codebooks, n_frames)
    Note over Code2Wav: Single forward pass: codec tokens -> waveform
    Code2Wav-->>OP: waveform samples
    OP-->>User: AudioResult{samples, num_samples, sample_rate}
```

**Key files:**
- `src/runtime/models/omni/pipeline.h/cpp` -- `OmniPipeline`, `OmniConfig`
- `src/runtime/models/omni/plugin.cpp` -- `OmniPlugin`

---

## 9. Error Propagation

All flows use the same error pattern:

1. `trtmc_create_pipeline_ex()` catches all exceptions via `try/catch(...)`.
2. On failure, `set_last_error(msg)` stores the message in a thread-local string.
3. The function returns `nullptr`.
4. The caller retrieves the error via `trtmc_last_error()`.

Pipeline method errors (e.g., `generate()` on an unsupported pipeline type) throw `std::runtime_error` from the default `IPipeline` virtual method implementations in `include/trtmc/pipeline.h`.

---

## 10. Thread Safety

- `g_last_error` in `trtmc_c.cpp` is `thread_local` -- safe for concurrent pipeline creation on different threads.
- Individual `IPipeline` instances are NOT thread-safe. Each pipeline owns an exclusive CUDA stream and device state (KvCache, family-owned recurrent state, TrtModule execution context). Concurrent `generate()` calls on the same pipeline instance produce undefined behavior.
- Different pipeline instances on different CUDA streams can run concurrently.

---

## 11. Testability

Each stage in these flows is independently testable:

| Stage | Test approach | Test location |
|-------|--------------|---------------|
| `ModelConfig.from_dir()` | Synthetic config.json | `tests/builder/test_config.py` |
| `find_plugin()` + `load_weights()` | Synthetic safetensors | `tests/builder/test_family_plugins.py` |
| `write_bundle()` / `ReadBundleFile()` | Round-trip in memory | `tests/builder/test_bundle_writer.py`, `tests/cpp/test_bundle_format.cpp` |
| `parse_base_config()` | Config JSON strings | `tests/cpp/test_config_schema_registry.cpp` |
| `find_section()` | Synthetic bundles | `tests/cpp/test_bundle_view.cpp` |
| KvCache state machine | cache update and mask behavior | `tests/cpp/test_kv_cache_new.cpp`, `tests/builder/test_cache_state_machine.py` |
| TrtModule forward | Real TRT engine | `tests/cpp/test_cuda_buffer.cpp` (buffer ops), E2E tests |
| Full pipeline | E2E harness | `tests/test_e2e.py` |
