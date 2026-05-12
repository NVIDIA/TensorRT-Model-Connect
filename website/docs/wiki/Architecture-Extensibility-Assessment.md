# Architecture Extensibility Assessment

Status of non-standard architecture support. MoE, Mamba/SSM, vision-language (Qwen-VL), and diffusion (Wan2.1 T2V) are fully implemented. DeepSeek MLA and hybrid SSM+Attention are planned.

## Executive Summary

With the Python build / C++ runtime split, adding a new family is Python-only **when it reuses an existing `runtime_strategy`** already handled by a C++ model runtime folder in `src/runtime/models/`. New strategy/state types require a new model runtime folder plus one manifest entry in `cmake/trtmc_pipeline_plugins.cmake` -- no edits to `pipeline_factory.cpp` are needed.

As of 2026-02-20, MoE, Mamba/SSM, vision-language, and diffusion (T2V) support are **fully implemented**. The standard decoder builder is parameterized to support LayerNorm, GELU, learned positions, and multiple activations. The VL image preprocessor supports 4 strategies with configurable interpolation. The diffusion pipeline supports text-to-video with T5 encoding, DiT denoising, and causal 3D VAE decoding.

| Architecture Class | Current Support | Effort to Add New Instance | Where Changes Needed |
|---|---|---|---|
| Standard decoder (RMSNorm + RoPE + SwiGLU) | **Works today** (15+ families) | ~30 LOC Python | Python plugin only |
| Extended decoder (LayerNorm, GELU, learned positions) | **Works today** (25+ families) | ~60 LOC Python | Python plugin only |
| MoE decoder (top-k softmax / SparseMixer routing) | **Works today** (4 families) | ~300 LOC Python | Python graph builder + checkpoint mapper |
| SSM / Mamba | **Works today** (Mamba 130M-2.8B) | ~400 LOC Python | Python graph builder (C++ plugin exists) |
| RWKV | **Works today** | ~400 LOC Python | Python graph builder (C++ plugin exists) |
| Vision-Language | **Works today** (Qwen-VL, InternVL, Phi4) | ~200 LOC Python | Python vision builder + plugin VL config |
| Diffusion (T2V/T2I) | **Works today** (Wan, FLUX, Z-Image, PixArt) | ~500 LOC Python | Python builders + family plugin |
| Encoder-only | **Works today** (BERT, ELECTRA, etc.) | ~60 LOC Python | Python plugin only |
| Encoder-decoder (seq2seq) | **Works today** (T5, Marian, M2M-100) | ~300 LOC Python + C++ | Python + C++ plugin |
| Multi-Latent Attention -- MLA (DeepSeek-V2/V3) | **Not yet implemented** | ~400 LOC Python + C++ | Python graph builder + C++ KV cache shape |
| Hybrid SSM+Attention | **Works today** (Nemotron-H) | ~500 LOC Python + C++ | Python + C++ hybrid state (implemented) |

---

## What Is Easy (Python-Only When Reusing Existing `runtime_strategy`)

### Adding a standard dense decoder family

Create a Python plugin file in `tensorrt_model_connect/tensorrt_model_connect/families/` with a checkpoint mapper. Uses the parameterized standard decoder builder. ~30-60 LOC.

**Implemented**: Qwen, LLaMA, Mistral, Gemma, Phi, Granite, InternLM (standard decoder); StarCoder2, GPT-2, OPT, Falcon, StableLM, OLMo, XGLM, GPT-NeoX, GPT-Neo, CodeGen, BLOOM, Nemotron (extended decoder).

### Adding a new MoE family

Write a Python graph builder for the expert routing logic. The C++ runtime uses the same KV-cache backend (routing is handled in the TRT graph). ~300 LOC.

**Implemented**: Phi-MoE (SparseMixer routing), Mixtral (standard top-2 softmax routing). Both use `runtime_strategy="decoder_moe"`.

### Adding a new Mamba/SSM family

Write a Python graph builder for the SSM architecture. The C++ `ssm_plugin.cpp` constructs a `RecurrentPipeline` with `RecurrentStateManager` for `runtime_strategy="ssm_recurrent"`. ~400 LOC Python.

**Implemented**: Mamba (130M-2.8B, selective scan + conv1d).

### Adding a new Vision-Language family

Write a Python plugin with `build_vision_engine()` for the vision encoder and `get_vl_config()` to specify preprocessing parameters (preprocessor_type, interpolation, prompt template, etc.). The C++ runtime handles 4 preprocessing strategies out of the box. ~200 LOC Python.

**Implemented**: Qwen-VL (Qwen2.5-VL, ViT + 3D RoPE + spatial merge, `runtime_strategy="vision_language"`).

### Adding a new Diffusion family

Write a Python family plugin that composes the shared builders (`t5_encoder_builder`, `standard_dit_builder`, `causal_vae_3d_builder`). The C++ `DiffusionBackendBase` provides shared infrastructure; family-specific backends extend it. ~500 LOC Python.

**Implemented**: Wan2.1-T2V (1.3B-14B, flow-match Euler scheduler, `runtime_strategy="diffusion"`).

---

## What Requires C++ Changes

### Different state management (done for Mamba/SSM/RWKV/Hybrid)

The C++ runtime supports multiple state management patterns via the plugin registry. Each model runtime folder in `src/runtime/models/` exposes a manifest-listed registrar for one or more `runtime_strategy` strings. 25 strategies are currently registered across model-owned plugin files:
- `decoder_kv_cache` / `decoder_moe` -> `text_generation/plugin.cpp` -> `TextGenerationPipeline` + `KvCache`
- `ssm_recurrent` -> `recurrent/ssm_plugin.cpp` -> `RecurrentPipeline` + `RecurrentStateManager`
- `rwkv_recurrent` -> `recurrent/rwkv_plugin.cpp` -> `RecurrentPipeline` + `RecurrentStateManager`
- `hybrid_mamba_attention` -> `recurrent/hybrid_plugin.cpp` -> `RecurrentPipeline` + `HybridStateManager`
- `vision_language` -> `vision_language/plugin.cpp` -> `VLPipeline`
- `diffusion_flux`/`diffusion_wan`/`diffusion_zimage`/`diffusion_pixart` -> separate model runtime folders

New state types require:
1. A new model runtime folder under `src/runtime/models/` with a plugin `.cpp` implementing `IPipelinePlugin`
2. Manifest registration via `REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST`
3. A source/symbol entry in `cmake/trtmc_pipeline_plugins.cmake`
4. A new or existing pipeline class in that model runtime folder

No edits to `pipeline_factory.cpp` are needed -- the registry handles dispatch automatically.

### Different KV cache shapes (DeepSeek MLA)

Compressed KV caches (e.g., `[cache_len, kv_lora_rank]` instead of `[cache_len, attention_size]`) need changes to `DeviceKvCache` in C++ to support variable cache sizes per layer.

---

## Architecture-Specific Deep Dives

### MoE (Phi-MoE -- IMPLEMENTED)

**Status**: Fully implemented. Phi-MoE plugin with SparseMixer routing.

**Python** (`families/phi_moe.py`):
- Checkpoint mapper: Maps router weights + per-expert gate/up/down projections
- Custom graph builder: SparseMixer routing (independent masked softmax, not standard top-k), per-expert SwiGLU MLPs with gather/scatter dispatch
- LayerNorm with bias, separate Q/K/V/O with biases

**C++**: No changes. Uses `runtime_strategy="decoder_moe"` which is handled by the same `decoder_plugin.cpp` as `decoder_kv_cache`, constructing a `TextGenerationPipeline` (routing is handled entirely in the TRT graph).

### Mamba / SSM (IMPLEMENTED)

**Status**: Fully implemented. Mamba plugin with selective scan + C++ MambaBackend.

**Python** (`families/mamba.py`):
- Checkpoint mapper: Maps SSM weights (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj)
- Custom graph builder: Selective scan, causal conv1d with cached state, input-dependent discretization
- Engine I/O: token_id + per-layer conv_state/ssm_state inputs, logits + present_conv/present_ssm outputs

**C++** (plugin-based):
- `ssm_plugin.cpp`: Self-registering plugin for `ssm_recurrent` strategy
- Constructs `RecurrentPipeline` with `RecurrentStateManager` wrapping `RecurrentState` (conv_state + ssm_state specs)
- State management via `RecurrentState` (`include/trtmc/runtime/recurrent_state.h`)

**Debug runner**: `MambaTrtRunner` in `debug_runner.py` for pure-Python Mamba TRT inference.

### Vision-Language -- VL (Qwen-VL -- IMPLEMENTED)

**Status**: Fully implemented. Qwen-VL plugin with vision encoder + text decoder.

**Python** (`families/qwen_vl.py`):
- Checkpoint mapper: Standard Qwen weights for text decoder + vision-specific weights (visual.* prefix)
- Vision engine builder: ViT with 3D RoPE + spatial merge (via `qwen_vl_vision_builder.py`)
- Text decoder: Standard Qwen2.5 with `embed_input=True` for VL prefill
- `get_vl_config()` returns preprocessor_type, interpolation, prompt template, token config

**C++** (`image_preprocessor.h/cpp`):
- 4 image preprocessing strategies: `qwen_merge_group`, `simple_chw`, `center_crop_chw`, `aspect_preserve_chw`
- Configurable interpolation: `bicubic` (default), `bilinear`, `nearest`
- Config parsed from bundle's `config.json` + `preprocessor_config.json`
- `format_vl_prompt()` for prompt template expansion

**Python debug runner** (`debug_runner.py`):
- `VisionTrtRunner`, `VLTrtRunner` for pure-Python VL inference
- `preprocess_image_for_trt()` dispatches to 4 strategies matching C++
- `_resolve_pil_interpolation()` maps mode strings to PIL constants
- Single-image constraint enforced via `NotImplementedError` for multi-image input

**Adding a new VL family**: Create a plugin with `build_vision_engine()` and `get_vl_config()` methods. The `preprocessor_type` and `interpolation` fields in `get_vl_config()` control C++ image preprocessing. See `families/qwen_vl.py` for an example.

### Diffusion -- T2V (Wan2.1 -- IMPLEMENTED)

**Status**: Fully implemented. Wan2.1 text-to-video plugin with T5 encoder, DiT denoiser, and causal 3D VAE decoder.

**Python** (`families/wan_t2v.py`):
- Family plugin that composes three shared builders:
  - `t5_encoder_builder.py`: UMT5 encoder (4096D, 24 layers, 226 token sequence)
  - `standard_dit_builder.py`: DiT denoiser (1536D, 12 heads, 30 layers for 1.3B) with AdaLN modulation, QK norm, 3D RoPE
  - `causal_vae_3d_builder.py`: Causal 3D VAE decoder (16 latent channels, per-frame with temporal caches)
- `_serialize_preprocessor_weights()`: Packs DiT preprocessor weights (patch embedding, timestep MLP, text projection) into binary with JSON index
- `get_diffusion_config()`: Returns pipeline configuration with preset latent statistics, scheduler type, guidance scale

**Python debug runner** (`diffusion_runner.py`):
- `DiffusionRunner`: Pure-Python TRT diffusion pipeline
  - `encode_text()`: T5 encoder with attention masking, zeros padding positions
  - `denoise()`: Flow-match Euler denoising loop with classifier-free guidance (CFG)
  - `decode_video()`: Frame-by-frame VAE decode with causal convolution caches
  - `generate()`: Full pipeline: text encode -> denoise -> VAE decode

**C++** (plugin-based):
- `wan_plugin.cpp`: Self-registering plugin for `diffusion_wan` and `diffusion_pixart` strategies. Constructs `WanPipeline`.
- `flux_plugin.cpp`: Plugin for `diffusion_flux`. Constructs `FluxPipeline`.
- `zimage_plugin.cpp`: Plugin for `diffusion_zimage`. Constructs `ZImagePipeline`.
- Shared diffusion helpers in `src/runtime/plugins/shared/diffusion_helpers.h/cpp`.
- Pipeline implementations in `src/runtime/models/wan/pipeline.cpp`, `flux_pipeline.cpp`, `z_image_pipeline.cpp`.
- `DiffusionConfig`, `PreprocessorWeights` in `src/runtime/domains/diffusion/diffusion_types.h`.
- `FlowMatchEulerScheduler` in `src/runtime/core/flow_match_euler_scheduler.cpp`.

**Schedulers**:
- Flow matching: `z_t = (1-t)*x + t*noise`, Euler step: `z_{t-dt} = z_t - dt*v`
- Configurable shift parameter for timestep adjustment
- C++ implementation in `flow_match_euler_scheduler.cpp`, Python in `tensorrt_model_connect/tensorrt_model_connect/diffusion_runner.py`

**Testing**: See [Testing and Validation](Testing-and-Validation.md#diffusion-domain-tests) for the 9-step component validation and frame quality checks.

**Adding a new diffusion family**: Create a plugin with `build_components()` that composes the shared builders, plus `get_diffusion_config()` for pipeline parameters. If the scheduler differs, add a new scheduler in `schedulers/`. If the architecture differs significantly from DiT, create a new builder module and C++ backend extending `DiffusionBackendBase`. See `families/wan_t2v.py` for an example.

### DeepSeek MLA (Multi-Latent Attention)

**Python changes**:
- Checkpoint mapper: Compressed Q/KV projections (`w_dq`, `w_uk`, `w_uv`, `kv_lora_rank`)
- Graph builder: Decomposed attention with low-rank KV

**C++ changes**:
- `DeviceKvCache` generalization for different cache shapes per layer

**Estimated work**: ~300 LOC Python + ~100 LOC C++.

### Hybrid SSM+Attention (Jamba, Zamba)

**Python + C++ changes**: Combination of Mamba and attention approaches.
- Per-layer type: some layers attention (KV cache), some Mamba (recurrent state)
- `HybridStepState` that manages both state types

**Estimated work**: ~500 LOC Python + ~300 LOC C++.

---

## Recommended Approach for New Families

### Tier 1: Python-only (existing runtime strategy), standard builder (implemented for 40+ families)
Standard and extended decoders using the parameterized graph builder:
- Already done: Qwen, LLaMA, Mistral, Gemma, Phi, Granite, InternLM, StarCoder2, GPT-2, OPT, Falcon, StableLM, OLMo, XGLM, GPT-NeoX, GPT-Neo, CodeGen, BLOOM, Nemotron
- Candidates: Yi (use llama), Baichuan, DeepSeek-dense, CodeLlama (use llama), Vicuna (use llama)
- ~30-60 LOC each, fully parallelizable

### Tier 2: Python custom graph builder (existing C++ plugin), implemented for 15+ families
Non-standard graph topologies with existing C++ backends:
- Already done: Phi-MoE (MoE, Python only), Mixtral (MoE, Python only), Mamba (SSM, Python + existing C++ backend), Qwen-VL (VL, Python + existing C++ image preprocessor), Wan2.1-T2V (diffusion, Python builders + C++ diffusion backend)
- Candidates: Other Mamba variants, LLaVA/InternVL (can reuse simple_chw/aspect_preserve_chw preprocessor), other DiT-based diffusion models (can reuse shared builders)
- ~200-500 LOC each

### Tier 3: Python + new C++ plugin (done for RWKV, Hybrid, Seq2Seq, Marian)
Fundamentally different architectures requiring new C++ state management or pipeline logic:
- RWKV (`rwkv_plugin.cpp` -- **implemented**)
- Hybrid SSM+Attention (`hybrid_plugin.cpp` -- **implemented**)
- T5 encoder-decoder (`t5_plugin.cpp` -- **implemented**)
- Marian machine translation (`marian_plugin.cpp` -- **implemented**)
- Seq2seq encoder-decoder (`seq2seq_plugin.cpp` -- **implemented**)
- DeepSeek MLA (Python + C++ cache shape, ~400 LOC total -- **not yet implemented**)

Each tier can be worked on independently. Tier 1 and 2 families are fully parallelizable since they only touch Python plugins with no shared file edits.
