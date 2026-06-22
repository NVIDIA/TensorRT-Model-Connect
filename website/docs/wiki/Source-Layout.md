# Source Layout

This page describes the actual directory structure of the repository. Every
path listed here exists in the source tree.

---

## Top-Level Layout

| Path | Purpose |
|------|---------|
| `include/trtmc/` | Public C/C++ API headers |
| `src/` | C++ runtime implementation |
| `python/tensorrt_model_connect/` | Python builder package |
| `tests/` | All test suites (C++, Python builder, tools, E2E) |
| `tools/` | Diff test framework, perf comparison, complexity checker |
| `scripts/` | Infrastructure and utility scripts |
| `website/docs/` | Website documentation, including the archived wiki pages |

---

## Public API Headers (`include/trtmc/`)

| File | Purpose |
|------|---------|
| `pipeline.h` | `IPipeline` interface, result types (`TextResult`, `ImageResult`, `AudioResult`, `EmbeddingResult`, `SegmentResult`, `TextEmbedding`), `GenerateConfig`, `trtmc::load()` factory, C ABI entry points |
| `bundle.h` | `BundleInfo` struct, `InspectBundle()`, `IsBundle()` |
| `tokenizer.h` | `ITokenizer` interface (full: `encode`, `decode`, `id_for_token`, `token_for_id`), factory functions for `VocabTokenizer`, `HfPythonTokenizer`, `IpaTokenizer` |
| `trtmc_io.hpp` | I/O helper types |

### `include/trtmc/runtime/`

| File | Purpose |
|------|---------|
| `pipeline_factory.h` | `PipelineFactory::from_bundle()` -- sole pipeline creation path |
| `trt_module.h` | `ITrtModule` -- pure virtual interface for TRT engine execution (`forward()`, `forward_device()`, `bind_external()`). Concrete impl lives in backend DSOs. |
| `trt_backend.h` | `IBackend` interface + `ModuleCreateOptions` for backend DSO dispatch |
| `kv_cache.h` | `KvCache` -- autoregressive KV cache with per-layer device tensors |
| `recurrent_state.h` | `RecurrentState` -- config-driven SSM/RWKV state manager |
| `scheduler.h` | `IScheduler` interface, `FlowMatchEulerScheduler` for diffusion |
| `tensor.h` | `Tensor`, `TensorMap`, `TensorInfo`, `DType` enum |
| `device_tensor.h` | `DeviceTensor` -- GPU-resident tensor with RAII |
| `tokenizer_interface.h` | Minimal `ITokenizer` (encode/decode only) |

### `include/trtmc/runtime/domains/audio/`

| File | Purpose |
|------|---------|
| `speech_decode_stop_policy.h` | Speech decode stopping criterion |
| `subprocess_runner.h` | Subprocess execution helper |

### `include/trtmc/runtime/domains/multimodal/`

| File | Purpose |
|------|---------|
| `image_transform_helper.h` | Image transform helpers for VL preprocessing |

---

## C++ Runtime Source (`src/`)

### `src/bundle/`

| File | Purpose |
|------|---------|
| `bundle_format.h` | Bundle magic bytes, `BundleSection`, `BundleFile`, read functions |
| `bundle_format.cpp` | `.trtfb` file reading implementation |

### `src/cabi/api/`

| File | Purpose |
|------|---------|
| `trtmc_c.cpp` | C ABI entry: `trtmc_create_pipeline()`, `trtmc_create_pipeline_ex()`, `trtmc_last_error()`, `trtmc_version()`, `trtmc_has_trt()` |

### `src/runtime/config/`

| File | Purpose |
|------|---------|
| `config_bundle.cpp` | Layered runtime config bundle merge and lookup |
| `schema_registry.cpp` | Runtime config schema registration and lookup |
| `cli_support.cpp` | CLI config-profile and `--set` parsing support |

### `src/cabi/bundle/`

| File | Purpose |
|------|---------|
| `bundle_helpers.h` | `BundleSections` struct, tokenizer/engine extraction |
| `bundle_helpers.cpp` | Bundle section extraction implementation |

### `src/runtime/backend/`

Backend DSO implementations. The main binary does not link libnvinfer -- it dlopen's a backend DSO at runtime based on the bundle's `engine_backend` config field. Each DSO exports an `IBackend` factory that creates `ITrtModule` instances.

| File | Purpose |
|------|---------|
| `backend_loader.h/cpp` | `BackendLoader::load()` -- dlopen's backend DSOs from the executable directory, explicit search dirs, or loader path; caches handles |
| `trt_module_impl.h/cpp` | `TrtModuleImpl` : `ITrtModule` -- shared engine wrapper compiled into both DSOs |
| `trt_backend.cpp` | Standard TRT `IBackend` impl. Links libnvinfer. Produces `libtrtmc_backend_trt.so` |
| `rtx_backend.cpp` | TRT-RTX `IBackend` impl. Links libtensorrt_rtx. Adds `IRuntimeCache` + `CudaGraphStrategy`. Produces `libtrtmc_backend_trt_rtx.so` |

### `src/runtime/registry/`

Factory, registry, and base config parsing for plugin dispatch.

| File | Purpose |
|------|---------|
| `pipeline_factory.cpp` | Central factory. Reads `.trtfb`, extracts strategy, normalizes legacy strings, looks up plugin, calls `plugin->create(ctx)` (~124 LOC, no switch/case) |
| `pipeline_registry.cpp` | Singleton registry mapping strategy strings to `IPipelinePlugin` instances |
| `pipeline_plugin.cpp` | `parse_base_config()` — universal base config from bundle JSON |

### `src/runtime/models/`

Model-owned runtime implementations. Each folder carries its plugin source,
pipeline source when needed, and a `MODEL.toml` manifest mapping that folder to
one or more `runtime_strategy` values. This keeps the C++ runtime code for a
strategy family in one place.

| Folder | Strategies | Pipeline class |
|------|-----------|---------------|
| `text_generation/` | `decoder_kv_cache`, `decoder_moe` | `TextGenerationPipeline` |
| `recurrent/` | `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention` | `RecurrentPipeline` |
| `encoder/` | `encoder_only`, `embedding`, `reranking`, `neural_operator`, `object_detection` | `EncoderPipeline` |
| `vision_language/` | `vision_language` | `VLPipeline` |
| `segmentation/` | `segmentation`, `prompted_segmentation` | `SegmentPipeline`, `SamPipeline` |
| `whisper/` | `speech_to_text` | `WhisperPipeline` |
| `bark/` | `text_to_audio_bark` | `BarkPipeline` |
| `magpie/` | `text_to_audio_magpie` | `MagpiePipeline` |
| `speech/` | `speech_to_speech` | `SpeechPipeline` |
| `omni/` | `omni_multimodal` | `OmniPipeline` |
| `t5/` | `text_to_text` | inline plugin pipeline |
| `marian/` | `marian_translation` | inline plugin pipeline |
| `seq2seq/` | `seq2seq_encoder_decoder` | inline plugin pipeline |
| `flux/` | `diffusion_flux` | `FluxPipeline` |
| `wan/` | `diffusion_wan`, `diffusion_pixart` | `WanPipeline` |
| `z_image/` | `diffusion_zimage` | `ZImagePipeline` |
| `pixart/`, `ltx_video/` | image/video diffusion strategies | family-specific pipelines |

Runtime helper code is model-local. Each `src/runtime/models/<model>/`
folder carries the helper copies it needs, such as `plugin_helpers.h/cpp`,
`diffusion_helpers.h/cpp`, or `audio_helpers.h/cpp`.

### `src/runtime/core/`

Common TRT runtime infrastructure.

| File | Purpose |
|------|---------|
| `trt_common.h/cpp` | TRT logger, CudaBuffer (RAII), CudaStream (RAII + move) |
| `trt_module.cpp` | Legacy `TrtModule` stubs (I/O binding delegates to `ITrtModule` from backend DSO) |
| `kv_cache.cpp` | `KvCache` implementation (bind_to, advance, mask, reset) |
| `recurrent_state.cpp` | `RecurrentState` implementation |
| `device_tensor.cpp` | `DeviceTensor` GPU memory management |
| `device_kv_cache.h/cpp` | Legacy `DeviceKvCache`, `DeviceResources`, `run_decoder_step_device()` |
| `device_kv_cache_update_plan.h` | KV cache update plan helpers |
| `trt_engine_lifecycle.h/cpp` | `DecoderStepEngine`, tensor validation |
| `trt_decode_runtime.h/cpp` | `select_argmax_token()`, `build_attention_mask()` |
| `trt_graph_builder.cpp` | TRT graph construction helpers |
| `flow_match_euler_scheduler.cpp` | `FlowMatchEulerScheduler` implementation |
| `generation_backend.h` | Legacy generation backend interface |
| `step_state.h` | Legacy `IStepState` interface |
| `decoded_image.h` | Decoded image data struct |
| `stb_impl.cpp` | stb_image implementation |

### `src/runtime/domains/audio/`

Audio-family backends and helpers.

| File | Purpose |
|------|---------|
| `whisper_backend.h/cpp` | Whisper encoder-decoder backend |
| `bark_backend.h/cpp` | Bark three-stage TTS backend |
| `magpie_tts_backend.h/cpp` | Magpie TTS backend |
| `speech_backend.h/cpp` | Speech-to-speech backend |
| `omni_backend.h/cpp` | Legacy omni-multimodal backend |
| `mel_spectrogram.h/cpp` | Mel filterbank extraction for Whisper |
| `audio_bundle_validation.h/cpp` | Audio bundle section validation |
| `audio_configs.h` | Audio model configuration structs |
| `magpie_kernels.cu/h` | CUDA kernels for Magpie |
| `bark_generation_plan.h` | Bark generation plan |
| `whisper_cross_kv_apply.h` | Whisper cross-attention KV application |
| `whisper_cross_kv_plan.h` | Whisper cross-KV plan |
| `whisper_decode_policy.h` | Whisper decode stopping policy |
| `whisper_host_plan.h` | Whisper host-side plan |
| `magpie_codec_plan.h` | Magpie codec plan |
| `magpie_decode_policy.h` | Magpie decode stopping policy |
| `magpie_decoder_plan.h` | Magpie decoder plan |
| `magpie_text_completion_policy.h` | Magpie text completion policy |
| `speech_delay_cache.h` | Speech delay cache |
| `speech_depth_plan.h` | Speech depth plan |
| `speech_generation_policy.h` | Speech generation stopping policy |
| `speech_mimi_decode_plan.h` | Speech MIMI decode plan |
| `speech_runtime_plan.h` | Speech runtime plan |
| `speech_temporal_embed_plan.h` | Speech temporal embedding plan |
| `speech_waveform_postprocess.h` | Speech waveform postprocessing |
| `omni_audio_plan.h` | Omni audio plan |

### `src/runtime/domains/diffusion/`

Diffusion-family helpers and types.

| File | Purpose |
|------|---------|
| `diffusion_types.h` | `DiffusionConfig`, `PreprocessorWeights`, `VideoResult` |
| `diffusion_math.h` | Math helpers (silu, gelu_tanh, timestep embedding) |
| `diffusion_preprocessor.cpp` | Preprocessor weight loading |
| `diffusion_preprocessor_weights_helpers.h` | Weight loading helpers |
| `diffusion_scheduler_helpers.h` | Scheduler step helpers |
| `diffusion_denoising_step_seam.h` | Denoising step abstraction |
| `diffusion_generation_plan.h` | Generation plan for diffusion |
| `wan_generation_conditioning.h` | Wan-specific conditioning |

### `src/runtime/domains/encoder/`

Encoder-family backends.

| File | Purpose |
|------|---------|
| `encoder_backend.h/cpp` | Encoder-only backend (BERT) |
| `embedding_backend.h/cpp` | Embedding model backend |
| `reranking_backend.h/cpp` | Reranking model backend |

### `src/runtime/domains/multimodal/`

Vision-language and multimodal support.

| File | Purpose |
|------|---------|
| `image_preprocessor.h/cpp` | VL image preprocessing (4 strategies) |
| `vision_engine.h/cpp` | Vision encoder engine lifecycle |
| `vl_backend.h/cpp` | Legacy VL backend |
| `vision_execution_plan.h` | Vision execution plan config |
| `vl_decode_policy.h` | VL decode step policy |

### `src/runtime/domains/perception/`

Segmentation and detection backends.

| File | Purpose |
|------|---------|
| `segmentation_backend.h/cpp` | SegFormer backend |
| `segmentation_preprocess_seam.h` | Segmentation preprocessing |
| `segmentation_postprocess_seam.h` | Segmentation postprocessing |
| `sam_backend.h/cpp` | SAM two-stage (encoder + decoder) backend |
| `sam_image_preprocess_seam.h` | SAM image preprocessing |
| `sam_prompt_seam.h` | SAM prompt encoding |
| `sam_output_selection.h` | SAM mask output selection |
| `sam_postprocess_seam.h` | SAM postprocessing |
| `detection_backend.h/cpp` | Object detection backend |
| `neural_operator_backend.h/cpp` | Neural operator (FNO) backend |

### `src/runtime/domains/recurrent/`

Recurrent model backends (Mamba, RWKV, Hybrid).

| File | Purpose |
|------|---------|
| `mamba_backend.h/cpp` | Legacy Mamba autoregressive loop |
| `mamba_decode_runtime.h/cpp` | Mamba step engine, `run_mamba_step` |
| `mamba_step_state.h/cpp` | Legacy `MambaStepState` |
| `rwkv_backend.h/cpp` | Legacy RWKV autoregressive loop |
| `rwkv_decode_runtime.h/cpp` | RWKV step engine |
| `rwkv_step_state.h/cpp` | Legacy `RwkvStepState` |
| `hybrid_backend.h/cpp` | Legacy Hybrid (Mamba + Attention) backend |
| `recurrent_step_contracts.h` | Shared recurrent step contracts |
| `recurrent_tensor_bindings.h` | Tensor binding helpers |

### `src/tokenizer/`

| File | Purpose |
|------|---------|
| `vocab_tokenizer.cpp` | Vocab.txt-based word-to-id tokenizer |
| `hf_python_tokenizer.cpp` | HuggingFace tokenizer via Python subprocess |
| `hf_python_tokenizer_helpers.h` | Shell quoting and parsing helpers |
| `ipa_tokenizer.cpp` | IPA phoneme tokenizer for speech models |

### `src/utils/`

| File | Purpose |
|------|---------|
| `data_dir.h/cpp` | Source/scripts directory resolution, env overrides |
| `text_parsers.h/cpp` | String/file parsing helpers (`starts_with`, `read_file`, etc.) |
| `json_helpers.h/cpp` | JSON extraction helpers (`extract_json_string`, etc.) |
| `wav_reader.h/cpp` | WAV file reading |
| `image_reader.cpp` | Image file reading (via stb_image) |

---

## Python Builder Package (`python/tensorrt_model_connect/`)

### Core Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package init, public `build()` API |
| `__main__.py` | `python -m tensorrt_model_connect` entry |
| `build_cli.py` | Python builder CLI used by `trtmc build` and `python -m tensorrt_model_connect` |
| `config.py` | `ModelConfig` from HF `config.json` |
| `checkpoint_mapper.py` | HF safetensors -> weight dict |
| `graph_ops.py` | Layer 1: atomic TRT graph ops |
| `graph_blocks.py` | Layer 2: composable blocks (attention, MLP, norm) |
| family-local `standard_decoder_builder.py` | Layer 3: standard decoder engine builder |
| `bundle_writer.py` | Write `.trtfb` files |
| `engine_builder.py` | Top-level orchestrator: HF -> TRT -> bundle |
| `debug_runner.py` | Pure-Python TRT inference (`TrtRunner`, `MambaTrtRunner`, `VLTrtRunner`) |
| `pipeline.py` | Pipeline subprocess wrapper |

### Specialized Builders

| File | Purpose |
|------|---------|
| `qwen_vl_vision_builder.py` | Vision encoder builders for Qwen2.5-VL and Qwen3-VL |
| `qwen3_encoder_builder.py` | Qwen3 encoder builder |
| `internvit_vision_builder.py` | InternViT vision builder |
| `phi4mm_vision_builder.py` | Phi4 multimodal vision builder |
| `onnx_vision_builder.py` | ONNX-based vision builder |
| `vision_encoder_builder.py` | Generic vision encoder builder |
| `encoder_builder.py` | Encoder-only model builder (BERT, embedding, reranking) |
| `mistral_encoder_builder.py` | Mistral encoder builder |
| `t5_encoder_builder.py` | T5 encoder builder |
| `clip_encoder_builder.py` | CLIP encoder builder |
| `standard_dit_builder.py` | Standard DiT (diffusion transformer) builder |
| `flux_dit_builder.py` | FLUX DiT builder |
| `families/flux/flux2_dit_builder.py` | FLUX2 DiT builder |
| `z_image_dit_builder.py` | Z-Image DiT builder |
| `vae_2d_builder.py` | 2D VAE builder |
| `flux_vae_builder.py` | FLUX VAE builder |
| `causal_vae_3d_builder.py` | Causal 3D VAE builder (Wan video) |
| `encodec_builder.py` | EnCodec audio codec builder |
| `nanocodec_builder.py` | NanoCodec builder |
| `diffusion_runner.py` | Python-side diffusion inference runner |

### Family Plugins (`python/tensorrt_model_connect/families/`)

63 auto-discovered family plugins. Each exports a module-level `plugin` attribute
implementing the `FamilyPlugin` protocol from `base.py`.

| Plugin | Model families |
|--------|---------------|
| `albert` | ALBERT |
| `bark` | Bark TTS |
| `bart` | BART |
| `bert` | BERT |
| `bloom` | BLOOM |
| `canary` | Canary ASR |
| `codegen` | CodeGen |
| `convbert` | ConvBERT |
| `deberta` | DeBERTa v1 (autopilot-generated) |
| `deepseek_ocr` | DeepSeek-OCR |
| `deepseek_v2` | DeepSeek-V2 |
| `distilbert` | DistilBERT |
| `dpr` | DPR (Dense Passage Retrieval) |
| `eagle_vlm` | Eagle VLM |
| `electra` | ELECTRA (autopilot-generated) |
| `falcon` | Falcon |
| `flux` | FLUX.1 |
| `fnet` | FNet |
| `gemma` | Gemma |
| `glm` | GLM |
| `gpt2` | GPT-2 |
| `gpt_neo` | GPT-Neo |
| `gpt_neox` | GPT-NeoX |
| `gpt_oss` | GPT-OSS |
| `granite` | Granite |
| `internlm` | InternLM |
| `internvl` | InternVL |
| `llama` | LLaMA |
| `m2m_100` | M2M-100 |
| `magpie_tts` | Magpie TTS |
| `mamba` | Mamba SSM |
| `marian` | Marian MT |
| `mistral` | Mistral |
| `mixtral` | Mixtral MoE |
| `modernbert` | ModernBERT (autopilot-generated) |
| `mpnet` | MPNet |
| `nemotron` | Nemotron |
| `nemotron_h` | Nemotron-H (Hybrid) |
| `olmo` | OLMo |
| `olmo2` | OLMo2 |
| `opt` | OPT |
| `personaplex` | PersonaPlex |
| `phi` | Phi-3 |
| `phi4_multimodal` | Phi-4 Multimodal |
| `phi_moe` | Phi-MoE |
| `pixart` | PixArt |
| `qwen` | Qwen, Qwen2, Qwen3 |
| `qwen3_5` | Qwen3.5 |
| `qwen3_omni` | Qwen3-Omni |
| `qwen_moe` | Qwen-MoE |
| `qwen_vl` | Qwen2.5-VL, Qwen3-VL |
| `roberta` | RoBERTa |
| `rwkv` | RWKV |
| `sam` | SAM |
| `segformer` | SegFormer |
| `stablelm` | StableLM |
| `starcoder2` | StarCoder2 |
| `t5` | T5 encoder-decoder (autopilot-generated) |
| `wan_t2v` | Wan2.1 Text-to-Video |
| `whisper` | Whisper |
| `xglm` | XGLM |
| `xlnet` | XLNet |
| `z_image` | Z-Image |

---

## Tests

### `tests/builder/` -- Python Builder Unit Tests

79 test modules (`test_*.py`) plus 3 infrastructure files (`conftest.py`,
`__init__.py`, `family_plugin_tester.py`).
Covers config parsing, checkpoint mapping, family plugins,
graph ops, graph blocks, standard decoder, bundle writer, engine builder,
debug runner, cache state machine, CLI, vision compute, and pipeline wrapper.

Key fixtures in `conftest.py`: `trt_runner` (GPU graph op testing),
`requires_trt` and `requires_tensorrt_model_connect` skip markers.

### `tests/cpp/` -- C++ Runtime Unit Tests

94 test executables. Plain `main()` programs with `check(condition, name)`
helpers. Registered in `CMakeLists.txt` with `add_executable` + `add_test`.

Covers: bundle format, tokenizers (vocab, HF Python), text/JSON parsers,
CLI args, data_dir, TRT logger, engine lifecycle, bundle helpers, image
preprocessor, CUDA buffer/stream, device KV cache, decode runtime, fast
path config, pipeline API, bundle E2E, C ABI entry.

Shared utilities in `test_helpers.h`.

### `tests/tools/` -- Tool Self-Tests

62 test modules (`test_*.py`) plus 2 infrastructure files (`conftest.py`,
`__init__.py`). Pure Python, no GPU needed. Covers diff framework, logit
comparison, audio diff, segmentation diff, diffusion helpers, perf compare,
parity testing, text comparator, E2E report generation, runtime strategy
matrix checker, E2E repro commands, runtime path guards, test impact analysis,
performance parity, and performance database.

### `tests/test_e2e.py` -- Unified E2E Entry Point

Single parametrized pytest file. One test case per model manifest in
`tests/e2e/models/`. Resolves paths, builds `RunContext`, invokes the
orchestrator.

### `tests/e2e/models/` -- Model Manifests

197 JSON manifest files are grouped under family-owned `manifests/`
directories and listed by 74 `MODEL.toml` indexes. Each manifest specifies
`hf_id`, `bundle`, `family`, `runtime_strategy`, `prompt`, `max_new_tokens`,
and optional fields like `logit_atol`, `trust_remote_code`, `skip`.

### `tests/e2e_harness/` -- E2E Test Framework

DIP-architected harness with plugin-based runners, references, and
comparators.

| File | Purpose |
|------|---------|
| `contracts.py` | `E2ECase`, `StageOutput`, `CompareResult`, protocols |
| `orchestrator.py` | Lifecycle: preflight -> build -> run -> compare |
| `registry.py` | Auto-discovery of runners/references/comparators |
| `manifest_loader.py` | JSON manifest -> `E2ECase` |
| `artifact_sink.py` | Persist artifacts (JSON, logits, audio, images) |
| `result_schema.py` | Result schema types |

#### `tests/e2e_harness/runners/`

| File | Task strategy |
|------|--------------|
| `text_generation.py` | `text_generation_causal` |
| `vision_language.py` | `vision_language_generation` |
| `audio_speech.py` | `speech_to_text`, `text_to_audio`, `speech_to_speech` |
| `diffusion.py` | `diffusion_media_generation` |
| `segmentation.py` | `segmentation`, `prompted_segmentation` |
| `embedding.py` | `embedding` |
| `reranking.py` | `reranking` |
| `encoder_only.py` | `encoder_only_nlp` |
| `neural_operator.py` | `neural_operator` |
| `object_detection.py` | `object_detection` |
| `omni.py` | `omni_multimodal` |

#### `tests/e2e_harness/references/`

| File | Purpose |
|------|---------|
| `hf_transformers.py` | HuggingFace Transformers reference |
| `hf_diffusers.py` | HuggingFace Diffusers reference |
| `torch_reference.py` | PyTorch reference (speech-to-speech) |
| `custom_python.py` | Custom Python reference scripts |
| `golden_snapshot.py` | Pre-computed golden snapshot reference |
| `invariant_only.py` | Invariant-only reference (no comparison) |

#### `tests/e2e_harness/comparators/`

| File | Modality |
|------|----------|
| `text.py` | Text generation (logit cosine, top-k, token agreement, NED) |
| `vision_language.py` | VL (vision cosine + text NED/agreement) |
| `text_to_audio.py` | TTS (RMS, duration ratio, mel distance) |
| `diffusion.py` | Diffusion (pixel stats, PSNR, SSIM) |
| `segmentation.py` | Segmentation (mIoU, pixel accuracy, boundary F-score) |
| `speech_to_text.py` | ASR transcript similarity |
| `audio.py` | Generic audio comparison |
| `speech_to_speech.py` | Speech-to-speech |
| `embedding.py` | Embedding cosine similarity |
| `encoder_only.py` | Encoder-only comparison |
| `reranking.py` | Reranking score comparison |
| `neural_operator.py` | Neural operator output comparison |
| `omni.py` | Omni-multimodal comparison |
| `_helpers.py` | Shared comparator helpers |

#### `tests/e2e_harness/thresholds/defaults/`

Per-strategy JSON threshold files defining default pass/fail criteria.

---

## Tools (`tools/`)

| File | Purpose |
|------|---------|
| `diff_logits.py` | E2E logit comparison (TRT vs HF, per-step) |
| `diff_layers.py` | Per-layer hidden state comparison |
| `diff_vl.py` | VL diff testing (vision features, generation) |
| `diff_audio.py` | Audio diff testing |
| `diff_segmentation.py` | Segmentation diff testing |
| `diff_personaplex.py` | PersonaPlex diff testing |
| `diff_t5.py` | T5 encoder diff testing |
| `diff.py` | Generic diff entry point |
| `diff_framework/` | Diff framework infrastructure (DiffResult, registry, runner) |
| `test_runner_parity.py` | Python vs C++ runtime parity verification |
| `test_graph_ops.py` | TRT graph operation testing |
| `perf_compare.py` | Performance benchmarking |
| `tool_helpers.py` | Shared helper functions (cosine_sim, compare_arrays) |
| `diffusion_helpers.py` | Diffusion-specific helpers |
| `check_cyclomatic_complexity.py` | Cyclomatic complexity gate (max CCN 10) |
| `check_legacy_runtime_freeze.py` | Legacy runtime freeze checker |
| `check_runtime_strategy_matrix.py` | Runtime strategy matrix validator |
| `validate_dit.py` | DiT model validation |
| `validate_t5.py` | T5 encoder validation |
| `debug_diffusion_pipeline.py` | Diffusion pipeline debugging |
| `coverage/` | Coverage tooling |
| `coverage_ci/` | CI coverage integration |

---

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `setup_container.sh` | One-shot container repo setup (editable install + build + tests) |
| `bootstrap_workspace.sh` | Bootstrap isolated workspace per agent |
| `docker_build_gb300.sh` | Build GB300 dev container |
| `docker_run_gb300.sh` | Launch GB300 dev container |
| `validate_family.sh` | One-command family validation (build + diff + parity) |
| `new_family.py` | Scaffold a new family plugin from HF repo |
| `autopilot/autorun.py` | One-command autopilot: discover → select → dispatch → validate |
| `autopilot/discover.py` | Query HF Hub → find unsupported model_types → JSON task list |
| `autopilot/dispatch.py` | Launch parallel agent CLI sessions across agent workspaces |
| `autopilot/run.sh` | Shell wrapper for discover + dispatch |
| `check_family_coverage.py` | Check family plugin test coverage |
| `hf_generate.py` | HuggingFace generation script |
| `hf_tokenizer.py` | HuggingFace tokenizer utility |
| `eval_mmlu.py` | MMLU evaluation script |
| `generate_e2e_report.py` | E2E test report generator |
| `run_e2e_parallel.sh` | Parallel E2E test runner |
| `schedule_e2e.py` | E2E test scheduler |
| `build_wan14b.py` | Wan 14B model builder |
| `magpie_codec_bridge.py` | Magpie codec bridge utility |
| `magpie_tokenizer.py` | Magpie tokenizer utility |
| `profile_magpie_tts.py` | Magpie TTS profiler |

---

## Documentation (`website/docs/`)

| File | Purpose |
|------|---------|
| `Home.md` | Wiki home page |
| `Architecture-Overview.md` | High-level architecture (ARCH-* contracts) |
| `Static-Design.md` | Unit design specification (UD-* identifiers) |
| `Dynamic-Design.md` | Runtime behavior and sequence diagrams |
| `Source-Layout.md` | This file |
| `Pipeline-Deep-Dive.md` | Pipeline architecture deep dive |
| `Runtime-Target-Architecture.md` | Runtime target architecture |
| `Architecture-Extensibility-Assessment.md` | Extensibility assessment |
| `Adding-a-Model-Family.md` | Guide for adding new model families |
| `HF-vs-TRT-Comparison.md` | HuggingFace vs TRT comparison |
| `TRT-Internals.md` | TensorRT internals |
| `Testing-and-Validation.md` | Test strategy and validation |
| `Traceability-Matrix.md` | Bi-directional traceability (ARCH -> UD -> UT/IT) |
| `diagrams/` | Mermaid and other diagram sources |
