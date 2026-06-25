# Testing and Validation

Comprehensive manual for the tensorrt-model-connect test infrastructure. Covers every abstraction layer, source file locations, intentions, pytest markers, and commands for running each suite.

---

## Test Architecture at a Glance

The test suite is organized into **six abstraction layers**, each with a distinct
purpose, dependency profile, and speed:

| Layer | Directory | Files | Tests | GPU? | Time | Purpose |
|-------|-----------|:--:|:--:|:--:|------|---------|
| 1. Builder unit | `tests/builder/` | 98 | ~940 | No | ~10 min | Python build logic in isolation |
| 2. C++ runtime unit | `tests/cpp/` | 92 | 70+ | Mix | ~8 s | C++ runtime correctness |
| 3. Tools self-tests | `tests/tools/` | 62 | ~160 | No | ~35 s | Diff framework + comparison utilities |
| 4. Graph-op GPU | `tests/builder/test_graph_*.py` | 3 | ~70 | TRT | ~2 min | TRT graph operations on real GPU |
| 5. Model E2E | `tests/e2e/models/<family>/` + `tests/e2e_harness/` | 197 manifests + 74 indexes | 197 | GPU | 2-3 h | Full pipeline (build + infer + compare) |
| 6. Diff framework | `tools/diff_logits.py`, `tools/diff_layers.py`, etc. | 6 checks | -- | GPU | varies | Ad-hoc TRT-vs-HF model comparison |

**Philosophy**: Every TRT engine must produce output matching HuggingFace
Transformers (the ground truth). Testing validates this at multiple
granularities: per-layer hidden states, per-step logits, full generation text,
and (for non-text modalities) modality-specific quality metrics.

---

## Test Intent Contract (Required for Every Test)

Every test added or modified in this repository must document:

| Field | Requirement |
|------|-------------|
| Intent | What behavior/contract the test proves (not implementation details) |
| Preconditions | Required setup assumptions: fixtures, runtime strategy, input shape/model capabilities, environment toggles |
| Postconditions | Observable outcomes that must hold after test execution (assertions/invariants) |

Required placement:
- Python tests: docstring on the test function/class.
- C++ tests: comment block directly above the `check(...)` sequence for that scenario.

Every documented test must also carry trace IDs (`ARCH-*`, `UD-*`, and test ID such as `UT-*`/`IT-*`) and map into [Traceability Matrix](Traceability-Matrix.md).

Python example:

```python
def test_runtime_strategy_matrix_includes_vision_language():
    """
    Intent: Validate that `vision_language` remains connected to the VL runner/comparator contract.
    Preconditions:
      - tests/runtime_strategy_matrix.yaml defines runtime_strategies.vision_language.
      - Registry modules for runner/comparator classes are importable.
    Postconditions:
      - runner_class resolves to VisionLanguageRunner.
      - comparator_class resolves to VisionLanguageComparator.
    Trace: ARCH-RT-002, UD-REG-VISION-001, UT-TOOLS-STRATEGY-MATRIX-002
    """
```

C++ example:

```cpp
// Intent: Validate FastPathModelConfig preserves diffusion runtime strategy from bundle config.
// Preconditions:
//   - Input config JSON includes "runtime_strategy": "diffusion".
//   - Parser is called through fast-path config load flow used by trtmc_c.cpp.
// Postconditions:
//   - Parsed runtime strategy equals "diffusion".
//   - Downstream dispatch can branch to create_diffusion_pipeline(...).
// Trace: ARCH-RT-003, UD-CFG-FASTPATH-001, UT-CPP-FASTPATH-CONFIG-001
```

### Bi-Directional Traceability Workflow

Use [Traceability Matrix](Traceability-Matrix.md) as the repository-level index from architecture to tests and back:

1. Add/update an `ARCH-*` contract row for the behavior being changed.
2. Link all affected design units (`UD-*`) such as strategy plugins, factories, runners, and comparators.
3. Link unit tests (`UT-*`) and integration tests (`IT-*`) that prove the contract.
4. Record verification evidence (command/artifacts/date).
5. Perform reverse check: from each changed test, confirm a valid `UD-*` and `ARCH-*` target exists.

Rows are incomplete until all four links are present:
`ARCH -> UD -> {UT, IT}` and `{UT, IT} -> UD -> ARCH`.

---

## Layer 1: Python Builder Unit Tests

**Directory**: `tests/builder/`

**Intent**: Verify Python build logic in isolation -- config parsing, weight
loading, checkpoint mapping, bundle writing, engine orchestration, family
plugin dispatch, graph ops, and debug runner infrastructure. No GPU required
for the majority of tests.

**How to run**:

```bash
# All builder tests (no GPU needed for most; TRT tests auto-skip when unavailable)
.venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Only unit-tier tests (Tier 0 + Tier 1, never needs GPU)
.venv/bin/python -m pytest tests/builder/ -v -m unit --ignore=tests/builder/test_cli.py

# Only GPU/TRT tests
.venv/bin/python -m pytest tests/builder/ -v -m gpu --ignore=tests/builder/test_cli.py
```

**Pytest markers used**: `@pytest.mark.unit` (no GPU), `@pytest.mark.trt` (needs TRT), `@pytest.mark.gpu` (needs GPU).

**Skip markers**: All files use `try/except` with `pytest.skip(allow_module_level=True)` so they skip cleanly when TRT or `tensorrt_model_connect` is not installed. GPU tests also use a `@requires_trt` skipif decorator.

### Sub-categories

#### Configuration & data parsing (no GPU)

| File | What it tests |
|------|---------------|
| `test_config.py` | `ModelConfig` parsing, VL `text_config` merge, edge cases, negative dimensions, type mismatches |
| `test_checkpoint_mapper.py` | Weight loading, GQA expansion (including 8x single-head), tied embeddings, biases |
| `test_bundle_writer.py` | Bundle format round-trip, section integrity, corrupted bundle detection (bad magic, truncated header/file) |
| `test_cache_state_machine.py` | Position, mask, cache append/shift logic, edge cases (max_cache_length=0, max_cache_length=1) |
| `test_manifest_validation.py` | E2E manifest schema validation (required fields, type checks, unknown runtime_strategy warnings) |

#### Family plugins (no GPU)

| File | What it tests |
|------|---------------|
| `test_families.py` | Plugin match/dispatch, runtime_strategy, embed_input, `matches()` returns bool |
| `test_families_coverage.py` | Plugin coverage tests |
| `test_family_plugins.py` | 10 family plugins: `load_weights()` correctness |
| `test_family_plugins_extended.py` | Extended family plugin tests |
| `test_family_plugins_extended2.py` | Additional extended family plugin tests |
| `test_family_bert.py` | BERT-specific plugin tests |
| `test_family_deepseek_v2.py` | DeepSeek-V2 plugin tests |
| `test_family_distilbert.py` | DistilBERT plugin tests |
| `python/tensorrt_model_connect/families/flux/tests/test_family.py` | FLUX diffusion plugin tests |
| `test_family_gpt_oss.py` | GPT-OSS plugin tests |
| `test_family_mpnet.py` | MPNet plugin tests |
| `test_family_nemotron_h.py` | Nemotron-H hybrid plugin tests |
| `test_family_phi4mm.py` | Phi-4 multimodal family |
| `test_family_pixart.py` | PixArt diffusion plugin tests |
| `test_family_qwen3_5.py` | Qwen3.5 family tests |
| `test_family_qwen_moe.py` | Qwen MoE plugin |
| `test_family_roberta.py` | RoBERTa plugin tests |
| `test_family_sam.py` | SAM prompted segmentation plugin |
| `test_family_wan_t2v.py` | Wan T2V diffusion plugin tests |
| `test_family_yolox.py` | YOLOX object detection plugin |
| `test_family_z_image.py` | Z-Image diffusion plugin tests |

#### Per-family engine tests (mixin-based, 3-tier)

These files inherit from `FamilyPluginTestMixin` in `family_plugin_test_mixin.py` and follow a standardized 3-tier pattern:

- **Tier 0** (`@pytest.mark.unit`): Plugin discovery, matching, required methods
- **Tier 1** (`@pytest.mark.unit`): Weight loading from synthetic safetensors -- correct keys, shapes, dtypes, determinism
- **Tier 2** (`@pytest.mark.trt`, `@pytest.mark.gpu`): Build real TRT engine, validate I/O tensor names and logits output shape

| File | Family | model_type | Tier 2 |
|------|--------|------------|:--:|
| `test_engine_bark.py` | bark | `bark` | skip (custom builder) |
| `test_engine_bloom.py` | bloom | `bloom` | yes |
| `test_engine_codegen.py` | codegen | `codegen` | yes |
| `test_engine_falcon.py` | falcon | `falcon` | yes |
| `test_engine_gemma.py` | gemma | `gemma2` | yes |
| `test_engine_gpt2.py` | gpt2 | `gpt2` | yes |
| `test_engine_gpt_neo.py` | gpt_neo | `gpt_neo` | yes |
| `test_engine_gpt_neox.py` | gpt_neox | `gpt_neox` | yes |
| `test_engine_granite.py` | granite | `granite` | yes |
| `test_engine_internlm.py` | internlm | `internlm2` | yes |
| `test_engine_llama.py` | llama | `llama` | yes |
| `test_engine_mamba.py` | mamba | `mamba` | skip (custom builder) |
| `test_engine_mistral.py` | mistral | `mistral` | yes |
| `test_engine_mixtral.py` | mixtral | `mixtral` | yes |
| `test_engine_nemotron.py` | nemotron | `nemotron` | yes |
| `test_engine_olmo.py` | olmo | `olmo` | yes |
| `test_engine_opt.py` | opt | `opt` | yes |
| `test_engine_phi.py` | phi | `phi3` | yes |
| `test_engine_phi_moe.py` | phi_moe | `phimoe` | skip (custom builder) |
| `test_engine_qwen.py` | qwen | `qwen3` | yes |
| `test_engine_rwkv.py` | rwkv | `rwkv6` | skip (custom builder) |
| `test_engine_segformer.py` | segformer | `segformer` | skip (custom builder) |
| `test_engine_stablelm.py` | stablelm | `stablelm` | yes |
| `test_engine_starcoder2.py` | starcoder2 | `starcoder2` | yes |
| `test_engine_whisper.py` | whisper | `whisper` | skip (custom builder) |
| `test_engine_xglm.py` | xglm | `xglm` | yes |

#### Graph operations (needs TRT/GPU)

| File | What it tests |
|------|---------------|
| `test_graph_ops.py` | 18 atomic graph ops: RoPE, ALiBi, RMSNorm, LayerNorm, attention, etc. |
| `test_graph_ops_extended.py` | YaRN RoPE, T5 relative bias, extended ALiBi, conv/norm/ELU/pad ops |
| `test_graph_blocks.py` | Composable blocks: `apply_norm`, SwiGLU MLP, GELU FC MLP |

Graph-op tests use the `trt_runner` fixture from `conftest.py`: a `build_fn(network, inputs)` closure constructs a small TRT graph, the fixture builds an engine, runs inference, and returns NumPy outputs for comparison against PyTorch/NumPy references via `np.testing.assert_allclose`.

#### Builder orchestration (mock-based, no GPU)

| File | What it tests |
|------|---------------|
| `test_engine_builder.py` | Engine builder mock tests |
| `test_engine_builder_extended.py` | `build_bundle` orchestration, GPU name, TRT version |
| `test_engine_builder_utils.py` | Engine builder utility tests |
| `test_pipeline.py` | Pipeline subprocess wrapper, binary detection |
| `test_debug_runner.py` | Debug runner mock tests |
| `test_debug_runner_extended.py` | Bundle section loading, runner cleanup, generate sequencing |
| `test_cli.py` | CLI inspect/build command dispatch (excluded from default run) |
| `test_cli_coverage.py` | CLI coverage tests |
| `test_quantization.py` | FP16/FP8/INT8/INT4/NVFP4/W4A8 quantization framework tests |
| `test_bark_tokenizer.py` | Bark tokenizer tests |
| `test_magpie_tokenizer_script.py` | Magpie-owned tokenizer module tests |
| `test_owned_builder_mocked_paths.py` | Builder mocked path tests |
| `test_owned_encoder_builders_coverage.py` | Encoder builder coverage tests |
| `test_owned_qwen3_t5_helpers.py` | Qwen3/T5 helper tests |
| `test_owned_schedulers.py` | Scheduler tests |

#### Additional configuration & coverage (no GPU)

| File | What it tests |
|------|---------------|
| `test_config_coverage.py` | ModelConfig coverage tests |
| `test_checkpoint_mapper_coverage.py` | Checkpoint mapper coverage tests |

#### Build-engine integration tests (needs TRT)

| File | What it tests |
|------|---------------|
| `test_build_engine_decoders.py` | Decoder engine build integration tests |
| `test_build_engine_std_decoders.py` | Standard decoder engine build tests |
| `test_build_engine_enc_dec.py` | Encoder-decoder engine build tests |
| `test_build_engine_integration.py` | Engine build integration tests |

#### Standard decoder & vision (needs TRT)

| File | What it tests |
|------|---------------|
| `test_standard_decoder.py` | Tensor naming contract, debug outputs |
| `test_vision_compute.py` | Vision encoder tests |
| `test_vision_compute_extended.py` | Vision RoPE, DeepStack config, patch embed, spatial merge |

---

## Layer 2: C++ Runtime Unit Tests

**Directory**: `tests/cpp/`

**Intent**: Verify C++ runtime correctness -- bundle parsing, tokenizers,
CUDA RAII wrappers, KV cache device operations, TRT engine lifecycle, image
preprocessing, CLI argument parsing, and helper utilities.

**How to run**:

```bash
# All C++ tests
ctest --test-dir build --output-on-failure

# Specific test
ctest --test-dir build -R test_bundle_format --output-on-failure
```

**Implementation**: Plain `main()` executables with no test framework. A
`check(condition, name)` helper accumulates `failures`; `main()` returns
non-zero if any failed. TRT-dependent tests guard with `#if TRTMC_HAS_TRT`
and skip gracefully (exit 0).

**RAII guards** (`test_helpers.h`):
- `EnvVarGuard` -- saves/restores environment variables (prevents env leakage between tests)
- `TempDirGuard` -- creates temp directory on construction, `remove_all` on destruction

### File inventory

#### Bundle and format tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_bundle_format.cpp` | Bundle magic, section parsing, round-trip | No |
| `test_bundle_e2e.cpp` | Bundle build + load round-trip | TRT |
| `test_bundle_view.cpp` | Bundle view API | No |
| `test_trtmc_io.cpp` | Bundle I/O operations | No |

#### Tokenizer tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_vocab_tokenizer.cpp` | Encode/decode, round-trip, case insensitivity | No |
| `test_bpe_tokenizer.cpp` | BPE tokenizer encode/decode | No |
| `test_bpe_golden.cpp` | BPE golden reference tests | No |
| `test_bpe_benchmark.cpp` | BPE tokenizer performance | No |
| `test_wordpiece_tokenizer.cpp` | WordPiece tokenizer encode/decode | No |
| `test_wordpiece_golden.cpp` | WordPiece golden reference tests | No |
| `test_unigram_tokenizer.cpp` | Unigram tokenizer encode/decode | No |
| `test_unigram_golden.cpp` | Unigram golden reference tests | No |
| `test_ipa_tokenizer.cpp` | IPA phoneme tokenizer | No |

#### CUDA and device tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_cuda_buffer.cpp` | RAII alloc, move semantics, data round-trip (with index on mismatch) | GPU |
| `test_cuda_stream.cpp` | RAII stream, move semantics | GPU |
| `test_device_tensor.cpp` | GPU-resident tensor operations | GPU |
| `test_kv_cache_new.cpp` | Additional KV cache tests | GPU |

#### TRT engine and runtime tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_trt_engine_lifecycle.cpp` | `layer_tensor_name`, constants | TRT |
| `test_trt_engine_lifecycle_fake_engine.cpp` | Engine lifecycle with fake engines | TRT |
| `test_trt_logger.cpp` | Severity names, error storage, explicit config controls | TRT |
| `test_trt_module.cpp` | TrtModule construction, tensor binding, lifecycle | TRT |
| `test_trt_runtime_lifetime.cpp` | TRT runtime lifetime management | TRT |
| `test_decode_runtime.cpp` | Argmax, mask building | TRT |
| `tests/cpp/models/qwen_image/test_qwen_image_flow_match_scheduler.cpp` | Qwen Image-owned Flow-matching Euler scheduler | No |

#### Pipeline and plugin tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_pipeline_api.cpp` | C API pipeline creation | TRT |
| `test_pipeline_registry.cpp` | Plugin registry, strategy lookup | TRT |
| `test_c_abi_entry.cpp` | C ABI entry point | TRT |
| `test_c_abi_runtime_regression.cpp` | C ABI runtime regression tests | TRT |
| `tests/cpp/models/llama/test_llama_pipeline.cpp` | Text generation pipeline | TRT |
| `test_encoder_pipeline.cpp` | Encoder pipeline (BERT, embedding, reranking) | TRT |
| `test_recurrent_pipeline.cpp` | Recurrent pipeline (Mamba, RWKV, hybrid) | TRT |
| `tests/cpp/models/*/test_*_recurrent_pipeline.cpp` | Model-owned recurrent state management through recurrent pipelines | TRT |
| `tests/cpp/models/*/test_*_recurrent_output_initializers.cpp` | Model-owned recurrent output initializers and step contracts | No |
| `test_vl_pipeline.cpp` | Vision-language pipeline | TRT |

#### Audio domain tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_audio_bundle_validation.cpp` | Bundle section validation for audio models | No |
| `test_audio_pipeline_new.cpp` | Audio pipeline construction | TRT |
| `test_bark_generation_plan.cpp` | Bark multi-stage codebook generation plan | No |
| `models/whisper/test_whisper_mel_spectrogram.cpp` | Whisper-owned mel spectrogram feature extraction | No |
| `models/canary/test_canary_mel_spectrogram.cpp` | Canary-owned mel spectrogram feature extraction | No |
| `models/nemotron_speech_streaming/test_nemotron_speech_streaming_audio_helpers.cpp` | RNNT-owned mel spectrogram feature extraction | No |
| `test_whisper_decode_policy.cpp` | Whisper decode policy | No |
| `test_whisper_host_plan.cpp` | Whisper host plan | No |
| `test_magpie_codec_plan.cpp` | Magpie TTS codec plan | No |
| `test_magpie_decode_policy.cpp` | Magpie TTS decode policy | No |
| `test_magpie_decoder_plan.cpp` | Magpie TTS decoder plan | No |
| `test_magpie_text_completion_policy.cpp` | Magpie TTS text completion policy | No |
| `test_omni_audio_plan.cpp` | Omni multimodal audio plan | No |
| `test_wav_reader.cpp` | WAV file reading | No |

#### Speech domain tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_speech_decode_stop_policy.cpp` | Speech decode stop policy | No |
| `test_speech_depth_plan.cpp` | Speech depth plan | No |
| `test_speech_generation_helpers.cpp` | Speech generation helpers | No |
| `test_speech_mimi_decode_plan.cpp` | Speech MIMI decode plan | No |
| `test_speech_runtime_plan.cpp` | Speech runtime plan | No |
| `test_speech_subprocess_seam.cpp` | Speech subprocess seam | No |
| `test_speech_temporal_embed_plan.cpp` | Speech temporal embedding plan | No |

#### Diffusion domain tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `models/flux/test_flux_denoising_step_seam.cpp` | Flux-owned denoising step seam | No |
| `models/wan/test_wan_denoising_step_seam.cpp` | Wan-owned denoising step seam | No |
| `models/pixart/test_pixart_denoising_step_seam.cpp` | PixArt-owned denoising step seam | No |
| `models/flux/test_flux_generation_plan.cpp` | Flux-owned generation plan | No |
| `models/wan/test_wan_generation_plan.cpp` | Wan-owned generation plan | No |
| `test_diffusion_math.cpp` | Diffusion math helpers | No |
| `models/flux/test_flux_pipeline.cpp` | Flux pipeline construction | TRT |
| `models/wan/test_wan_pipeline.cpp` | Wan pipeline construction | No |
| `models/z_image/test_z_image_pipeline.cpp` | Z Image pipeline construction | No |
| `models/ltx_video/test_ltx_video_pipeline.cpp` | LTX Video pipeline construction | No |
| `models/wan/test_wan_generation_conditioning.cpp` | Wan-specific generation conditioning | No |

#### Perception and segmentation tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `models/segformer/test_segformer_preprocess_seam.cpp` | SegFormer preprocessing seam | No |
| `models/segformer/test_segformer_postprocess_seam.cpp` | SegFormer postprocessing seam | No |
| `models/sam/test_sam_image_preprocess_seam.cpp` | SAM image preprocessing seam | No |
| `models/sam/test_sam_prompt_seam.cpp` | SAM prompt encoding seam | No |

#### Image and multimodal tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_image_preprocessor.cpp` | All 4 strategies, config parsing, prompt formatting (via `TempDirGuard`) | No |
| `test_image_reader.cpp` | Image file reading | No |

#### Utility and helper tests

| File | What it tests | GPU? |
|------|---------------|:--:|
| `test_cli_args.cpp` | CLI argument parsing | No |
| `test_data_dir.cpp` | Source/scripts dir resolution, env overrides (via `EnvVarGuard`) | No |
| `test_text_parsers.cpp` | String/file parsing helpers | No |
| `test_json_helpers.cpp` | JSON extraction helpers | No |

---

## Layer 3: Tools Self-Tests

**Directory**: `tests/tools/`

**Intent**: Verify diff framework and comparison utilities in isolation --
logit comparison, layer diffing, perf benchmarking, audio/segmentation metrics
-- without needing models or GPU.

**How to run**:

```bash
.venv/bin/python -m pytest tests/tools/ -v
```

**Implementation**: Pure Python tests, no TRT/GPU needed. `conftest.py` adds
`tools/` to import path. Tests use `importlib.import_module` for lazy
importing. Comparison logic tested with synthetic NumPy arrays.

### File inventory

| File | What it tests |
|------|---------------|
| `test_tool_helpers.py` | `cosine_sim`, `compare_arrays` |
| `test_diff_logits.py` | Logit comparison, argmax match, top-k overlap |
| `test_diff_layers.py` | Layer-wise hidden state comparison |
| `test_diff_vl.py` | Vision-language diff utilities |
| `test_diff_audio.py` | Energy computation, WAV I/O round-trip, token stats |
| `test_segformer_diff_segmentation.py` | SegFormer pixel agreement, logit diff, argument parsing |
| `test_diff_framework.py` | `DiffResult`, registry, runner, CLI parsing |
| `test_diffusion_helpers.py` | silu, gelu_tanh, bundle config/weights, timestep embedding |
| `test_parity.py` | Text/token comparison for runner parity |
| `test_perf_compare.py` | Stats, formatting, JSON output, serial GPU execution |
| `test_perf_parity.py` | Performance parity validation |
| `test_perfdb.py` | Performance database utilities |
| `test_text_comparator.py` | Text comparator logic |
| `test_coverage_map.py` | Coverage mapping utilities |
| `test_generate_report.py` | Report generation |
| `test_generate_perf_report.py` | Performance report generation |
| `test_test_impact.py` | Test impact analysis |
| `test_e2e_repro_commands.py` | E2E reproduction command generation |
| `test_e2e_runner_cli_alignment.py` | E2E runner CLI alignment validation |
| `test_e2e_runtime_path_guard.py` | E2E runtime path guard validation |
| `test_runtime_strategy_matrix_checker.py` | Runtime strategy matrix consistency checks |
| `test_prompted_segmentation_harness.py` | Prompted segmentation harness validation |

---

## Layer 4: Graph-Op GPU Tests

**Directory**: `tests/builder/test_graph_ops.py`, `test_graph_ops_extended.py`, `test_graph_blocks.py`

**Intent**: Validate TRT graph operations (RMSNorm, RoPE, attention, conv, etc.)
and composable graph blocks (SwiGLU MLP, GELU MLP, attention block) on real GPU
with real TRT engine execution.

**How to run**:

```bash
# All graph-op GPU tests
.venv/bin/python -m pytest tests/builder/test_graph_ops.py \
  tests/builder/test_graph_ops_extended.py \
  tests/builder/test_graph_blocks.py -v -m trt

# A single op
.venv/bin/python -m pytest tests/builder/test_graph_ops.py::TestRMSNorm -v
```

**Dependency**: Requires TRT + GPU. Uses `trt_runner` conftest fixture for
engine build + execution.

---

## Layer 5: Model E2E Tests

**Directory**: `tests/e2e/models/<family>/` + `tests/e2e_harness/`

**Intent**: Validate the full pipeline end-to-end -- build bundle from HF,
run C++ inference, compare output against HuggingFace reference. This is the
gold-standard correctness gate. Each family owns its pytest runner surface,
manifests, optional waives, and default artifact folder while the shared
harness provides contracts and orchestration.

**How to run**:

```bash
# Single model (auto-builds bundle if missing)
.venv/bin/python -m pytest tests/e2e/models/qwen --e2e-model qwen3-0.6b -v \
  --engine-dir /mnt/storage/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
  --model-plugin-dir ./build/models

# Force rebuild bundle from HF
.venv/bin/python -m pytest tests/e2e/models/qwen --e2e-model qwen3-0.6b -v \
  --engine-dir /mnt/storage/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
  --model-plugin-dir ./build/models --rebuild-engines

# All 197 models with artifact output
.venv/bin/python -m pytest tests/e2e/models -v \
  --engine-dir /mnt/storage/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
  --model-plugin-dir ./build/models --rebuild-engines \
  --e2e-artifacts-dir /tmp/e2e_artifacts

# Filter by modality
.venv/bin/python -m pytest tests/e2e/models -v \
  --e2e-task-strategy text_generation_causal \
  --engine-dir /mnt/storage/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
  --model-plugin-dir ./build/models
```

`tests/test_e2e.py` remains available for repository-wide compatibility runs,
but new model work should use the owning family directory.

**Available `--e2e-task-strategy` values**:

| Strategy | Models | Runner |
|----------|:--:|--------|
| `text_generation_causal` | 38 | `text_generation.py` |
| `encoder_only_nlp` | 18 | `encoder_only.py` |
| `vision_language_generation` | 5 (1 skip) | `vision_language.py` |
| `diffusion_media_generation` | 4 | `diffusion.py` |
| `speech_to_text` | 4 | `audio_speech.py` |
| `text_to_audio` | 3 | `audio_speech.py` |
| `t5_text_to_text` | 3 | `text_generation.py` |
| `embedding` | 2 | `embedding.py` |
| `speech_to_speech` | 1 | `audio_speech.py` |
| `segmentation` | 1 | `segmentation.py` |
| `prompted_segmentation` | 1 | `segmentation.py` |
| `reranking` | 1 | `reranking.py` |

### E2E harness architecture (DIP-first)

```
tests/e2e/models/*/MODEL.toml        # Per-family manifest indexes
tests/e2e/models/*/runner.py         # Model-owned pytest runner
tests/e2e/models/*/test_*_e2e.py     # Model-owned pytest entrypoint
tests/e2e/models/*/e2e_plugins/*.py  # Optional model-owned runner/reference/comparator plugins
tests/e2e/models/*/manifests/*.json  # 197 per-model JSON manifests
tests/e2e/models/*/thresholds/*.json # Model-owned threshold sidecars
tests/e2e/models/*/waives.txt        # Optional model-owned waives
tests/e2e_harness/
  __init__.py                        # save_full_stderr() helper
  contracts.py                       # E2ECase, StageOutput, CompareResult, protocols
  orchestrator.py                    # Lifecycle: preflight -> build -> run -> compare
  manifest_loader.py                 # JSON -> E2ECase (with schema validation)
  registry.py                        # Auto-discover runners/references/comparators
  artifact_sink.py                   # Persist artifacts (JSON, logits, audio, images)
  runners/                           # TRT strategy runners (one per task_strategy)
    text_generation.py               # Causal LM (decoder, MoE, SSM, RWKV)
    vision_language.py               # VL models (Qwen VL, InternVL)
    audio_speech.py                  # Whisper, Bark, PersonaPlex
    diffusion.py                     # Wan T2V, FLUX, Z-Image
    segmentation.py                  # SegFormer, SAM
    embedding.py                     # Eagle-embed
    reranking.py                     # Eagle-rerank
    encoder_only.py                  # BERT
    omni.py                          # Qwen3-Omni
    object_detection.py              # YOLOX
    neural_operator.py               # DeepONet / FNO
  references/                        # Gold-standard reference backends
    hf_transformers.py               # HF Transformers (text, VL, audio, seg)
    hf_diffusers.py                  # HF Diffusers (Wan, FLUX, Z-Image)
    torch_reference.py               # PyTorch (speech-to-speech)
    custom_python.py                 # Custom Python scripts
    golden_snapshot.py               # Pre-saved reference data
    invariant_only.py                # No external reference (self-consistency)
  comparators/                       # Metric computation + threshold gating
    text.py                          # 6-metric composite: logit cosine, top1 match, NED
    vision_language.py               # Vision cosine + text NED/agreement
    text_to_audio.py                 # RMS, duration ratio, mel/spectral distance
    speech_to_text.py                # Transcript text similarity
    speech_to_speech.py              # Audio quality metrics
    diffusion.py                     # Pixel stats, temporal consistency, PSNR/SSIM
    segmentation.py                  # mIoU, pixel accuracy, boundary F-score
    encoder_only.py                  # Hidden state / CLS cosine similarity
    embedding.py                     # Embedding cosine distance
    reranking.py                     # Score correlation
    omni.py                          # Multi-modal composite
    neural_operator.py               # Field comparison
    audio.py                         # Re-export umbrella
  thresholds/                        # Default + per-model threshold profiles
    defaults/                        # Per-strategy JSON threshold files
```

### Manifest schema

Each model is defined by a JSON manifest in
`tests/e2e/models/<family>/manifests/<model-name>.json`, listed from
`tests/e2e/models/<family>/MODEL.toml`:

```json
{
  "name": "qwen3-0.6b",
  "hf_id": "Qwen/Qwen3-0.6B",
  "bundle": "qwen3-0.6b.trtfb",
  "family": "qwen",
  "runtime_strategy": "decoder_kv_cache",
  "max_cache_length": 256,
  "prompt": "The capital of France is",
  "max_new_tokens": 20,
  "logit_atol": 1e-3,
  "trust_remote_code": false
}
```

**Required fields**: `name` (always); `hf_id` and `family` (when not skipped).

**Type-checked fields**: `max_new_tokens` and `max_cache_length` must be `int`.

**Schema validation**: `manifest_loader._validate_manifest()` runs automatically on load. Unknown `runtime_strategy` values emit a warning.

**Optional**: `skip` (string reason to skip), `threshold_overrides` (per-metric), `test_image` (VL), diffusion-specific fields.

### Comparator diagnostics

Every `CompareResult` returned by any comparator includes:
- `passed` (bool) -- overall pass/fail
- `metrics` (dict) -- raw metric values
- `per_metric_pass` (dict) -- per-metric bool (which individual metrics passed)
- `gate_details` (list of str) -- human-readable explanation of each gate decision
- `message` (str) -- summary including full traceback on exception

### Error diagnostics

- **Full stderr**: When a subprocess fails, `save_full_stderr()` writes the
  complete stderr to `{artifacts_dir}/{case}_{stage}_stderr.log` and includes
  the path in the error message. The inline message shows only the last 2000
  chars.
- **Full tracebacks**: Exception blocks in the orchestrator capture
  `traceback.format_exc()` and include it in the `CompareResult.message`.

### Model manifests by category

| Category | Count | Models |
|----------|:--:|---------|
| Standard decoder | 32 | Qwen3, LLaMA, Mistral, Phi, GPT-2, OPT, Bloom, Nemotron, OLMo, etc. |
| Encoder-only | 18 | BERT, ALBERT, DeBERTa, DistilBERT, ELECTRA, ModernBERT, RoBERTa, XLNet, ConvBERT, FNet, etc. |
| Vision-language | 4+1 skip | Qwen2.5-VL, Qwen3-VL, InternVL3, Phi4-multimodal (+DeepSeek-OCR skip) |
| Diffusion (T2V/T2I) | 4 | Wan2.1-T2V, FLUX.1-schnell, FLUX-2-dev, Z-Image-Turbo, PixArt-Sigma |
| Speech-to-text | 4 | Whisper-tiny, Whisper-tiny-fp16, Whisper-large-v3-turbo, Canary-1b-v2 |
| MoE decoder | 3 | Mixtral, GPT-OSS, Qwen3-MoE |
| Text-to-audio | 3 | Bark-small, Bark-large, Magpie-TTS |
| Seq2seq / translation | 3 | T5-small, BART-base, Marian-en-ru (+NLLB skip) |
| Hybrid (Mamba+Attention) | 2 | Nemotron-H, Qwen3.5 |
| Embedding | 2 | Eagle-embed, Nemotron-embed |
| SSM / RWKV | 2 | Mamba, RWKV |
| Speech-to-speech | 1 | PersonaPlex |
| Segmentation | 1 | SegFormer |
| Prompted segmentation | 1 | SAM |
| Reranking | 1 | Eagle-rerank |

---

## Layer 6: Diff Framework

**Directory**: `tools/diff.py` + `tools/diff_framework/`

**Intent**: Ad-hoc GPU-accelerated TRT-vs-HF comparison for development and
debugging. Auto-detects `runtime_strategy` from HF config or bundle header
and runs applicable checks.

**How to run**:

```bash
# List all registered checks
python tools/diff.py list

# Run all applicable checks for a model
python tools/diff.py run --model Qwen/Qwen3-0.6B

# Specific checks with a bundle
python tools/diff.py run --model Qwen/Qwen3-0.6B \
  --bundle qwen3.trtfb --binary ./build/trtmc \
  --test logit_diff --test runner_parity

# VL model with test image
python tools/diff.py run --model Qwen/Qwen2.5-VL-3B-Instruct \
  --bundle qwen25vl.trtfb --image test.jpg --binary ./build/trtmc
```

### 6 registered checks

| Check | Strategies | Bundle? | What it validates |
|-------|-----------|:--:|---|
| `logit_diff` | decoder_kv_cache, decoder_moe, mamba_ssm_recurrent | No | Per-step logit comparison (4-prompt battery) |
| `layer_diff` | decoder_kv_cache, decoder_moe | No | Per-layer hidden state comparison (debug engine) |
| `runner_parity` | decoder_kv_cache, decoder_moe, mamba_ssm_recurrent | Yes | Python TrtRunner vs C++ binary (token-for-token) |
| `vl_pipeline` | vision_language | Yes | 4-stage VL: vision features, embed_input, generation, C++ parity |
| `perf_benchmark` | decoder_kv_cache, decoder_moe, mamba_ssm_recurrent | No | TRT vs HF latency/throughput |
| `diffusion_components` | diffusion | Yes | 9-step component comparison |

---

## Regression Tiers

Standard regression gate before merging changes. Run in order; each tier
catches progressively harder issues.

### Tier 1: Unit tests (no GPU, ~10 min)

Fast, deterministic tests for logic correctness. Always run first.

```bash
# Python builder unit tests
.venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Tools self-tests
.venv/bin/python -m pytest tests/tools/ -v

# C++ unit tests
ctest --test-dir build --output-on-failure
```

**What's covered**:
- Python: 98 test modules -- config, checkpoint_mapper, bundle_writer, family plugins, per-family engine tests, build-engine integration, manifest validation, debug runner, cache state machine, quantization
- Tools: 63 modules -- diff framework, logits, layers, VL, audio, segmentation, diffusion helpers, perf_compare, coverage map, report generation, E2E harness alignment
- C++: 94 test executables -- bundle format, tokenizers (vocab, BPE, WordPiece, unigram, IPA), CUDA RAII, KV cache, TRT module, pipelines (text gen, recurrent, VL, encoder, audio, diffusion, perception), image preprocessor, CLI args

### Tier 1.5: C++ Cyclomatic Complexity Gate (no GPU, under 1 min)

Cyclomatic complexity is measured with `lizard`, which is baked into the
repository Docker image (`Dockerfile`) and verified in
`scripts/bootstrap_workspace.sh`.

Use the repository checker:

```bash
# Report-only scan for C++ runtime sources
python tools/check_cyclomatic_complexity.py src

# Gate: fail if any function is above CCN 10
python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

CI gate job: `Cyclomatic complexity` in GitHub Actions.
Threshold can be tuned via:
- `CCM_MAX_CCN`

Current policy and status:
- CI default is strict: fail on any function with `CCN > 10`.
- As of March 4, 2026, repository scan reports `CCN max: 9` and `CCN >= 10: 0`.

### Tier 2: Graph-op GPU tests (~2 min, needs TRT)

```bash
.venv/bin/python -m pytest tests/builder/test_graph_ops.py \
  tests/builder/test_graph_ops_extended.py \
  tests/builder/test_graph_blocks.py -v -m trt
```

### Tier 3: E2E single-model smoke test (~5 min, needs GPU)

```bash
.venv/bin/python -m pytest tests/e2e/models/qwen --e2e-model qwen3-0.6b -v \
  --engine-dir /mnt/storage/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
  --model-plugin-dir ./build/models --rebuild-engines
```

### Tier 4: Full E2E suite (~2-3 hours, needs GPU)

All 197 models, force-rebuild every bundle. Gold-standard regression gate.

```bash
.venv/bin/python -m pytest tests/e2e/models -v \
  --engine-dir /mnt/storage/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
  --model-plugin-dir ./build/models --rebuild-engines \
  --e2e-artifacts-dir /tmp/e2e_artifacts
```

### Tier 5: Performance regression (~10 min per model)

```bash
python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /mnt/storage/tensorrt-model-connect/engines/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" --max-new-tokens 20 --json results.json
```

### What to Run When

| Change type | Tiers to run |
|-------------|-------------|
| Python builder logic | 1, 2 |
| Family plugin | 1 (includes per-family engine tests), 2, 3 (the specific model) |
| C++ runtime | 1 (ctest), 3 |
| Graph ops / graph blocks | 1, 2 |
| KV cache / mask / position logic | 1 (ctest + cache_state_machine), 3, 4 |
| debug_runner.py | 1 (debug_runner_extended), 3 |
| Image preprocessor | 1 (ctest test_image_preprocessor) |
| Tokenizer (vocab or HF) | 1 (ctest test_vocab_tokenizer / test_hf_python_tokenizer) |
| perf_compare.py | 1 (tools tests), 5 |
| Diff tools (audio/seg/diffusion) | 1 (tools tests) |
| Vision encoder / VL pipeline | 1 (vision_compute_extended), 2, 3 |
| New model family | 1, 2, `validate_family.sh`, then add manifest + tier 4 |
| New model (existing family) | Add JSON manifest, run tier 3 with that model |
| E2E harness (runners/comparators) | Tier 3 or 4 (run affected models) |
| Manifest loader changes | 1 (test_manifest_validation.py), tier 3 |

---

## Pytest Markers Reference

Registered in `pyproject.toml`:

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests -- no GPU, no TRT |
| `gpu` | Requires NVIDIA GPU |
| `trt` | Requires TensorRT |
| `slow` | Slow tests (>30s) |
| `e2e` | End-to-end tests |
| `text` | Text generation models |
| `vision` | Vision/VL models |
| `audio` | Audio models (Whisper, Bark, PersonaPlex) |
| `diffusion` | Diffusion models (Wan, FLUX, Z-Image) |

Usage:

```bash
# Run only unit tests (fast, no GPU)
pytest tests/builder/ -m unit -v

# Exclude GPU tests
pytest tests/builder/ -m "not gpu" -v

# Only TRT graph tests
pytest tests/builder/ -m trt -v
```

---

## Coverage

Coverage is configured in `pyproject.toml` and enforced by dedicated scripts.

```bash
# Python gates (line and branch coverage must be 100%)
tools/coverage/python_coverage.sh -v --ignore=tests/builder/test_cli.py

# C++ gate (line/function/branch coverage must each be 100%)
tools/coverage/cpp_coverage.sh

# Combined local run
tools/coverage/run_coverage_all.sh
```

Configuration:
- **Source**: `python/tensorrt_model_connect` (the build package)
- **Omit**: `*/tests/*`, `*/__pycache__/*`
- **Excluded lines**: `pragma: no cover`, `if __name__ == "__main__"`, `raise NotImplementedError`

CI integration (GitHub Actions):
- `coverage-python` runs `tools/coverage_ci/run_python_coverage.sh`
  - Emits `coverage/python-cobertura.xml`
  - Enforces line=100% and branch=100%
  - Uploads Cobertura artifacts
- `coverage-cpp` runs `tools/coverage_ci/run_cpp_coverage.sh`
  - Emits `coverage/cpp-cobertura.xml`
  - Uploads Cobertura artifacts
- Both jobs are hard gates in the test DAG before smoke/E2E jobs.

Note:
- Python function coverage is not natively reported by `coverage.py`; Python gates therefore enforce line + branch.
- C++ gates enforce line + function + branch via `gcovr`.
- Coverage setup and local reproduction commands now live in the website reference pages.

---

## `validate_family.sh` Workflow

The primary validation gate for new model families:

```
validate_family.sh <hf-repo-or-path> [options]
  |
  +-- Step 1: Build bundle
  |     ./build/trtmc build <model> -o /tmp/<name>.trtfb --max-cache-length 256
  |
  +-- Step 2: diff_logits battery
  |     python tools/diff_logits.py --model <model> --atol 1e-3 --battery
  |
  +-- Step 3: diff_layers
  |     python tools/diff_layers.py --model <model> --atol 0.05
  |
  +-- Step 4: runner_parity (if binary exists)
        python tools/test_runner_parity.py --bundle /tmp/<name>.trtfb \
          --binary ./build/trtmc --hf-python .venv/bin/python --max-new-tokens 20
```

All 4 steps must pass. Step 4 is skipped if `./build/trtmc` is not found.

---

## Adding a Model to the Test Suite

1. **Run `validate_family.sh`** to confirm the model builds and passes diff checks.

2. **Create a manifest JSON** in `tests/e2e/models/<family>/manifests/<model-name>.json` and list it in `tests/e2e/models/<family>/MODEL.toml`.

3. **Run Tier 3 smoke test**:
   ```bash
   .venv/bin/python -m pytest tests/e2e/models/<family> --e2e-model my-model -v \
     --engine-dir /mnt/storage/tensorrt-model-connect/engines \
     --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
     --model-plugin-dir ./build/models --rebuild-engines
   ```

4. **Run Tier 4 full suite** to confirm no regressions.

---

## Text Generation Comparator: Composite Gating

The text comparator (`tests/e2e_harness/comparators/text.py`) uses 6 metrics
with composite gating. Understanding the gating logic is important for
diagnosing test failures.

### Metrics

| Metric | Compares | Pass criterion |
|--------|----------|----------------|
| `logit_cosine_p5` | TRT debug runner logits vs HF logits | at least 0.99 (5th percentile) |
| `logit_rel_l2_p95` | TRT debug runner logits vs HF logits | at most 0.05 (95th percentile) |
| `stable_top1_match_rate` | Argmax agreement on confident tokens | at least 0.9 |
| `unstable_topk_hit_rate` | Top-k overlap on ambiguous tokens | at least 0.8 |
| `token_agreement_rate` | Raw argmax token-for-token match | at least 0.8 |
| `normalized_text_edit_distance` | C++ binary text vs HF text | at most 0.2 |

### Composite gating rule

```
passed = logit_quality_ok AND token_level_ok AND text_ok
```

Where:
- `logit_quality_ok` = cosine_p5 passes OR rel_l2_p95 passes
- `token_level_ok` = agreement passes OR (stable_top1 passes AND unstable_topk passes)
- `text_ok` = NED passes (with two adjustments below)

### Prompt-echo stripping

The C++ binary outputs `prompt + generation` while HF returns only
`generation`. Before computing NED, the comparator strips the prompt prefix
from the C++ text (using the known prompt from `trt.data["prompt"]`). This
prevents inflated NED from the echo.

### NED hard-fail threshold

When NED >= 0.65, the test fails **regardless** of logit/token metrics. This
catches genuinely broken C++ text (repetition loops, empty output, chat
template bugs) that would otherwise be masked by good debug runner logits.

For NED < 0.65 with good token metrics, the NED failure is treated as
acceptable divergence (minor sampling differences, max_new_tokens budget
differences, etc.).

### Why logits can match but text differs

The comparator compares **debug runner logits** (Python TRT) against **HF logits**.
Both use the same tokenization (`tokenizer.encode(prompt)` with defaults).
The C++ binary uses a separate tokenizer path (`hf_python_tokenizer.py`).
If the C++ tokenizer uses different `add_special_tokens` settings, the input
tokens differ, causing completely different generated text despite "perfect"
logit match. This is why the `tokenizer_add_special_tokens` bundle field
exists — see [Architecture Overview](Architecture-Overview.md#53-self-describing-config).

---

## Accuracy Tolerances Reference

| Model | Strategy | `logit_atol` | `layer_atol` | Rationale |
|-------|----------|:--:|:--:|---|
| Most standard decoders | `decoder_kv_cache` | `1e-3` | `0.05` | Baseline FP32 precision |
| mamba-130m | `mamba_ssm_recurrent` | `2e-3` | -- | Recurrent state drift |
| mixtral-stories-15m | `decoder_moe` | `2e-3` | `0.05` | Expert routing precision |
| phi-moe | `decoder_kv_cache` | `1e-3` | `0.05` | SparseMixer is deterministic |
| VL models | `vision_language` | `1e-3` | `0.05` | Vision features `atol=0.1` |
| wan21-t2v-1.3b | `diffusion` | -- | -- | Cosine sim (0.95 single step, 0.8 full) |

**Why some models need looser tolerances**:
- **Distilled models** (minitron): Pruning amplifies kernel differences
- **Recurrent models** (mamba): FP32 drift accumulates across recurrence
- **MoE models** (mixtral): Expert routing softmax is precision-sensitive
- **Cross-lingual** (xglm): Multi-language vocabulary increases variance
