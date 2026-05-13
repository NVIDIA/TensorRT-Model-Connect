# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

C++ bundle-only runtime for TensorRT inference, paired with a Python build package (`tensorrt_model_connect/`) that converts HuggingFace models into `.trtfb` bundles. The Python package handles model loading, TRT engine building, and bundle packaging. The C++ runtime loads `.trtfb` bundles, deserializes TRT engines, and runs autoregressive inference. Public C++ API lives in `include/trtmc/`. Everything is in the `trtmc` namespace. C++17, compiled with `-Wall -Wextra -Wpedantic`.

## Workspace isolation

Use isolated repo clones and containers per team on shared GB300 hosts.

- **Repo clone per team**: `/workspace/users/yifeif/workspaces/<id>/tensorrt-model-connect`
- **Container per team**: `trtmc-dev-gb300-<id>` (do not use shared `trtmc-dev-gb300`)
- **Shared read-only resources**: HF cache (`/mnt/storage/tensorrt-model-connect/model-weights`), engines (`/workspace/users/yifeif/tensorrt-model-connect/engines`)
- **Isolated per team**: git state, `build/`, editable installs, branches

Bootstrap a new isolated workspace:
```bash
./scripts/bootstrap_workspace.sh --id <team-id> --branch <branch> --detach
```

Run commands in your team container:
```bash
docker exec trtmc-dev-gb300-<team-id> <command>
```

Rely on CI (GitLab pipeline) as the quality gate — push your branch and let CI validate. Do NOT run the full E2E suite locally unless specifically asked.

## Build commands

ALWAYS DO EVERYTHING IN CONTAINER. Run command with `docker exec trtmc-dev-gb300-<your-team-id>`

### C++ runtime

Use the container-baked TRT/CUDA paths:
```bash
cmake -S . -B build -G Ninja \
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

With TRT-RTX backend DSO (optional, needs TRT-RTX SDK):
```bash
cmake -S . -B build -G Ninja \
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so \
  -DTRTMC_BUILD_BACKEND_RTX=ON \
  -DTRTMC_RTX_LIBRARY_DIR=<path-to-libtensorrt_rtx> \
  -DTRTMC_RTX_INCLUDE_DIR=<path-to-TensorRT-RTX-SDK/include>
cmake --build build -j
```

### Python build package

```bash
pip install --no-deps -e tensorrt_model_connect/

# Build from HF repo ID (auto-downloads) or local directory
trtmc-build build Qwen/Qwen3-0.6B -o <output.trtfb> [--max-cache-length N] [--verbose]
trtmc-build build <model-dir> -o <output.trtfb> [--max-cache-length N] [--verbose]

# Build for TRT-RTX backend (sets engine_backend="trt_rtx" in bundle metadata)
trtmc-build build Qwen/Qwen3-0.6B -o <output.trtfb> [--rtx]

# Or use the Python API
python3 -c "import tensorrt_model_connect; tensorrt_model_connect.build('Qwen/Qwen3-0.6B', 'qwen3.trtfb')"

trtmc-build inspect <bundle.trtfb>
trtmc-build version
```

## Running tests

C++ runtime unit tests:
```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R test_pipeline --output-on-failure
```

Python builder unit tests:
```bash
pytest tests/builder/ -v
```

Diff framework self-tests:
```bash
pytest tests/tools/ -v
```

Cyclomatic complexity gate (required for C++ runtime changes):
```bash
# Report
python tools/check_cyclomatic_complexity.py src

# Gate (must pass before merge; CI uses the same default threshold)
python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

Unified E2E tests (requires GPU + engine bundles):
```bash
# All models (68 models — auto-builds missing bundles):
pytest tests/test_e2e.py -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# Single model:
pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# Force rebuild all bundles from HF:
pytest tests/test_e2e.py -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python --rebuild-engines

# Filter by task strategy:
pytest tests/test_e2e.py -v --e2e-task-strategy text_generation_causal \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# With artifact output (WAV, PNG, frames, logits):
pytest tests/test_e2e.py -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --e2e-artifacts-dir /tmp/e2e_artifacts
```

All Python tests at once:
```bash
pytest tests/ -v --ignore=tests/cpp
```

C++ tests are plain executables (no framework) in `tests/cpp/`. They use `main()`, print to stderr on failure, and return 0 on success / non-zero on failure. Test names in CMake match their source files.

## Traceability requirements

Every new or modified test must document:
- Intent
- Preconditions
- Postconditions

And must include trace IDs linking to architecture and design contracts (`ARCH-*`, `UD-*`, `UT-*`/`IT-*`).

Reference:
- `website/docs/wiki/Traceability-Matrix.md` (bi-directional architecture -> unit design -> test process)
- `website/docs/wiki/Testing-and-Validation.md` (test intent field requirements and examples)

## Test architecture

Tests are organized in four layers, each with different scope, dependencies,
and intended use.

### Layer 1: Python builder unit tests (`tests/builder/`)

**Intent:** Verify Python build logic in isolation — config parsing, weight
loading, checkpoint mapping, bundle writing, engine orchestration, family
plugin dispatch, graph ops, and debug runner infrastructure.

**Implementation:**
- Tests use `pytest` with shared fixtures from `tests/builder/conftest.py`.
- Two skip markers: `requires_trt` (TRT + CUDA available) and
  `requires_tensorrt_model_connect` (tensorrt_model_connect importable). All files use
  `try/except` with `pytest.skip(allow_module_level=True)` so they skip
  cleanly when TRT is not installed.
- TRT graph tests use the `trt_runner` fixture: a `build_fn(network, inputs)`
  closure constructs a small TRT graph, the fixture builds an engine, runs
  inference, and returns NumPy outputs for comparison against PyTorch/NumPy
  references via `np.testing.assert_allclose`.
- Mock-based tests (engine_builder, pipeline, debug_runner cleanup) use
  `unittest.mock.patch` and require no GPU.
- Family plugin tests create synthetic safetensors with correct weight shapes,
  call `plugin.load_weights()`, and assert on returned WeightDict keys and
  family-specific transforms (e.g., Gemma gamma+1.0, Phi fused QKV split).

**Key files:**
| File | Tests | GPU? |
|------|-------|------|
| `test_config.py` | ModelConfig parsing, VL text_config merge, edge cases | No |
| `test_checkpoint_mapper.py` | Weight loading, GQA expansion, tied embeddings, biases | No |
| `test_family_plugins.py` | 10 family plugins: load_weights correctness | No |
| `test_families.py` | Plugin match/dispatch, runtime_strategy, embed_input | No |
| `test_graph_ops.py` | 18 graph ops (RoPE, ALiBi, RMSNorm, attention, etc.) | Mix |
| `test_graph_ops_extended.py` | YaRN RoPE, T5 bias, extended ALiBi, conv/norm/ELU/pad ops | Mix |
| `test_graph_blocks.py` | apply_norm, SwiGLU MLP, GELU FC MLP blocks | TRT |
| `test_engine_builder_extended.py` | build_bundle orchestration, GPU name, TRT version | Mock |
| `test_pipeline.py` | Pipeline subprocess wrapper, binary detection | Mock |
| `test_debug_runner_extended.py` | Bundle section loading, runner cleanup, generate sequencing | Mix |
| `test_vision_compute_extended.py` | Vision RoPE, DeepStack config, patch embed, spatial merge | Mix |
| `test_standard_decoder.py` | Tensor naming contract, debug outputs | TRT |
| `test_bundle_writer.py` | Bundle format round-trip, section integrity | No |
| `test_cache_state_machine.py` | Position, mask, cache append/shift logic | No |
| `test_cli.py` | CLI inspect/build command dispatch | Mock |

### Layer 2: C++ runtime unit tests (`tests/cpp/`)

**Intent:** Verify C++ runtime correctness — bundle parsing, tokenizers,
CUDA RAII wrappers, KV cache device operations, TRT engine lifecycle, image
preprocessing, CLI argument parsing, and helper utilities.

**Implementation:**
- Plain `main()` executables with no test framework. A `check(condition, name)`
  helper accumulates `failures`; `main()` returns non-zero if any failed.
- Registered in `CMakeLists.txt` with `add_executable` + `add_test`.
- TRT-dependent tests guard with `#if TRTMC_HAS_TRT` and skip gracefully
  (exit 0) when TRT headers are unavailable.
- Shared utilities in `test_helpers.h`: temp dirs, safetensors writing,
  standard decoder checkpoint generation.

**Key files:**
| File | Tests | GPU? |
|------|-------|------|
| `test_bundle_format.cpp` | Bundle magic, section parsing, round-trip | No |
| `test_vocab_tokenizer.cpp` | Encode/decode, round-trip, case insensitivity | No |
| `test_hf_python_tokenizer.cpp` | Shell quoting, int parsing, HF output sanitization | No |
| `test_text_parsers.cpp` | String/file parsing helpers | No |
| `test_json_helpers.cpp` | JSON extraction helpers | No |
| `test_cli_args.cpp` | CLI argument parsing | No |
| `test_data_dir.cpp` | Source/scripts dir resolution, config override | No |
| `test_trt_logger.cpp` | Severity names, error storage, explicit config controls | TRT headers |
| `test_trt_engine_lifecycle.cpp` | layer_tensor_name, constants | TRT headers |
| `test_bundle_helpers.cpp` | find_bundle_sections for all bundle types | TRT headers |
| `test_image_preprocessor.cpp` | All 4 strategies, config parsing, prompt formatting | No |
| `test_cuda_buffer.cpp` | RAII alloc, move semantics, data round-trip | GPU |
| `test_cuda_stream.cpp` | RAII stream, move semantics | GPU |
| `test_device_kv_cache.cpp` | Cache construction, mask progression, position clamping, reset | GPU |
| `test_decode_runtime.cpp` | Argmax, mask building | TRT headers |
| `test_fast_path_config.cpp` | Config JSON parsing | TRT headers |
| `test_pipeline_api.cpp` | C API pipeline creation | TRT headers |
| `test_bundle_e2e.cpp` | Bundle build + load round-trip | TRT headers |
| `test_c_abi_entry.cpp` | C ABI entry point | TRT headers |

### Layer 3: Tool self-tests (`tests/tools/`)

**Intent:** Verify diff framework and comparison utilities in isolation —
logit comparison, layer diffing, perf benchmarking, audio/segmentation
metrics — without needing models or GPU.

**Implementation:**
- Pure Python tests, no TRT/GPU needed. `conftest.py` adds `tools/` to path.
- Tests use `importlib.import_module` for lazy importing of tool modules.
- Comparison logic tested with synthetic NumPy arrays and known expected values.
- WAV round-trip tests create real audio files via `write_wav_f32`/`read_wav_f32`.
- Bundle config/weight loading tests create synthetic `.trtfb` bundles in memory.

**Key files:**
| File | Tests |
|------|-------|
| `test_tool_helpers.py` | cosine_sim, compare_arrays |
| `test_diff_audio.py` | Energy computation, WAV I/O, token stats |
| `test_diff_segmentation.py` | Pixel agreement, logit diff, argument parsing |
| `test_diffusion_helpers.py` | silu, gelu_tanh, bundle config/weights, timestep embedding |
| `test_diff_logits.py` | Logit comparison, argmax match, top-k overlap |
| `test_diff_framework.py` | DiffResult, registry, runner, CLI parsing |
| `test_parity.py` | Text/token comparison for runner parity |
| `test_perf_compare.py` | Stats, formatting, JSON output, serial GPU execution |

### Layer 4: Unified E2E tests (`tests/test_e2e.py` + `tests/e2e_harness/`)

**Intent:** Validate the full pipeline end-to-end — build bundle from HF,
run C++ inference, compare output against HuggingFace reference. This is
the gold-standard correctness gate. All modalities use the same harness.

**Architecture (DIP-first):**
- `tests/test_e2e.py` — single parametrized pytest entrypoint, one test
  per model manifest. Resolves paths, builds `RunContext`, invokes the
  orchestrator.
- `tests/e2e_harness/orchestrator.py` — coordinates the full lifecycle:
  preflight → bundle resolve/build → per-stage TRT run → reference run →
  comparison → artifact persistence. Depends only on abstract contracts.
- `tests/e2e_harness/contracts.py` — dataclasses and protocols
  (`E2ECase`, `StageOutput`, `CompareResult`, `TaskStrategyRunner`,
  `ReferenceBackendRunner`, `Comparator`, `ArtifactSink`).
- `tests/e2e_harness/registry.py` — auto-discovers plugins from
  `runners/`, `references/`, `comparators/` via module-level `plugin`
  attributes.
- `tests/e2e_harness/manifest_loader.py` — loads per-model JSON manifests,
  infers `task_strategy` from `runtime_strategy`, builds default stages
  and preflight requirements.

**Strategy runners** (`tests/e2e_harness/runners/`):
Each runner handles one `task_strategy` and executes TRT inference via
subprocess (C++ binary or Python debug runner).

| Runner | Task Strategy | Models |
|--------|--------------|--------|
| `text_generation.py` | text_generation_causal | All decoders, MoE, SSM, RWKV |
| `vision_language.py` | vision_language_generation | Qwen2.5-VL, Qwen3-VL, InternVL3 |
| `audio_speech.py` | speech_to_text, text_to_audio, speech_to_speech | Whisper, Bark, PersonaPlex |
| `diffusion.py` | diffusion_media_generation | Wan2.1-T2V, FLUX, Z-Image |
| `segmentation.py` | segmentation, prompted_segmentation | SegFormer, SAM |
| `embedding.py` | embedding | Eagle-embed |
| `reranking.py` | reranking | Eagle-rerank |
| `encoder_only.py` | encoder_only_nlp | BERT |

**Reference backends** (`tests/e2e_harness/references/`):
| Backend | Used by |
|---------|---------|
| `hf_transformers.py` | Text gen, VL, audio, segmentation, embedding, reranking |
| `hf_diffusers.py` | Diffusion (Wan, FLUX, Z-Image) |
| `torch_reference.py` | Speech-to-speech |

**Comparators** (`tests/e2e_harness/comparators/`):
Each comparator computes modality-specific metrics and applies
threshold-based gating.

| Comparator | Key Metrics |
|-----------|-------------|
| `text.py` | logit_cosine_p5, stable_top1_match, token_agreement, NED (composite rule) |
| `vision_language.py` | vision_embedding_cosine, NED, word agreement (composite) |
| `text_to_audio.py` | RMS bounds, duration ratio, mel spectrogram distance |
| `diffusion.py` | pixel mean/std range, temporal consistency, PSNR, SSIM |
| `segmentation.py` | mIoU, pixel accuracy, boundary F-score |
| `speech_to_text.py` | Transcript text similarity |

**Thresholds:** Defaults in `tests/e2e_harness/thresholds/defaults/*.json`,
per-model overrides via manifest `threshold_overrides` or inline fields
(`logit_atol`, `layer_atol`, etc.).

**Rich artifacts:** All runners persist human-inspectable artifacts to
`--e2e-artifacts-dir`: WAV audio, PNG frames/images, transcript text,
logits `.npy`, colorized segmentation maps.

**Model manifests (68 models)** in `tests/e2e/models/*.json`:

| Category | Count | Examples |
|----------|-------|---------|
| Standard decoder | 24 | Qwen3, LLaMA, Mistral, Phi, GPT-2, OPT, Bloom, Nemotron |
| MoE decoder | 3 | Mixtral, Phi-MoE, Qwen3-MoE |
| SSM | 2 | Mamba, RWKV |
| Encoder-only | 1 | BERT |
| Speech-to-text | 1 | Whisper |
| Text-to-audio | 2 | Bark-small, Bark-large |
| Speech-to-speech | 1 | PersonaPlex |
| Segmentation | 1 | SegFormer |
| Prompted segmentation | 1 | SAM |
| Vision-language | 3+2 skip | Qwen2.5-VL, Qwen3-VL, InternVL3 (+Phi4, InternLM2 skip) |
| Text-to-video diffusion | 1 | Wan2.1-T2V |
| Text-to-image diffusion | 2 | FLUX.1-schnell, Z-Image-Turbo |
| Embedding/Reranking | 2 skip | Eagle-embed, Eagle-rerank (repos 404) |
| Hybrid | 1 skip | Nemotron-H (needs mamba-ssm) |

**Manifest schema:**
```json
{
  "name": "model-name",
  "hf_id": "HuggingFace/Model-ID",
  "bundle": "model.trtfb",
  "family": "family_name",
  "runtime_strategy": "decoder_kv_cache",
  "max_cache_length": 256,
  "prompt": "Test prompt",
  "max_new_tokens": 20,
  "logit_atol": 1e-3,
  "trust_remote_code": false,
  "skip": "optional reason to skip this model"
}
```

The `runtime_strategy` auto-maps to a `task_strategy` which selects the
runner, comparator, reference, default stages, and default thresholds.
Adding a new model requires only a JSON manifest — no code changes.

## Regression test plan

Standard regression gate before merging changes. Run in order; each tier
catches progressively harder issues. All E2E tests use the unified harness
(`tests/test_e2e.py`).

### Tier 1: Unit tests (no GPU, ~60s)

Fast, deterministic tests for logic correctness. Always run first.

```bash
# Python builder unit tests (config, checkpoint_mapper, bundle_writer, family plugins, etc.)
/opt/venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Tools self-tests (diff framework, perf_compare, audio/segmentation/diffusion helpers)
/opt/venv/bin/python -m pytest tests/tools/ -v

# C++ unit tests (bundle format, tokenizers, CUDA wrappers, KV cache, image preprocessor)
ctest --test-dir build --output-on-failure
```

### Tier 2: Graph-op GPU tests (~2 min, needs TRT)

Validates TRT graph operations (RMSNorm, RoPE, attention, conv, norm, etc.) and
composable graph blocks (SwiGLU MLP, GELU MLP, attention block) on real GPU.

```bash
/opt/venv/bin/python -m pytest tests/builder/test_graph_ops.py tests/builder/test_graph_ops_extended.py tests/builder/test_graph_blocks.py -v -m trt
```

### Tier 3: E2E single-model smoke test (~5 min, needs GPU)

Quick sanity with one small model. Always use `--rebuild-engines` to build
the bundle from scratch — avoids testing against stale cached bundles.

```bash
/opt/venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --rebuild-engines
```

### Tier 4: Full E2E suite (~2-3 hours, needs GPU)

All 68 models via the unified harness. Force-rebuild every bundle, then
infer/compare. This is the gold-standard regression gate.

```bash
# All models:
/opt/venv/bin/python -m pytest tests/test_e2e.py -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --rebuild-engines --e2e-artifacts-dir /tmp/e2e_artifacts

# By modality (faster targeted runs):
/opt/venv/bin/python -m pytest tests/test_e2e.py -v \
  --e2e-task-strategy text_generation_causal \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# Available task strategies for filtering:
#   text_generation_causal    (26 models — decoders, MoE, SSM, RWKV)
#   vision_language_generation (5 models — Qwen VL, InternVL, Phi4)
#   diffusion_media_generation (3 models — Wan T2V, FLUX, Z-Image)
#   text_to_audio             (2 models — Bark)
#   speech_to_text            (1 model — Whisper)
#   speech_to_speech          (1 model — PersonaPlex)
#   segmentation              (1 model — SegFormer)
#   prompted_segmentation     (1 model — SAM)
#   encoder_only_nlp          (1 model — BERT)
#   embedding                 (1 model — Eagle-embed, currently skipped)
#   reranking                 (1 model — Eagle-rerank, currently skipped)
```

### Tier 5: Performance regression (~10 min per model, needs GPU + bundle)

Spot-check inference speed for key models. Not in CI; run manually for
perf-sensitive changes.

```bash
python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /workspace/users/yifeif/tensorrt-model-connect/engines/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" --max-new-tokens 20 --json results.json
```

### What to run when

| Change type | Tiers to run |
|-------------|-------------|
| Python builder logic | 1, 2 |
| Family plugin | 1 (includes plugin load_weights tests), 2, 3 (the specific model) |
| C++ runtime | 1 (ctest — includes CUDA, KV cache, tokenizer tests), 3 |
| Graph ops / graph blocks | 1, 2 |
| KV cache / mask / position logic | 1 (ctest test_device_kv_cache + Python cache_state_machine), 3, 4 |
| debug_runner.py | 1 (debug_runner_extended), 3 |
| Image preprocessor | 1 (ctest test_image_preprocessor) |
| Tokenizer (vocab or HF) | 1 (ctest test_vocab_tokenizer / test_hf_python_tokenizer) |
| perf_compare.py | 1 (tools tests), 5 |
| Diff tools (audio/seg/diffusion) | 1 (tools tests) |
| Vision encoder / VL pipeline | 1 (vision_compute_extended), 2, 3 |
| New model family | 1, 2, validate_family.sh, then add manifest to tests/e2e/models/ + tier 4 |
| New model (existing family) | Add JSON manifest to tests/e2e/models/, run tier 3 with that model |
| E2E harness (runners/comparators) | Tier 3 or 4 (run affected models) |

## Running executables

```bash
./build/trtmc run     <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--hf-python PATH]
./build/trtmc run     <bundle.trtfb> --prompt "text" --image <image.jpg> [--max-new-tokens N] [--hf-python PATH]
./build/trtmc run     <bundle.trtfb> --prompt "text" [--runtime-cache PATH] [--cuda-graphs]
./build/trtmc inspect <bundle.trtfb>
./build/trtmc version
```

For vision-language models, pass `--image` with the path to an image file. The VL bundle must have been built from a VL model (e.g. Qwen2.5-VL) and contains both a text decoder and vision encoder engine.

For TRT-RTX bundles, `--runtime-cache PATH` persists JIT compilation results to disk (speeds up subsequent runs), and `--cuda-graphs` enables CUDA graph capture/replay for reduced kernel launch overhead.
If backend DSOs are not next to the `trtmc` executable or visible through the
platform loader path, pass `--backend-dir PATH` explicitly.

## Runtime configuration

Runtime knobs flow through bundle metadata and the generic config surface:
`--config <file>` and repeated `--set namespace.field=value` overrides.
TensorRT logger controls live under `platform.trt_log_stderr` and
`platform.trt_log_min_severity`.

## Architecture

The system is split into two stages:

1. **Python build** (`tensorrt_model_connect/`) — takes an HF model directory (with `config.json` + safetensors), builds a TRT engine via TensorRT's Python API, and packages it into a `.trtfb` bundle. Model family plugins in `tensorrt_model_connect/tensorrt_model_connect/families/` handle family-specific weight mapping and graph construction.

2. **C++ runtime** — loads a `.trtfb` bundle, deserializes the TRT engine plan, and runs inference. The runtime is bundle-only: it does not load HF model directories directly.

**Backend abstraction (dlopen-based dispatch):** The main `trtmc` binary links only libcudart -- it does not link libnvinfer or any TRT SDK at compile time. TRT engine execution is delegated to dynamically loaded backend DSOs. At runtime, `BackendLoader::load()` reads the `engine_backend` field from the bundle's `config.json` and `dlopen`s the matching shared library:
- `libtrtmc_backend_trt.so` -- standard TRT backend, links libnvinfer.
- `libtrtmc_backend_trt_rtx.so` -- TRT-RTX backend, links libtensorrt_rtx. Adds `IRuntimeCache` (JIT compilation cache persisted to disk) and `CudaGraphStrategy` (CUDA graph capture/replay for reduced launch overhead).
- Old bundles without an `engine_backend` field default to `"trt"`.

All pipelines interact with TRT engines through `ITrtModule`, a pure virtual interface declared in `include/trtmc/runtime/trt_module.h`. The concrete implementation (`TrtModuleImpl`) lives inside the backend DSOs. Each DSO exports an `IBackend` factory that creates `ITrtModule` instances with backend-specific options (e.g., runtime cache path, CUDA graph enablement).

**Plugin registry dispatch:** The `runtime_strategy` field in the bundle's `config.json` selects the backend via `PipelineRegistry` — a singleton that maps strategy strings to `IPipelinePlugin` instances. Each plugin lives in its own `.cpp` file under `src/runtime/plugins/`; the CMake manifest generates explicit registrar calls for those plugins. The registry and factory live in `src/runtime/registry/` — `pipeline_factory.cpp` is ~124 LOC: it reads the bundle, extracts the strategy, normalizes legacy strategy strings (e.g. `"diffusion"` → `"diffusion_flux"`), looks up the plugin, and calls `plugin->create(ctx)`. No switch/case, no edits needed when adding new strategies.

**Runtime strategies (17 registered):**
| Strategy | Plugin | Pipeline |
|----------|--------|----------|
| `decoder_kv_cache` | DecoderPlugin | TextGenerationPipeline |
| `decoder_moe` | DecoderPlugin | TextGenerationPipeline |
| `ssm_recurrent` | SsmPlugin | RecurrentPipeline |
| `rwkv_recurrent` | RwkvPlugin | RecurrentPipeline |
| `hybrid_mamba_attention` | HybridPlugin | RecurrentPipeline |
| `vision_language` | VLPlugin | VLPipeline |
| `speech_to_text` | WhisperPlugin | WhisperPipeline |
| `text_to_text` | T5Plugin | T5Pipeline |
| `text_to_audio_bark` | BarkPlugin | BarkPipeline |
| `text_to_audio_magpie` | MagpiePlugin | MagpiePipeline |
| `omni_multimodal` | OmniPlugin | OmniPipeline |
| `diffusion_flux` | FluxPlugin | FluxPipeline |
| `diffusion_wan` | WanPlugin | WanPipeline |
| `diffusion_zimage` | ZImagePlugin | ZImagePipeline |
| `encoder_only_nlp` | EncoderPlugin | EncoderPipeline |
| `segmentation` | SegmentationPlugin | SegmentPipeline |
| `prompted_segmentation` | SegmentationPlugin | SamPipeline |
| `object_detection` | ObjectDetectionPlugin | EncoderPipeline |

**Legacy strategy normalization:** Old bundles may contain ambiguous strategy strings (`"text_to_audio"`, `"diffusion"`). `pipeline_factory.cpp` auto-rewrites these to their unambiguous equivalents using bundle config fields (`magpie_tts`, `diffusion_backend_type`).

**BaseConfig:** Universal bundle metadata parsed by `parse_base_config()` in `pipeline_plugin.cpp` — vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, head_dim, max_cache_length, bos/eos IDs, runtime_strategy, tokenizer flags. Each plugin parses additional strategy-specific fields from the raw config JSON. See `include/trtmc/runtime/pipeline_plugin.h` for all fields.

Tokenizer implementations (`ITokenizer`):
- `VocabTokenizer` — vocab.txt-based lookup.
- `HfPythonTokenizer` — bridges to HuggingFace tokenizers via Python subprocess. The `add_special_tokens` flag is controlled by the bundle's `tokenizer_add_special_tokens` config field (detected at build time from HF tokenizer config). When the field is absent (old bundles), defaults to `true` to match HF's `tokenizer.encode()` default.

Bundle self-describing config: The `.trtfb` bundle header contains JSON metadata that captures all build-time decisions (runtime_strategy, max_cache_length, tokenizer_add_special_tokens, etc.). The C++ runtime reads these fields to configure backends, cache sizes, and tokenizer behavior — no external configuration needed.

## Source layout

```
tensorrt_model_connect/                          # Python package (engine builder)
  tensorrt_model_connect/
    __init__.py
    __main__.py
    cli.py                           # CLI: trtmc-build build|inspect|version
    config.py                        # ModelConfig from config.json
    graph_ops.py                     # Layer 1: Atomic TRT graph ops (tensor-in/tensor-out)
    graph_blocks.py                  # Layer 2: Composable blocks (attention, SwiGLU, GELU MLP, norm)
    standard_decoder_builder.py      # Layer 3: Standard decoder engine builder (uses graph_blocks)
    checkpoint_mapper.py             # HF safetensors -> weight dict
    bundle_writer.py                 # Write .trtfb files
    engine_builder.py                # Orchestrator: load -> build -> bundle
    debug_runner.py                  # TrtRunner + MambaTrtRunner + VLTrtRunner for pure-Python inference
    qwen_vl_vision_builder.py        # Vision encoder builders: Qwen2.5-VL (3D RoPE) + Qwen3-VL (DeepStack)
    families/
      __init__.py                    # Auto-discover plugins
      base.py                        # FamilyPlugin protocol
      qwen.py llama.py mistral.py gemma.py phi.py phi_moe.py
      granite.py internlm.py starcoder2.py gpt2.py opt.py
      falcon.py stablelm.py mamba.py qwen_vl.py olmo.py nemotron.py
      xglm.py gpt_neox.py gpt_neo.py codegen.py bloom.py mixtral.py
      bert.py sam.py segformer.py whisper.py bark.py personaplex.py
      internvl.py phi4_multimodal.py eagle_vlm.py nemotron_h.py rwkv.py
      deepseek_v2.py qwen_moe.py qwen3_omni.py deepseek_ocr.py
      canary.py distilbert.py glm.py gpt_oss.py magpie_tts.py
      mpnet.py pixart.py qwen3_5.py roberta.py
      wan_t2v.py flux.py z_image.py                # Diffusion T2V/T2I
      electra.py modernbert.py deberta.py t5.py     # Autopilot-generated
  pyproject.toml
include/trtmc/runtime/                # Public C++ headers (plugin system)
  pipeline_factory.h                 # PipelineFactory::from_bundle()
  pipeline_plugin.h                  # IPipelinePlugin interface, BaseConfig, PipelineContext
  pipeline_registry.h                # PipelineRegistry singleton, manifest registration macro
  trt_module.h                       # ITrtModule pure virtual interface (forward, forward_device, bind)
  trt_backend.h                      # IBackend interface + ModuleCreateOptions
src/                                 # C++ bundle-only runtime
  bundle/
    bundle_format.h/cpp              # Read .trtfb files
  cabi/
    api/trtmc_c.cpp                   # C ABI: trtmc_create_pipeline_ex()
  runtime/
    registry/                        # Factory + plugin dispatch
      pipeline_factory.cpp           # Registry-based dispatch (~124 LOC, no switch/case)
      pipeline_plugin.cpp            # BaseConfig parsing (parse_base_config)
      pipeline_registry.cpp          # Singleton registry + generated plugin registration
    backend/                         # Backend DSO implementations (dlopen-loaded)
      backend_loader.h/cpp           # dlopen dispatch, DSO caching
      trt_module_impl.h/cpp          # TrtModuleImpl : ITrtModule (compiled into both DSOs)
      trt_backend.cpp                # Standard TRT IBackend → libtrtmc_backend_trt.so
      rtx_backend.cpp                # TRT-RTX IBackend → libtrtmc_backend_trt_rtx.so
    plugins/                         # Self-registering pipeline plugins (strategy → pipeline factories)
      shared/
        plugin_helpers.h/cpp         # ITrtModule loading via backend, tokenizer, KV helpers
        diffusion_helpers.h/cpp      # Shared diffusion config/loading
        audio_helpers.h/cpp          # Audio config, mel, speech helpers
      decoder_plugin.cpp             # decoder_kv_cache, decoder_moe
      ssm_plugin.cpp                 # ssm_recurrent
      rwkv_plugin.cpp                # rwkv_recurrent
      hybrid_plugin.cpp              # hybrid_mamba_attention
      encoder_plugin.cpp             # encoder_only_nlp, embedding, reranking
      segmentation_plugin.cpp        # segmentation, prompted_segmentation
      object_detection_plugin.cpp    # object_detection
      vl_plugin.cpp                  # vision_language
      whisper_plugin.cpp             # speech_to_text
      bark_plugin.cpp                # text_to_audio_bark
      magpie_plugin.cpp              # text_to_audio_magpie
      speech_plugin.cpp              # speech_to_speech
      omni_plugin.cpp                # omni_multimodal
      flux_plugin.cpp                # diffusion_flux
      wan_plugin.cpp                 # diffusion_wan, diffusion_pixart
      zimage_plugin.cpp              # diffusion_zimage
      t5_plugin.cpp                  # text_to_text (encoder-decoder seq2seq)
      cmake/trtmc_pipeline_plugins.cmake # Plugin source/anchor manifest
    pipelines/                       # Pipeline implementations (one class per file, fully isolated)
      text_generation_pipeline.h/cpp # TextGenerationPipeline: standard decoder + MoE
      recurrent_pipeline.h/cpp       # RecurrentPipeline: SSM, RWKV, hybrid (via IStateManager)
      vl_pipeline.h/cpp              # VLPipeline: vision-language
      encoder_pipeline.h/cpp         # EncoderPipeline: encoder-only, embedding, reranking
      segment_pipeline.h/cpp         # SegmentPipeline: SegFormer segmentation
      sam_pipeline.h/cpp             # SamPipeline: SAM prompted segmentation
      whisper_pipeline.h/cpp         # WhisperPipeline: speech-to-text
      bark_pipeline.h/cpp            # BarkPipeline: text-to-audio (Bark)
      magpie_pipeline.h/cpp          # MagpiePipeline: text-to-audio (Magpie TTS)
      speech_pipeline.h/cpp          # SpeechPipeline: speech-to-speech (PersonaPlex)
      omni_pipeline.h/cpp            # OmniPipeline: omni-multimodal
      flux_pipeline.h/cpp            # FluxPipeline: FLUX text-to-image diffusion
      wan_pipeline.h/cpp             # WanPipeline: Wan/PixArt text-to-video diffusion
      z_image_pipeline.h/cpp         # ZImagePipeline: Z-Image text-to-image diffusion
    core/                            # Foundational runtime infrastructure
      trt_common.h/cpp               # TRT logger, CUDA helpers (CudaBuffer/CudaStream with move semantics)
      trt_engine_lifecycle.h/cpp     # Engine deserialization, tensor validation
      trt_decode_runtime.h/cpp       # select_argmax_token, build_attention_mask
      trt_module.cpp                 # TrtModule: engine wrapper (forward, bind, I/O)
      device_kv_cache.h/cpp          # DeviceKvCache, DeviceResources, run_decoder_step_device
      kv_cache.cpp                   # KvCache: device-resident K/V buffer management
      recurrent_state.cpp            # RecurrentState: Mamba/RWKV recurrent state management
      device_tensor.cpp              # DeviceTensor: GPU-resident tensor reference
      flow_match_euler_scheduler.cpp # Flow-matching Euler scheduler (diffusion)
      step_state.h                   # IStepState interface
    domains/                         # Domain-specific components (grouped by modality)
      audio/                         # Audio processing (mel, whisper, bark, speech, magpie)
        mel_spectrogram.h/cpp        # Mel spectrogram extraction
        audio_types.h/cpp            # WAV I/O, audio format types
        audio_bundle_validation.h/cpp # Bundle section validation for audio models
        whisper_config.h             # Whisper model config
        bark_config.h                # Bark model config
        audio_configs.h              # Shared audio config structs
        speech_*.h                   # Speech-to-speech plans and policies
        magpie_*.h                   # Magpie TTS plans and policies
        omni_audio_plan.h            # Omni multimodal audio plan
      diffusion/                     # Diffusion model components
        diffusion_types.h            # DiffusionConfig, PreprocessorWeights
        diffusion_preprocessor.cpp   # Weight preprocessing for diffusion
        diffusion_math.h             # Diffusion math helpers
        diffusion_generation_plan.h  # Denoising loop plan
        wan_generation_conditioning.h # Wan-specific conditioning
      multimodal/                    # Vision-language components
        image_preprocessor.h/cpp     # VL image preprocessing: 4 strategies + configurable interpolation
        vision_engine.h/cpp          # Vision encoder TRT module wrapper
        vision_execution_plan.h      # Vision encoder execution plan
        vl_decode_policy.h           # VL decode policy
      perception/                    # Segmentation and detection components
        perception_types.h           # Shared types for perception models
        sam_*.h                      # SAM prompt, preprocess, postprocess seams
        segmentation_*.h             # SegFormer preprocess/postprocess seams
      recurrent/                     # Recurrent model components (Mamba/RWKV)
        recurrent_step_contracts.h   # Step contract types
        recurrent_tensor_bindings.h  # Tensor binding helpers
  tokenizer/
    vocab_tokenizer.cpp              # Word-to-id lookup from vocabulary list
    bpe_tokenizer.cpp                # BPE tokenizer (native C++)
    wordpiece_tokenizer.cpp          # WordPiece tokenizer (native C++)
    unigram_tokenizer.cpp            # Unigram tokenizer (native C++)
    ipa_tokenizer.cpp                # IPA phoneme tokenizer
  utils/
    data_dir.h/cpp                   # centralized source-dir resolution
    text_parsers.h/cpp               # shared string/file parsing (starts_with, read_file, etc.)
    json_helpers.h/cpp               # shared JSON extraction (extract_json_string, etc.)
tools/                               # Diff test framework (TRT vs HF comparison)
  diff_logits.py                     # E2E logit comparison (trtmc vs HF transformers)
  diff_layers.py                     # Per-layer hidden state comparison
  diff_vl.py                         # VL diff testing (vision features, generation, C++ parity)
  test_runner_parity.py              # Python vs C++ runtime parity
  test_graph_ops.py                  # TRT graph operation testing
scripts/                             # Infrastructure & utility scripts
  new_family.py                      # Scaffold a new family plugin from HF repo
  validate_family.sh                 # One-command validation gate (build + diff + parity)
  autopilot/                         # Autonomous model family discovery + implementation
    autorun.py                       # One-command entry: discover → select → dispatch → validate
    discover.py                      # Query HF Hub → find unsupported model_types → JSON task list
    dispatch.py                      # Launch parallel Claude Code agents across agent workspaces
    run.sh                           # Shell wrapper for discover + dispatch
tests/
  test_e2e.py                        # Unified E2E entrypoint (parametrized over all manifests)
  conftest.py                        # Shared CLI options (--engine-dir, --trtmc-binary, etc.)
  cpp/                               # C++ runtime unit tests
    test_helpers.h                   # Shared helpers: temp dirs, safetensors writing
    test_bundle_format.cpp ...       # 61 test executables (bundle, tokenizers, CUDA, KV cache, etc.)
  builder/                           # Python builder unit tests
    conftest.py                      # TRT runner fixture, skip markers
    test_config.py ...               # 70 test modules (config, weights, graph ops/blocks, plugins, etc.)
  tools/                             # Diff framework self-tests
    conftest.py                      # Adds tools/ to import path
    test_diff_logits.py ...          # 18 test modules (logits, audio, segmentation, diffusion, etc.)
  e2e/                               # E2E model manifests
    models/                          # 68 per-model JSON manifests (one file per model)
  e2e_harness/                       # Unified E2E test framework (DIP architecture)
    contracts.py                     # Dataclasses + protocols (E2ECase, StageOutput, CompareResult)
    orchestrator.py                  # Lifecycle coordinator (preflight -> build -> run -> compare)
    registry.py                      # Auto-discovery of runners, references, comparators
    manifest_loader.py               # JSON manifest -> E2ECase with auto-inferred fields
    artifact_sink.py                 # Persist artifacts (JSON metadata, logits, audio, images)
    runners/                         # TRT strategy runners (one per task_strategy)
      text_generation.py             # Causal LM (decoder, MoE, SSM, RWKV)
      vision_language.py             # VL models (Qwen VL, InternVL)
      audio_speech.py                # Whisper, Bark, PersonaPlex
      diffusion.py                   # Wan T2V, FLUX, Z-Image
      segmentation.py                # SegFormer, SAM
      embedding.py reranking.py      # Eagle models
      encoder_only.py                # BERT
    references/                      # Gold-standard reference backends
      hf_transformers.py             # HF Transformers (text, VL, audio, segmentation)
      hf_diffusers.py                # HF Diffusers (Wan, FLUX, Z-Image)
      torch_reference.py             # PyTorch reference (speech-to-speech)
    comparators/                     # Metric computation + threshold gating
      text.py                        # 6-metric composite gating for text gen
      vision_language.py             # Vision cosine + text NED/agreement
      text_to_audio.py               # RMS, duration ratio, mel/spectral distance
      diffusion.py                   # Pixel stats, temporal consistency, PSNR/SSIM
      segmentation.py                # mIoU, pixel accuracy, boundary F-score
      speech_to_text.py audio.py     # Transcript similarity
    thresholds/                      # Default + per-model threshold profiles
      defaults/                      # Per-strategy JSON threshold files
```

## Torch-TRT model transforms

When transforming a new model to torch-trt (building a `.trtfb` bundle via `tensorrt_model_connect/`):

1. **Read current website docs and source first** — use `website/docs/features/model-families.md`, `website/docs/features/runtime-strategies.md`, and the live `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/` source as ground truth.
2. **Do not stop until output is produced and validated** — iterate through build → inference → diagnose → fix cycles until the model produces correct output. Save all outputs to `/workspace/tensorrt-model-connect/outputs/<model_name>/` with descriptive filenames so the user can inspect them.
3. **Record new issues and work in the MR description** — include symptom, root cause, fix, failed attempts, and validation output.

## Autopilot: autonomous model family onboarding

The autopilot system discovers unsupported HuggingFace model families and
dispatches parallel agents to implement them end-to-end — Python plugin,
C++ runtime plugin (if needed), validation, and E2E manifest. One command:

```bash
# Fully autonomous — discovers gaps, implements, validates, reports
python3 scripts/autopilot/autorun.py --auto

# Interactive (shows candidates, asks Y/n before each batch)
python3 scripts/autopilot/autorun.py

# Control scope
python3 scripts/autopilot/autorun.py --auto --limit 8 --min-downloads 5000000

# Dry run (preview what would be dispatched)
python3 scripts/autopilot/autorun.py --dry-run
```

### How it works

1. **Discover** (`discover.py`): Queries HuggingFace Hub for top N transformer
   models by downloads, extracts `model_type`, checks against existing family
   plugins via `find_plugin()`, outputs ranked gaps as JSON.

2. **Select** (`autorun.py`): Filters gaps by feasibility — skips exotic
   vision/audio-only models, optionally skips `trust_remote_code` models.
   Ranks by total downloads.

3. **Dispatch** (`dispatch.py` or `autorun.py`): Launches `claude --print`
   sessions in parallel across isolated agent workspaces (agent-1 through
   agent-4). Each agent gets a self-contained prompt.

4. **Agent loop** (per model): Each agent autonomously runs:
   - Scaffold plugin via `new_family.py`
   - Build bundle via `trtmc-build build`
   - Validate: compare TRT output against HuggingFace (agent picks the
     metric — cosine similarity, text match, PSNR, etc.)
   - If build or validation fails → read error, read HF source code, fix
     plugin, rebuild, re-validate
   - If no existing C++ runtime strategy fits → create a new C++ plugin
     (.cpp file), register it, rebuild C++ binary
   - Iterate until `./build/trtmc run <bundle> --prompt "..." --max-new-tokens N`
     produces correct output
   - Create E2E manifest (no skip field)

### Architecture principles

- **Decoupled plugins**: Each model family gets its own isolated files.
  Agents never modify existing plugins. Shared code only through
  `graph_ops.py`, `graph_blocks.py`, and `shared/plugin_helpers.h`.
- **Agent decides validation**: The prompt does not prescribe HOW to
  validate. The agent reads the model's modality, picks the right metric,
  and reports actual numbers.
- **Full E2E required**: Every onboarded model must work via the C++ binary.
  No "skip" in manifests. If a new C++ runtime plugin is needed, the agent
  implements it.

### Prerequisites

```bash
# Bootstrap agent workspaces (one-time)
./scripts/bootstrap_workspace.sh --id agent-1 --branch master --detach
./scripts/bootstrap_workspace.sh --id agent-2 --branch master --detach
./scripts/bootstrap_workspace.sh --id agent-3 --branch master --detach
./scripts/bootstrap_workspace.sh --id agent-4 --branch master --detach

# claude CLI must be in PATH
which claude
```

## Adding a new model family (manual path)

Adding a new model family manually is done in the Python build package. For standard decoder, MoE, and extended-decoder families, no C++ changes are needed. Families with fundamentally different state management (e.g., Mamba/SSM, diffusion, audio) require a new C++ runtime plugin — see "Adding a new C++ runtime plugin" below.

For the **automated path**, use the autopilot (see above).

### Quick path (scaffolding script)

```bash
# 1. Generate a plugin from a HuggingFace model
python3 scripts/new_family.py \
  --model-type phi3 \
  --hf-repo microsoft/Phi-3-mini-4k-instruct \
  --family-name phi

# 2. Review and customize the generated plugin
$EDITOR tensorrt_model_connect/tensorrt_model_connect/families/phi.py

# 3. Validate end-to-end
./scripts/validate_family.sh microsoft/Phi-3-mini-4k-instruct
```

### Manual path

1. **Create `tensorrt_model_connect/tensorrt_model_connect/families/<family>.py`** — implement the `FamilyPlugin` protocol (see `base.py`). This handles:
   - Matching HF `model_type` / `architectures`
   - Mapping HF safetensors weight keys to the engine builder's expected names
   - Any family-specific weight pre-processing (e.g., Gemma's +1.0 to RMSNorm gamma, embedding scaling)
   - Graph construction customization if the family diverges from the standard decoder pattern

2. **The plugin is auto-discovered** — `families/__init__.py` uses `pkgutil.iter_modules()` to scan for any `.py` file with a module-level `plugin` attribute. Zero edits to shared files needed.

3. **Validate** — run `./scripts/validate_family.sh <hf-repo-or-path>` which builds a bundle, runs diff_logits (battery), diff_layers, and runner parity tests.

4. **Add E2E manifest** — create `tests/e2e/models/<model-name>.json`:
   ```json
   {
     "name": "my-model",
     "hf_id": "org/model-name",
     "bundle": "my-model.trtfb",
     "family": "my_family",
     "runtime_strategy": "decoder_kv_cache",
     "max_cache_length": 256,
     "prompt": "The capital of France is",
     "max_new_tokens": 20,
     "logit_atol": 1e-3
   }
   ```
   The `runtime_strategy` auto-selects the runner, comparator, reference,
   stages, and thresholds. No code changes needed. Run:
   ```bash
   pytest tests/test_e2e.py::test_e2e[my-model] -v \
     --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
     --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
     --rebuild-engines
   ```

### Adding a model with an existing family (no new plugin)

If the model's `model_type` matches an existing family plugin, you only need
the JSON manifest — no Python code at all:

```bash
# 1. Create the manifest
cat > tests/e2e/models/my-new-model.json << 'EOF'
{
  "name": "my-new-model",
  "hf_id": "org/my-new-model",
  "bundle": "my-new-model.trtfb",
  "family": "qwen",
  "runtime_strategy": "decoder_kv_cache",
  "max_cache_length": 256,
  "prompt": "Hello world",
  "max_new_tokens": 20
}
EOF

# 2. Run E2E (auto-builds bundle, runs TRT, compares against HF)
pytest tests/test_e2e.py::test_e2e[my-new-model] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --rebuild-engines
```

### runtime_strategy to task_strategy mapping

| runtime_strategy | task_strategy | Runner | Reference |
|-----------------|---------------|--------|-----------|
| decoder_kv_cache | text_generation_causal | text_generation | hf_transformers |
| decoder_moe | text_generation_causal | text_generation | hf_transformers |
| ssm_recurrent | text_generation_causal | text_generation | hf_transformers |
| rwkv_recurrent | text_generation_causal | text_generation | hf_transformers |
| hybrid_mamba_attention | text_generation_causal | text_generation | hf_transformers |
| vision_language | vision_language_generation | vision_language | hf_transformers |
| speech_to_text | speech_to_text | audio_speech | hf_transformers |
| text_to_audio_bark | text_to_audio | audio_speech | hf_transformers |
| text_to_audio_magpie | text_to_audio | audio_speech | hf_transformers |
| speech_to_speech | speech_to_speech | audio_speech | torch_reference |
| omni_multimodal | omni_multimodal | omni | torch_reference |
| diffusion_flux | diffusion_media_generation | diffusion | hf_diffusers |
| diffusion_wan | diffusion_media_generation | diffusion | hf_diffusers |
| diffusion_zimage | diffusion_media_generation | diffusion | hf_diffusers |
| diffusion_pixart | diffusion_media_generation | diffusion | hf_diffusers |
| segmentation | segmentation | segmentation | hf_transformers |
| prompted_segmentation | prompted_segmentation | segmentation | hf_transformers |
| object_detection | object_detection | segmentation | hf_transformers |
| encoder_only | encoder_only_nlp | encoder_only | hf_transformers |
| embedding | embedding | embedding | hf_transformers |
| reranking | reranking | reranking | hf_transformers |
| text_to_text | text_to_text | text_generation | hf_transformers |
| neural_operator | neural_operator | encoder_only | torch_reference |

Legacy aliases (`text_to_audio`, `diffusion`) are auto-normalized by `pipeline_factory.cpp` — new bundles use the split strategy strings above.

### Example

See `tensorrt_model_connect/tensorrt_model_connect/families/qwen.py` for the Qwen3 plugin (standard decoder).
See `tensorrt_model_connect/tensorrt_model_connect/families/phi.py` for Phi-3 (fused QKV/gate_up weight splitting).
See `tensorrt_model_connect/tensorrt_model_connect/families/phi_moe.py` for Phi-MoE (MoE with SparseMixer routing, uses `graph_blocks.add_attention_block`).
See `tensorrt_model_connect/tensorrt_model_connect/families/qwen_vl.py` for Qwen VL (Qwen2.5-VL standard + Qwen3-VL DeepStack via `graph_blocks` composition).
See `tensorrt_model_connect/tensorrt_model_connect/families/mamba.py` for Mamba (SSM, custom graph + C++ backend).
See `tensorrt_model_connect/tensorrt_model_connect/families/t5.py` for T5 (encoder-decoder seq2seq, custom graph + `text_to_text` C++ plugin).
See `tensorrt_model_connect/tensorrt_model_connect/families/deberta.py` for DeBERTa (disentangled attention, custom encoder graph).
See `tensorrt_model_connect/tensorrt_model_connect/families/modernbert.py` for ModernBERT (PRE-norm, fused QKV, GeGLU, per-layer RoPE).
See `tensorrt_model_connect/tensorrt_model_connect/families/base.py` for the plugin protocol.

### Adding a new C++ runtime plugin

When a model family requires fundamentally different runtime behavior (new state management, multi-engine orchestration, non-autoregressive inference), add a new C++ pipeline plugin. **No edits to `pipeline_factory.cpp` are needed.**

1. **Create `src/runtime/plugins/<name>_plugin.cpp`** — implement `IPipelinePlugin`:
   ```cpp
   #include "trtmc/runtime/pipeline_registry.h"
   #include "runtime/plugins/shared/plugin_helpers.h"

   namespace trtmc {
   class MyPlugin final : public IPipelinePlugin {
   public:
       std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
           // Load engines, create tokenizer, build pipeline
           auto loaded = load_trt_module_from_plan(
               find_section(ctx.bundle, "engine_plan"), "engine_plan");
           auto tokenizer = create_tokenizer_from_bundle(ctx.bundle, ctx.hf_python);
           // ... construct and return pipeline
       }
   };
   } // namespace trtmc

   // Register through the generated plugin manifest (inside namespace trtmc):
   REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_my_plugin, MyPlugin, "my_strategy");
   // Or for multiple strategies:
   // REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_my_plugin, MyPlugin,
   //                                          "strategy_a", "strategy_b");
   ```

2. **Add one manifest entry** to `cmake/trtmc_pipeline_plugins.cmake`:
   `my_plugin.cpp|register_my_plugin`.

3. Reconfigure/rebuild. The manifest adds the `.cpp` to `trtmc_core` and
   generates the explicit registrar call.

4. **Create or reuse a pipeline class** in `src/runtime/pipelines/` that implements `IPipeline`.

5. **Set `runtime_strategy`** in the Python family plugin so bundles carry the new strategy string.

**Examples of C++ runtime plugins:**
- `decoder_plugin.cpp` — decoder-only text generation (simplest reference)
- `encoder_plugin.cpp` — encoder-only (BERT, ELECTRA, ModernBERT, DeBERTa)
- `whisper_plugin.cpp` — encoder-decoder with mel spectrogram input
- `t5_plugin.cpp` — encoder-decoder with text token input (T5, seq2seq)
- `flux_plugin.cpp` — diffusion model (multi-step denoising loop)

**Note:** The autopilot system creates C++ plugins automatically when no
existing strategy fits the model. See "Autopilot" section above.

### Diff-test framework

Pure-Python TRT-vs-HF comparison (no C++ binary needed). Requires `torch`.

```bash
# E2E logit comparison (per-step logits, text match)
python3 tools/diff_logits.py \
  --model Qwen/Qwen3-0.6B --atol 1e-3 --battery

# Per-layer hidden state comparison (embedding, all layers, logits)
python3 tools/diff_layers.py \
  --model Qwen/Qwen3-0.6B --atol 0.05

# For models requiring custom tokenizer code (e.g., Phi-3), add --trust-remote-code:
python3 tools/diff_logits.py \
  --model microsoft/Phi-3-mini-4k-instruct --atol 1e-3 --battery --trust-remote-code
```

The diff-test framework uses `tensorrt_model_connect.debug_runner.TrtRunner` for pure-Python TRT inference with device-resident KV cache, matching the C++ `DeviceKvCache` behavior exactly. For Mamba/SSM models, `MambaTrtRunner` handles device-resident recurrent state. For VL models, `VLTrtRunner` combines vision + text decoders with image preprocessing. `diff_layers.py` builds a debug engine with per-layer hidden state outputs via `debug_layer_outputs=True`.

**VL diff testing** (`diff_vl.py`):
```bash
# Vision encoder feature comparison (TRT vs HF)
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --model Qwen/Qwen2.5-VL-3B-Instruct --atol 0.1

# Vision-only smoke test (no HF model needed)
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg --vision-only

# Debug with preprocessor override
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --vision-only --preprocessor-type simple_chw

# Full VL generation + C++ binary parity
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --binary ./build/trtmc --hf-python /opt/venv/bin/python
```

**Runner parity guarantee**: If you change the C++ mask/cache/position logic (`trt_decode_runtime.cpp`, `device_kv_cache.cpp`), you MUST also update `debug_runner.py` and verify with:
```bash
python3 tools/test_runner_parity.py \
  --bundle /tmp/qwen3.trtfb --binary ./build/trtmc \
  --hf-python /opt/venv/bin/python --max-new-tokens 20
```

## Container workflow (TRT GPU)

Prerequisites: Docker + NVIDIA Container Toolkit. The container is fully self-contained. TRT headers come from apt (`libnvinfer-headers-dev`), TRT libs come from pip (`tensorrt_cu13`), and `LD_LIBRARY_PATH` is preconfigured in the image.

### 1) Build and launch the dev container
```bash
./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

### 2) Install local packages and build (inside container)
```bash
pip install --no-deps -e tensorrt_model_connect/
pip install --no-deps -e tensorrt_model_connect/

cmake -S . -B build -G Ninja \
  -DTRTMC_TRT_INCLUDE_DIR=$TRT_INC_DIR \
  -DTRTMC_TRT_LIBRARY=$TRT_LIB_DIR/libnvinfer.so \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j
```

### 3) Build bundle + validate Qwen3 TRT E2E (inside container)
```bash

# Build a bundle (auto-downloads from HuggingFace)
trtmc-build build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256

# Inspect it
trtmc-build inspect /tmp/qwen3.trtfb

# Run from bundle using the C++ runtime
./build/trtmc run /tmp/qwen3.trtfb --prompt "Hello" --max-new-tokens 5 \
  --hf-python /opt/venv/bin/python
```

### 4) Build + run a vision-language model (inside container)
```bash

# Build a VL bundle (text decoder + vision encoder)
trtmc-build build Qwen/Qwen2.5-VL-3B-Instruct -o /tmp/qwen25vl.trtfb --max-cache-length 384

# Run with an image
./build/trtmc run /tmp/qwen25vl.trtfb --prompt "Describe this image." \
  --image /path/to/image.jpg --max-new-tokens 30 \
  --hf-python /opt/venv/bin/python
```

### 5) Build + run Qwen3-VL with DeepStack (inside container)
```bash

# Build a Qwen3-VL bundle (text decoder with DeepStack + vision encoder with multi-level outputs)
trtmc-build build Qwen/Qwen3-VL-2B-Instruct -o /tmp/qwen3vl.trtfb --max-cache-length 256

# Text-only inference (DeepStack inactive during text-only)
./build/trtmc run /tmp/qwen3vl.trtfb --prompt "The capital of France is" \
  --max-new-tokens 20 --hf-python /opt/venv/bin/python

# VL inference with image (DeepStack active during image token prefill)
./build/trtmc run /tmp/qwen3vl.trtfb --prompt "Describe this image." \
  --image /path/to/image.jpg --max-new-tokens 30 \
  --hf-python /opt/venv/bin/python
```

### 6) Run unified E2E suite (inside container)
```bash

# Single model (auto-builds bundle if missing):
python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# All 68 models (force rebuild):
python -m pytest tests/test_e2e.py -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --rebuild-engines --e2e-artifacts-dir /tmp/e2e_artifacts

# Text-gen models only (~30 min):
python -m pytest tests/test_e2e.py -v \
  --e2e-task-strategy text_generation_causal \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# Diffusion models (Wan T2V, FLUX, Z-Image — ~45 min):
python -m pytest tests/test_e2e.py -v \
  --e2e-task-strategy diffusion_media_generation \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```

Artifacts (WAV audio, PNG frames/images, logits, transcripts) are saved to
`--e2e-artifacts-dir` for human inspection.

### 7) MMLU sanity check (inside container)
```bash
/opt/venv/bin/python scripts/eval_mmlu.py \
  --backend trtmc --model /tmp/qwen3.trtfb \
  --trtmc-binary ./build/trtmc \
  --subject all --split test \
  --num-samples 4 --min-accuracy 0.0
```
