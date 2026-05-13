# TensorRT-Model-Connect Wiki

A split-language system for TensorRT inference: **Python builds** optimized TRT engines from HuggingFace model checkpoints, **C++ runs** them at maximum speed. The Python `tensorrt_model_connect/` package reads safetensors, constructs TRT networks via the TensorRT Python API, and produces self-contained `.trtfb` bundles. The C++ runtime loads those bundles, deserializes TRT engines, resolves the `runtime_strategy`, and runs inference through the appropriate pipeline implementation.

## Quick Navigation

| Page | Description |
|------|-------------|
| **[Architecture Overview](Architecture-Overview.md)** | End-to-end architecture: Python builder, C++ runtime, strategy dispatch |
| **[Static Design](Static-Design.md)** | Current class and module structure |
| **[Dynamic Design](Dynamic-Design.md)** | Runtime and build-time sequence diagrams |
| **[Pipeline Deep Dive](Pipeline-Deep-Dive.md)** | Detailed walkthrough of bundle loading, factory dispatch, and pipeline assembly |
| **[Source Layout](Source-Layout.md)** | File and directory guide for the current codebase |
| **[Runtime Target Architecture](Runtime-Target-Architecture.md)** | **IMPLEMENTED** -- describes the plugin-registry-based runtime design (now the current architecture) |
| **[Testing and Validation](Testing-and-Validation.md)** | Test tiers, smoke/E2E policy, CCN gate, traceability requirements |
| **[Traceability Matrix](Traceability-Matrix.md)** | Bi-directional architecture/design/test traceability (ARCH/UD/UT/IT) |
| **[ISO 26262 Compliance](ISO-26262-Compliance.md)** | Safety-related development process alignment (NEW) |
| **[Adding a Model Family](Adding-a-Model-Family.md)** | How to add a new family plugin (autopilot + manual paths) |
| **[Architecture Extensibility Assessment](Architecture-Extensibility-Assessment.md)** | Coverage of model families and runtime strategies |
| **[HF vs TRT Comparison](HF-vs-TRT-Comparison.md)** | HuggingFace reference flow versus TRT runtime flow |
| **[TRT Internals](TRT-Internals.md)** | TensorRT graph-building and engine lifecycle details |

## Core Design Principles

1. **Python builds, C++ runs.** Checkpoint loading, graph construction, and engine compilation stay in Python (`tensorrt_model_connect/`). Low-latency inference stays in C++ (`src/`).
2. **The bundle is self-describing.** Each `.trtfb` bundle carries a `config.json` with `runtime_strategy`, model dimensions, tokenizer settings, and all metadata the C++ runtime needs. No external configuration files are required.
3. **Strategy is resolved once at bundle load.** `PipelineFactory::from_bundle()` reads `runtime_strategy` from the bundle's config, looks up the matching `IPipelinePlugin` in the `PipelineRegistry` singleton, and delegates pipeline construction to the plugin. There is no per-request strategy redispatch.
4. **Family plugins are auto-discovered.** Python family plugins in `tensorrt_model_connect/tensorrt_model_connect/families/` are found via `pkgutil.iter_modules()`. Adding a new family requires only a new `.py` file with a module-level `plugin` attribute -- no edits to shared registration code.
5. **Complexity budget is enforced.** C++ cyclomatic complexity must stay at or below the repository gate (CCN at most 10), checked by `tools/check_cyclomatic_complexity.py` and CI.
6. **Traceability is required.** Architecture decisions (ARCH-*), unit designs (UD-*), and tests (UT-*/IT-*) must stay linked per the traceability matrix.

## Architecture At A Glance

The system has two phases.

### 1. Build Phase (Python)

```
trtmc-build build <hf-model-or-dir> -o model.trtfb
```

- Family plugin matches `config.json` model_type
- Checkpoint mapper normalizes HF safetensors weights
- Graph builder emits TRT networks (graph_ops -> graph_blocks -> standard_decoder_builder)
- Engine compiler produces optimized TRT engine plans
- Bundle writer packages engine plans, tokenizer files, and metadata into `.trtfb`

### 2. Run Phase (C++)

```
trtmc run model.trtfb --prompt "Hello" --max-new-tokens 20
```

- `trtmc_create_pipeline_ex()` validates input and reads the `.trtfb` bundle
- `PipelineFactory::from_bundle()` extracts `config.json`, parses `runtime_strategy`
- `PipelineRegistry::instance().lookup(strategy)` finds the registered `IPipelinePlugin`
- The plugin's `create()` method loads TRT engines, creates tokenizers, allocates KV cache or recurrent state
- Returns an `IPipeline` pointer to the caller

```text
trtmc_create_pipeline_ex(bundle_path)
  -> ReadBundleFile()
  -> extract_json_string("runtime_strategy")
  -> normalize_legacy_strategy()
  -> PipelineRegistry::instance().lookup(strategy)
  -> plugin->create(PipelineContext{...})
  -> IPipeline*
```

## Container Setup

All development and test workflows run inside the dev container.

```bash
# Build and launch
./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh

# One-shot repo setup (editable install + C++ build + unit tests)
./scripts/bootstrap_workspace.sh
```

For isolated multi-agent workflows, use per-team containers:

```bash
./scripts/bootstrap_workspace.sh --id <team-id> --branch <branch> --detach
docker exec trtmc-dev-gb300-<team-id> <command>
```

## Built-In Model Support

Family plugins are auto-discovered from `tensorrt_model_connect/tensorrt_model_connect/families/`. Any HF model whose `model_type` maps to a plugin is buildable. There is no static registration list to update.

To enumerate all family plugins in your checkout:

```bash
ls tensorrt_model_connect/tensorrt_model_connect/families/*.py \
  | sed 's|.*/||; s|\.py$||' \
  | grep -v -E '^(__init__|base)$' \
  | sort
```

As of **March 2026**, the repository contains **63** family modules covering standard decoders (Qwen, LLaMA, Mistral, Phi, GPT-2, OPT, Bloom, Gemma, Falcon, etc.), MoE (Mixtral, Phi-MoE, Qwen-MoE, DeepSeek-V2), SSM/recurrent (Mamba, RWKV), encoder-only (BERT, ELECTRA, ModernBERT, DeBERTa, DistilBERT, RoBERTa, MPNet, Albert, ConvBERT, FNet, DPR, XLNet), encoder-decoder (T5, Marian, BART, M2M-100), speech (Whisper, Bark, MagpieTTS, PersonaPlex, Qwen3-Omni, Canary), vision-language (Qwen-VL, InternVL, Phi4-Multimodal, Eagle-VLM), segmentation (SegFormer, SAM), diffusion (Wan-T2V, FLUX, Z-Image, PixArt), and embedding/reranking (Eagle). 84 E2E model manifests cover the full test matrix.

To add more families automatically, use the autopilot:
```bash
python3 scripts/autopilot/autorun.py --auto
```

## Quick Start

```bash
# Build a bundle from HuggingFace (inside container)
trtmc-build build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256

# Inspect the bundle
trtmc-build inspect /tmp/qwen3.trtfb

# Run inference
./build/trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python

# Vision-language model
trtmc-build build Qwen/Qwen2.5-VL-3B-Instruct -o /tmp/qwen25vl.trtfb --max-cache-length 384
./build/trtmc run /tmp/qwen25vl.trtfb \
  --prompt "Describe this image." --image photo.jpg \
  --max-new-tokens 30 --hf-python /opt/venv/bin/python

# Run E2E validation (single model)
pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```
