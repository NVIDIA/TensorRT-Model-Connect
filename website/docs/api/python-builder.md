---
title: Python Builder API
---

The Python package lives under `python/tensorrt_model_connect/`. Complete
[Installation](../getting-started/installation.md) before using this reference.
Editable developer installs and release-wheel boundaries are documented there,
not duplicated on the API page.

## Public package exports

The package root exports the following names:

| Export | Contract |
| --- | --- |
| `build` | Resolve a Hugging Face ID or local directory, import its family `model.py`, call `model.build()` once, and write the destination bundle. |
| `write_bundle` | Low-level serializer for callers that already own a `BundleInfo` and ordered `BundleSection` objects. It does not resolve a model or build an engine. |
| `ModelConfig` | Dataclass parser for Hugging Face `config.json` architecture fields, including supported nested text/language config forms. |
| `Pipeline` | Thin subprocess wrapper over the native `trtmc` CLI for text and vision-language generation plus bundle inspection. |
| `__version__` | Package version string. |

These exports are lazy-loaded by `tensorrt_model_connect.__init__` so importing
the package does not immediately bind the TensorRT backend.

## Builder usage

```python
import tensorrt_model_connect

tensorrt_model_connect.build(
    "Qwen/Qwen3-0.6B",
    "/tmp/qwen3.bundle",
    verbose=True,
)
```

Both this API and `trtmc build` resolve the model and call the selected family
module directly. Qwen owns its exact optimized-profile decision inside
`qwen/model.py`; all other build policy remains inside the corresponding
family module. The shared builder does not retry another implementation.

## Python runtime wrapper

`Pipeline` is useful when a Python application wants the established CLI text
contract without linking the C++ API:

```python
from tensorrt_model_connect import Pipeline

pipe = Pipeline("/tmp/qwen3.bundle")
text = pipe("The capital of France is", max_new_tokens=20, timeout=120)
metadata = pipe.inspect()
```

The constructor also accepts explicit `binary` and `hf_python` paths. Without
`binary`, it first asks the installed package resources for a packaged `trtmc`
executable, then checks a source checkout's `build/trtmc`, then `PATH`.
Construction raises `FileNotFoundError` when none is executable.

Calling the wrapper forwards `prompt`, optional `image`, optional
`lora_adapter`/`lora_adapter_id`, and `max_new_tokens` to `trtmc run`. It
returns stripped standard output. A nonzero CLI exit raises `RuntimeError`
containing the exit code and standard error; `subprocess.TimeoutExpired`
propagates when the call's time limit expires. The wrapper does not expose the
full typed multimodal `IPipeline` surface or an in-process runtime. Use the C++
API or CLI directly for other task-specific operations.

## Build inputs

| Input | Meaning |
| --- | --- |
| Hugging Face repo ID | The builder resolves and downloads model files. |
| Local directory | The builder reads local `config.json`, weights, tokenizer, and model-specific assets. |
| Diffusers model directory | The resolver maps `model_index.json` through family `diffusion_pipeline_classes`. |

## Complete `build()` parameter reference

`build()` currently has 31 public parameters.

| Parameter | Purpose |
| --- | --- |
| `model_id_or_path` | Hugging Face repository ID or resolved local model directory. |
| `output_path` | Destination `.bundle` bundle path. |
| `max_cache_length` | Explicit KV cache length. Omitted/`None` lets the family choose: eligible dense Qwen3/Llama use `max_position_embeddings`, while other family recipes normally use 256. |
| `model_revision` | Hugging Face commit, tag, or branch to resolve. |
| `decoder_engine_layout` | `split` or `dual_profile` for supported decoders. |
| `dynamic_kv_cache` | Build decoder bundles with runtime-resizable KV cache support. |
| `dynamic_kv_profile_rows_override` | Explicit dynamic-KV profile upper bounds. |
| `precision` | Engine precision: `fp32`, `fp16`, or `bf16`. |
| `fp32_layers` | Model-local layer indices that should compute in FP32. |
| `quantize` | Structured quantization format such as `fp8` or `int4_awq`. |
| `quant_scales` | Path to precomputed quantization scales when the selected quantizer accepts them. |
| `quant_calibration_samples` | Maximum calibration sample count; defaults to 512. |
| `verbose` | Emit detailed builder diagnostics. |
| `kernel_artifacts` | Optional named shared libraries to embed beside the generated engine plans. |
| `fp8_scales` | FP8 scale mapping or serialized scale source used by compatible native families. |
| `save_fp8_scales` | Optional output path for calibrated FP8 scales. |
| `rtx` | Build for TensorRT-RTX backend selection. |
| `triattention_stats_path` | TriAttention statistics input used for KV compaction. |
| `triattention_kv_budget` | Retained KV-token budget. |
| `triattention_divide_length` | Compaction scoring division length; defaults to 128. |
| `triattention_recent_window` | Recent-token protection window; defaults to 128. |
| `triattention_score_aggregation` | Score aggregation mode, currently `mean` or `max`. |
| `triattention_count_prompt_tokens` | Include prompt tokens in TriAttention accounting. |
| `triattention_protect_prefill` | Protect prefill tokens during compaction. |
| `triattention_disable_mlr` | Disable the MLR score component. |
| `triattention_disable_trig` | Disable the trigonometric score component. |
| `family_build_options` | Opaque model-family build options interpreted by the selected `model.py`. |
| `parallel_config` | Programmatic tensor-parallel build configuration. |
| `diffusion_overrides` | Image/video shape and inference-step overrides for diffusion models. |
| `build_timing_path` | Structured build-timing JSON output path. |
| `max_batch_size` | Maximum supported diffusion batch size, subject to family component policy. |

`decoder_engine_layout` is interpreted by the selected family. A split build
requires that family to implement compatible prefill/decode roles. The emitted
`config.json.decoder_engine_layout`
records the actual `split`, `dual_profile`, or `single` result, and only an
actual split bundle contains `prefill_engine_plan`.

## Required family model module

Family packages are indexed from
`python/tensorrt_model_connect/families/<family>/MODEL.toml`:

1. For a full config, `architecture_patterns`, aliases, and prefixes select
   bounded candidates whose required `matches(config)` functions run.
2. For a string or `model_type`, discovery uses the same descriptor index.
3. For a Diffusers pipeline class, discovery uses only descriptor
   `diffusion_pipeline_classes`; there is no `pkgutil` fallback.

Discovery imports `families.<family>.model` directly. `__init__.py` remains
empty so lightweight metadata/config consumers do not import TensorRT or
optional family dependencies. There is no base class, protocol, module field,
package scan, or compatibility shim.

```python
def matches(config) -> bool: ...

def build(model_dir: str, output_path: str, **options) -> None:
    # config → weights → engine/component plans → bundle sections → write_bundle
    ...
```

Families may split this recipe into local helpers, but those helpers are not a
cross-family protocol. Copy model-owned code when isolation is more valuable
than deduplication; share only model-independent leaf primitives.

{/* Collaborative review anchor: batch 2. */}
