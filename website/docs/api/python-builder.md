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

## `build()` API

The shared API is intentionally only:

```python
build(model_id_or_path, output_path, **options)
```

The shared layer resolves or downloads the model, selects its owner, and calls
that owner's `model.build()` exactly once. `model_revision` is consumed while
resolving a Hugging Face snapshot; `rtx` selects the TensorRT backend before
the owner module is imported. Every other option, its default, and its
validation belong to the selected model owner. For example, decoder owners may
accept `max_cache_length` or `decoder_engine_layout`, while diffusion owners may
accept `diffusion_overrides` or `max_batch_size`. Unsupported options fail in
the owner rather than expanding a central build protocol.

Feature pages document the option names accepted by the relevant owners. The
CLI's namespaced `--config` and repeatable `--set` inputs provide the generic
extension path without adding a new shared Python parameter for each model.

## Required family model module

Family packages are indexed from
`python/tensorrt_model_connect/models/<family>/MODEL.toml`:

1. For a full config, `architecture_patterns`, aliases, and prefixes select
   bounded candidates whose required `matches(config)` functions run.
2. For a string or `model_type`, discovery uses the same descriptor index.
3. For a Diffusers pipeline class, discovery uses only descriptor
   `diffusion_pipeline_classes`; there is no `pkgutil` fallback.

Discovery imports `models.<family>.model` directly. `__init__.py` remains
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
