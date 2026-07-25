---
title: Python Builder API
---

The Python package lives under `python/tensorrt_model_connect/`.

## Install

```bash
pip install -e . -C py-only=true
```

This is a developer-only editable install. It points imports at
`python/tensorrt_model_connect/` and skips the native wheel build. It does not
install the native `trtmc` executable or backend DSOs; pair it with a CMake
source build when using `./build/trtmc`.

Use `pip install --no-deps -e . -C py-only=true` only in a dev container that already has the declared dependencies installed. In a fresh Python environment, skipping dependencies will hide required packages such as `transformers`, `safetensors`, `onnx`, `onnxscript`, and `tensorrt`.
The release wheel installs the same builder package plus the native `trtmc`
executable and declares TensorRT as a dependency; use the wheel when you want
`trtmc build` and `trtmc run` available from one pip install.

## Python usage

```python
import tensorrt_model_connect

tensorrt_model_connect.build(
    "Qwen/Qwen3-0.6B",
    "/tmp/qwen3.trtfb",
    max_cache_length=256,
    precision="fp16",
    verbose=True,
)
```

`tensorrt_model_connect.__init__` lazily imports heavyweight builder helpers. That keeps TensorRT backend selection from happening too early when `--rtx` is used.

Both this API and `trtmc build` first try a family-owned optimized-runtime
provider after resolving the model and family. Exactly one qualified
model/revision/active-target/options profile may claim the request and produce
an optimized bundle. If no provider claims it, the normal native
`FamilyPlugin` path handles the options below. A selected provider build
failure is terminal rather than a native fallback.

## Build inputs

| Input | Meaning |
| --- | --- |
| HuggingFace repo ID | The builder resolves and downloads model files. |
| Local directory | The builder reads local `config.json`, weights, tokenizer, and model-specific assets. |
| Diffusers model directory | The builder uses `model_index.json` and `find_diffusion_plugin()`. |

## Important options

| Option | Purpose |
| --- | --- |
| `model_revision` | Hugging Face commit, tag, or branch to resolve. |
| `max_cache_length` | Default KV cache length for decoder-style bundles. |
| `decoder_engine_layout` | `split` or `dual_profile` for supported decoders. |
| `precision` | Engine precision: `fp32`, `fp16`, or `bf16`. |
| `fp32_layers` | Model-local layer indices that should compute in FP32. |
| `quantize` | Structured quantization format such as `fp8` or `int4_awq`. |
| `dynamic_kv_cache` | Build decoder bundles with runtime-resizable KV cache support. |
| `dynamic_kv_profile_rows_override` | Explicit dynamic-KV profile upper bounds. |
| `parallel_config` | Programmatic tensor-parallel build configuration. |
| `rtx` | Build for TensorRT-RTX backend selection. |
| `diffusion_overrides` | Image/video shape and inference-step overrides for diffusion models. |
| `max_batch_size` | Maximum supported diffusion batch size, subject to family component policy. |
| `family_build_options` | Opaque model-family build options for the selected plugin. |
| `build_timing_path` | Structured build-timing JSON output path. |

## Family plugin protocol

Family packages are indexed from
`python/tensorrt_model_connect/families/<family>/MODEL.toml`. The lookup route
depends on the input:

1. For a full config, `architecture_patterns` select bounded candidates whose
   `matches_config()` predicates run first. No match triggers the legacy
   `pkgutil` fallback over every non-private family module/package.
2. For a string or `model_type`, discovery tries a direct descriptor ID,
   alias/prefix candidates, then the same all-package fallback.
3. For a Diffusers pipeline class, discovery uses only descriptor
   `diffusion_pipeline_classes`; there is no `pkgutil` fallback.

Discovery imports the selected package and reads the package-level `plugin`
exported by `__init__.py`. The descriptor's `module` field is
specialization/tooling metadata, not an arbitrary runtime import selector, and
a loose module found only through compatibility scanning is not a complete
supported family. The protocol itself is defined in
`python/tensorrt_model_connect/families/base.py`.

```python
class FamilyPlugin(Protocol):
    name: str

    def matches(self, model_type: str) -> bool: ...

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict: ...

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes: ...
```

Optional methods add split decoder roles, quantization, vision-language,
diffusion component/bundle ownership, and FP8 calibration behavior. Treat the
live protocol as the source of truth instead of copying its complete optional
surface into downstream integrations.
