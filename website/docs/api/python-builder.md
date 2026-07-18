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

## Build inputs

| Input | Meaning |
| --- | --- |
| HuggingFace repo ID | The builder resolves and downloads model files. |
| Local directory | The builder reads local `config.json`, weights, tokenizer, and model-specific assets. |
| Diffusers model directory | The builder uses `model_index.json` and `find_diffusion_plugin()`. |

## Important options

| Option | Purpose |
| --- | --- |
| `max_cache_length` | Default KV cache length for decoder-style bundles. |
| `precision` | Optional engine precision override: `fp32`, `fp16`, or `bf16`. When omitted, the family default is used (BF16 for Wan2.2; FP32 fallback for families without a declaration). |
| `quantize` | Structured quantization format such as `fp8` or `int4_awq`. |
| `dynamic_kv_cache` | Build decoder bundles with runtime-resizable KV cache support. |
| `rtx` | Build for TensorRT-RTX backend selection. |
| `diffusion_overrides` | Image/video shape and inference-step overrides for diffusion models. |

## Family plugin protocol

Family plugins implement `python/tensorrt_model_connect/families/base.py`. Required pieces are:

```python
class FamilyPlugin(Protocol):
    name: str
    def matches(self, model_type: str) -> bool: ...
    def load_weights(self, model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict: ...
    def build_engine(self, config: ModelConfig, weights: WeightDict, max_cache_length: int, *, precision: str = "fp32", quant_ctx=None, verbose: bool = False) -> bytes: ...
```

Optional methods add quantization, vision-language, diffusion, and FP8 calibration behavior.
