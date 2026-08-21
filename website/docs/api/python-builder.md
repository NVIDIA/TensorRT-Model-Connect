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
| `build` | Resolve a Hugging Face ID or local directory and its family, honor a model-owned native default route when declared, otherwise try one exact qualified optimized profile before the native fallback, and write the destination bundle. This is the normal public build entry point. |
| `build_bundle` | Lower-level native builder for an already resolved local model directory. It writes the bundle directly and exposes additional builder-internal inputs; most callers should use `build`. |
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

Both this API and `trtmc build` resolve the model and family first. A family
whose `default_build_route` accepts the checkpoint owns the native build
directly; eligible dense Qwen3 and Llama currently use this route. Other
families probe their own optimized-runtime providers. Exactly one qualified
model/revision/active-target/options profile may claim such a request and
produce an optimized bundle; no claim continues to the native `FamilyPlugin`.
A selected provider build failure is terminal rather than a native fallback.

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
| Diffusers model directory | The builder uses `model_index.json` and `find_diffusion_plugin()`. |

## Complete `build()` parameter reference

`build()` currently has 30 public parameters.

| Parameter | Purpose |
| --- | --- |
| `model_id_or_path` | Hugging Face repository ID or resolved local model directory. |
| `output_path` | Destination `.bundle` bundle path. |
| `max_cache_length` | Explicit KV cache length. Omitted/`None` lets the family choose: eligible dense Qwen3/Llama use `max_position_embeddings`, while other native or legacy paths normally use 256. Qwen3 rejects a non-full-context value instead of falling back; other families apply their own capability policy. |
| `model_revision` | Hugging Face commit, tag, or branch to resolve. |
| `decoder_engine_layout` | `split` or `dual_profile` for supported decoders. |
| `dynamic_kv_cache` | Build compatible decoder bundles with runtime-resizable KV cache support. Dense Qwen3 rejects this option and requires fixed-capacity native KV. |
| `dynamic_kv_profile_rows_override` | Explicit dynamic-KV profile upper bounds. |
| `precision` | Engine precision: `fp32`, `fp16`, or `bf16`. |
| `fp32_layers` | Model-local layer indices that should compute in FP32. |
| `quantize` | Structured quantization format such as `fp8` or `int4_awq`. |
| `quant_scales` | Path to precomputed quantization scales when the selected quantizer accepts them. |
| `quant_calibration_samples` | Maximum calibration sample count; defaults to 512. |
| `verbose` | Emit detailed builder diagnostics. |
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
| `family_build_options` | Opaque model-family build options for the selected plugin. |
| `parallel_config` | Programmatic tensor-parallel build configuration. |
| `diffusion_overrides` | Image/video shape and inference-step overrides for diffusion models. |
| `build_timing_path` | Structured build-timing JSON output path. |
| `max_batch_size` | Maximum supported diffusion batch size, subject to family component policy. |

`decoder_engine_layout` is a requested layout, not a guarantee. A split build
requires a native decoder-KV runtime, no tensor parallelism, no dynamic KV or
TriAttention, and explicit family support for separate prefill/decode roles.
An embed-input family must opt into that contract separately. When a requested
split is unsupported, the builder logs the fallback and uses the family's
existing single-engine path. The emitted `config.json.decoder_engine_layout`
records the actual `split`, `dual_profile`, or `single` result, and only an
actual split bundle contains `prefill_engine_plan`.

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

{/* Collaborative review anchor: batch 2. */}
