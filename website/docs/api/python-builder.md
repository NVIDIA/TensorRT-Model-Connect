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
| Hugging Face repo ID | The builder resolves and downloads model files. |
| Local directory | The builder reads local `config.json`, weights, tokenizer, and model-specific assets. |
| Diffusers model directory | The builder uses `model_index.json` and `find_diffusion_plugin()`. |

## Complete `build()` parameter reference

`build()` currently has 31 public parameters. The source signature and this
table are checked together by the documentation workflow.

| Parameter | Purpose |
| --- | --- |
| `model_id_or_path` | Hugging Face repository ID or resolved local model directory. |
| `output_path` | Destination `.trtfb` bundle path. |
| `max_cache_length` | Default KV cache length for decoder-style bundles. |
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
| `trust_remote_code` | Strict `bool`, default `False`. Permit reviewed repository-provided tokenizer code. With `build()`, pin `model_revision`; with `build_bundle()`, use a reviewed immutable local snapshot. |
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

Both `build()` and `build_bundle()` reject non-boolean
`trust_remote_code` values, including strings such as `"false"`. When the
selected family requires a tokenizer, the builder also refuses to write a
bundle if it cannot reuse or generate `tokenizer.json`. Review any
repository-provided tokenizer code before explicitly setting
`trust_remote_code=True`. For `build()`, pin `model_revision` to the reviewed
commit. Because `build_bundle()` accepts a local directory rather than a
revision, pass it a reviewed, revision-fixed local snapshot. Tokenizer repair
may require a writable copy as described below.

## Tokenizer repair mutates the resolved model directory

On the native path, a family that requires a tokenizer reuses a compatible
`tokenizer.json`. If that file is missing, incompatible, or an undersized
WordPiece export, the builder starts a transaction in the resolved model
directory. It tries standard Hugging Face slow-to-fast conversion first and
then, if available, the family-owned `ensure_tokenizer_json` fallback. A
successful repair creates or replaces `tokenizer.json` in that directory
before special-token metadata is detected and before the bundle is written.
The resolved local directory therefore must be writable; a caller that needs
an immutable source snapshot should build from a writable copy.

Repairs targeting the same resolved directory share a process-reentrant,
cross-process advisory lock. Repair creates the persistent regular-file
sentinel `.trtmc-tokenizer-repair.lock` before any canonical mutation and never
removes it; the file is lock metadata, is not included in the bundle, and does
not by itself indicate an active repair. A waiting builder revalidates
`tokenizer.json` after acquiring ownership, so it reuses another builder's
committed result rather than regenerating or rolling it back. A compatible
snapshot with no sentinel keeps the read-only fast path. Unsafe sentinel types
or an unavailable lock fail closed before the canonical tokenizer is moved or
replaced.

Before either attempt, the transaction atomically quarantines an existing file
at `original-tokenizer.json` in a unique hidden `tokenizer-recovery-*`
directory. If that initial move fails, the canonical original remains
untouched and repair stops; the same is true if the recovery directory cannot
be reserved. If both attempts fail, the builder removes the unsuccessful
candidate and restores the original bytes and file type.
When an original existed, candidate-cleanup or restoration failures are
terminal and report the durable recovery path that still retains it for manual
recovery. If there was no original, ordinary failed-repair cleanup leaves
`tokenizer.json` absent. If that cleanup itself fails, the unsuccessful
candidate can remain at the canonical path and the terminal error reports the
cleanup failure; there is no original-recovery path to report. No bundle is
written after a failed repair. After a validated replacement commits, the
builder removes the quarantined old artifact on a best-effort basis. A
post-commit cleanup failure leaves the new compatible canonical
`tokenizer.json` committed, does not turn the repair into a failure, and emits
a warning identifying the recovery directory where cleanup residue may
remain.

The optional family hook has this effective contract:

```python
def ensure_tokenizer_json(
    self,
    model_dir,
    *,
    previous_error=None,
    trust_remote_code=False,
) -> bool: ...
```

The builder passes optional keywords only when the hook accepts them. The hook
runs only after standard conversion fails, while the rejected original is
quarantined. It must return a truthy value and leave
`model_dir/tokenizer.json` as a non-empty regular, non-symlink file accepted by
the native tokenizer validator; otherwise the outer transaction rolls it
back. Diffusion plugins choose their tokenizer directories through
`diffusion_tokenizer_bundle_sections()` and invoke its supplied
`ensure_tokenizer_json` callback for each required directory. Those repairs
finish before diffusion special-token detection and before generated or
pre-rendered bundle config is reconciled with the repaired tokenizer.

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
