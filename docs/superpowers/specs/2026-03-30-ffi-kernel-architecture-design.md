# FFI Kernel Architecture for Performance Agent Optimization

**Date:** 2026-03-30
**Branch:** tvm-ffi
**Status:** Design

## Problem

The TVM-FFI kernel bridge lets a performance agent pull in arbitrary FFI-compatible kernels (FlashInfer, CuTe DSL, custom CUDA) to replace subgraphs in the TRT forward pass. The current implementation works end-to-end but has three problems:

1. **FlashInfer wiring is tangled inside `graph_blocks.py`** (lines 154-187) — kernel-specific reshapes, extra_args, and workspace are inlined in the structural attention block, mixed with model-architecture code that other agents edit.

2. **No deployment path to C++ runtime** — the kernel only exists in the Python process that built the engine. The `.trtfb` bundle contains the kernel NAME but not the kernel CODE. The C++ runtime calls `TVMFFIFunctionGetGlobal()` at inference time and fails because nobody registered the kernel.

3. **Kernel selection via environment variable** (`TRTMC_FFI_ATTENTION_KERNEL`) is invisible shared state that force-enables FP16 as a side effect.

## Design Constraints

- **Source code is the interface.** The perf agent modifies Python and C++ source code directly. No config schemas, registries, or frameworks that limit the search space.
- **Branch isolation.** Each perf agent works on its own branch/workspace. Merge conflicts are resolved at integration time. The architecture minimizes conflict surface area.
- **Any model type.** The architecture must work for decoders, encoders, diffusion (DiT + VAE), speech (Whisper, Bark), vision-language, SSM (Mamba, RWKV), MoE, segmentation — all 5 architecture patterns and 61 family plugins.
- **Forward pass + runtime co-design.** The perf agent can change both the TRT engine graph AND the C++ runtime (KV cache layout, dynamic shapes for batched prefill, speculative decode). These are coordinated changes on the same branch.
- **Validation tiers.** Graph-op-only changes validate against torch/HF at the op level. Changes touching the C++ runtime require E2E validation.

## Design

Three changes. No new abstractions.

### Change 1: Extract kernel setup into `kernels/` directory

Create `tensorrt_model_connect/tensorrt_model_connect/kernels/` — a flat directory of Python modules, one per kernel family. Each module has a `setup()` function that prepares the kernel (JIT compile, register, export `.so`).

**New file: `tensorrt_model_connect/tensorrt_model_connect/kernels/flashinfer_decode.py`**

```python
"""FlashInfer single-decode kernel — JIT compile, register, export .so."""

def setup(head_dim, dtype=None):
    """Prepare FlashInfer decode kernel for the given head_dim.

    Returns (kernel_name, so_path) where:
        kernel_name: TVM-FFI global function name for graph wiring
        so_path: path to exported .so for bundle packaging
    """
    import torch
    import tvm_ffi
    import flashinfer.decode as fi_dec

    if dtype is None:
        dtype = torch.float16

    mod = fi_dec.gen_single_decode_module(
        dtype, dtype, dtype, head_dim, head_dim,
        pos_encoding_mode=0,
        use_sliding_window=False,
        use_logits_soft_cap=False,
    ).build_and_load()

    name = f"flashinfer.decode_f16_d{head_dim}"
    tvm_ffi.register_global_func(name, mod.run, override=True)

    so_path = f"/tmp/{name.replace('.', '_')}.so"
    mod.export_library(so_path)

    return name, so_path
```

**Why a separate directory:**
- Heavy imports (flashinfer, tvm_ffi, torch) are isolated. Importing `graph_ops` doesn't pull in kernel dependencies.
- Each kernel is a new file. Agents adding kernels in parallel never touch the same file.
- An empty `__init__.py` makes it a package. Callers do `from ..kernels import flashinfer_decode`.

**What this is NOT:**
- Not a registry. No discovery mechanism. No base class or protocol.
- Not a framework. Each file is standalone. The perf agent can structure it however they want.

### Change 2: Extract attention implementations from `graph_blocks.py` into `graph_ops.py`

Move the two inline attention implementations (decomposed and FFI) from `graph_blocks.py` into named functions in `graph_ops.py`, alongside the 5 existing attention variants (`add_self_attention_block`, `add_self_attention_block_with_rope`, `add_windowed_self_attention_with_rope`, `add_cross_attention`, `add_self_attention_block` in `graph_blocks.add_vae_spatial_attention`).

**New functions in `graph_ops.py`:**

```python
def add_decoder_attention_decomposed(
    network, q, all_k, all_v, attention_mask, *,
    num_heads, head_dim, attention_window,
    attn_scale_tensor,
    alibi_slopes_tensor=None,
    alibi_indices_tensor=None,
    position_id=None,
):
    """Decomposed decoder attention: Q@K^T -> scale -> mask -> softmax -> @V.

    Inputs:
        q:              [1, attention_size]
        all_k, all_v:   [attention_window, attention_size]
        attention_mask:  [1, attention_window]
    Returns:
        context:        [1, attention_size]
    """
    # ~60 lines — extracted verbatim from graph_blocks.py lines 188-246
```

```python
def add_decoder_attention_ffi(
    network, q, all_k, all_v, *,
    kernel_name, num_heads, head_dim, attention_window,
):
    """Decoder attention via TVM-FFI kernel (FlashInfer, CuTe, etc).

    Same input/output contract as add_decoder_attention_decomposed.
    The kernel must be registered as a TVM-FFI global before engine build.
    """
    # ~30 lines — extracted from graph_blocks.py lines 154-187
    # Internally calls add_tvm_ffi_kernel()
```

**Changes to `graph_blocks.py`:**

The `add_attention_block()` function (lines 152-246) replaces the inline if/else with a single call:

```python
# Before (graph_blocks.py lines 152-246): ~95 lines of inline attention code
# After: one delegation call

if ffi_attention_kernel is not None:
    context_flat = graph_ops.add_decoder_attention_ffi(
        network, q, all_k.get_output(0), all_v.get_output(0),
        kernel_name=ffi_attention_kernel,
        num_heads=num_heads, head_dim=head_dim,
        attention_window=attention_window)
else:
    context_flat = graph_ops.add_decoder_attention_decomposed(
        network, q, all_k.get_output(0), all_v.get_output(0),
        attention_mask,
        num_heads=num_heads, head_dim=head_dim,
        attention_window=attention_window,
        attn_scale_tensor=attn_scale_tensor,
        alibi_slopes_tensor=alibi_slopes_tensor,
        alibi_indices_tensor=alibi_indices_tensor,
        position_id=position_id)
```

**Changes to `standard_decoder_builder.py`:**

Delete the env var mechanism (lines 111-112, 252-255):

```python
# DELETE: Force FP16 via env var
# if os.environ.get("TRTMC_FFI_ATTENTION_KERNEL"):
#     trt_config.set_flag(trt.BuilderFlag.FP16)

# DELETE: Env var lookup
# ffi_attention_kernel = os.environ.get("TRTMC_FFI_ATTENTION_KERNEL")
```

Kernel selection becomes explicit code. The perf agent controls it on their branch by modifying the call site in `graph_blocks.py` (or the builder, or the family plugin — wherever makes sense for their optimization).

**Input/output contract:**

The contract between `graph_blocks.py` and any attention implementation is:

| Tensor | Shape | Type |
|--------|-------|------|
| q (input) | `[1, attention_size]` | float32 |
| all_k (input) | `[attention_window, attention_size]` | float32 |
| all_v (input) | `[attention_window, attention_size]` | float32 |
| attention_mask (input) | `[1, attention_window]` | float32 |
| context (output) | `[1, attention_size]` | float32 |

Where `attention_size = num_heads * head_dim` and `attention_window = max_cache_length + 1`.

Any function that accepts these inputs and produces this output can be used as an attention implementation — decomposed TRT ops, FFI FlashInfer, FFI CuTe, or anything else. The perf agent is free to ignore the mask input, use different internal dtypes, reshape however they want — as long as the boundary shapes match.

This same pattern applies to MLP. If a perf agent wants to replace SwiGLU with a fused FFI kernel, the contract is:

| Tensor | Shape | Type |
|--------|-------|------|
| inp (input) | `[1, hidden_size]` | float32 |
| out (output) | `[1, hidden_size]` | float32 |

### Change 3: Bundle kernel `.so` in `.trtfb` for C++ runtime

**Problem:** The C++ runtime deserializes the TRT engine, which instantiates `TvmFfiKernelPlugin`. On the first `enqueue()` call, the plugin does `TVMFFIFunctionGetGlobal(kernel_name)`. This fails because no kernel is registered in the C++ process.

**Solution:** Package the compiled kernel `.so` files inside the `.trtfb` bundle. The C++ runtime extracts and loads them before deserializing the engine.

**Build-time changes (Python):**

The engine builder collects kernel artifacts during the build and passes them to the bundle writer.

`bundle_writer.py` — add kernel sections:

```python
# Existing sections: config.json, engine_plan, vocab.txt, etc.
# New sections:
#   kernel_manifest.json  — list of required kernels
#   kernel_<name>.so      — compiled kernel shared libraries

# kernel_manifest.json format:
# {
#     "kernels": [
#         {
#             "global_name": "flashinfer.decode_f16_d64",
#             "func_name": "run",
#             "section": "kernel_flashinfer_decode_f16_d64.so"
#         }
#     ]
# }
```

The `BundleSection` dataclass and `write_bundle()` function are unchanged — kernel `.so` files are just additional sections with binary data. The manifest is a JSON section. No format changes needed.

**Runtime changes (C++):**

Add a function to `plugin_helpers.h/.cpp` (or a new small file) that loads FFI kernels from the bundle:

```cpp
// Load all TVM-FFI kernels listed in the bundle's kernel_manifest.json.
// Must be called BEFORE deserializing any TRT engine that uses FFI plugins.
// No-op if the bundle has no kernel_manifest.json section (non-FFI bundles).
void load_ffi_kernels_from_bundle(const BundleFile& bundle);
```

Implementation (~30 lines):

```cpp
void load_ffi_kernels_from_bundle(const BundleFile& bundle)
{
    const auto* manifest_sec = find_section(bundle, "kernel_manifest.json");
    if (!manifest_sec) return;  // No FFI kernels in this bundle

    std::string manifest_str(manifest_sec->data.begin(), manifest_sec->data.end());
    // Parse JSON, iterate kernels
    // For each kernel:
    //   1. Find the .so section by name
    //   2. Write to a temp file
    //   3. Call load_tvm_ffi_module_func(tmp_path, func_name, global_name)
}
```

Each pipeline plugin (`decoder_plugin.cpp`, `encoder_plugin.cpp`, etc.) calls this before `load_trt_module_from_plan()`:

```cpp
// In decoder_plugin.cpp create():
load_ffi_kernels_from_bundle(ctx.bundle);  // NEW — one line
auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
```

**SM arch specificity:**

The kernel `.so` contains cubin compiled for the build GPU's SM architecture. The TRT engine plan is also SM-specific. Both are built on the same device and packaged in the same bundle. This is consistent — the entire `.trtfb` is device-specific. Cross-device deployment requires rebuilding the bundle on the target device.

**Bundles without FFI kernels:**

If `kernel_manifest.json` is absent, `load_ffi_kernels_from_bundle()` is a no-op. All existing bundles continue to work unchanged.

## Perf Agent Workflow

### Adding a new FFI kernel (e.g., CuTe fused attention)

On the perf agent's branch:

1. **Create `kernels/cute_fused_attention.py`** — new file, ~20 lines. JIT compile or load the kernel, register as TVM-FFI global, export `.so`.

2. **Add `add_decoder_attention_cute()` to `graph_ops.py`** — new function at end of file, ~40 lines. Reshapes inputs to the kernel's expected layout, calls `add_tvm_ffi_kernel()`, reshapes output back to contract shape.

3. **Change call site in `graph_blocks.py`** — one line change: call the new function instead of `add_decoder_attention_decomposed`.

4. **Validate** — run graph-op-level test against torch reference. If correct, run single-model E2E.

Files touched: 1 new file + 2 modified files (additive function + one-line call site change).

### Adding a non-attention kernel (e.g., fused SwiGLU MLP)

Same pattern:

1. Create `kernels/cute_fused_swiglu.py`
2. Add `add_swiglu_mlp_ffi()` to `graph_ops.py` alongside existing `add_swiglu_mlp` (which is currently in `graph_blocks.py` — it could stay there or move to graph_ops; the perf agent can do either on their branch)
3. Change call site in builder or graph_blocks
4. Validate

### Runtime co-design (e.g., paged KV cache)

On the perf agent's branch:

1. Modify `graph_ops.py` — change attention implementation to use paged KV inputs
2. Modify `standard_decoder_builder.py` — change input tensor shapes/names
3. Modify C++ runtime (`device_kv_cache.cpp`, `decoder_plugin.cpp`) — implement paged cache management
4. Validate with E2E tests (required for runtime changes)

This touches shared C++ files, which is fine — branch isolation handles it. The design doesn't try to prevent this; it just minimizes conflicts for the common case (graph-only changes).

### Composing optimizations from multiple agents

Agent A: FlashInfer attention (edits `graph_ops.py` + `graph_blocks.py` + adds `kernels/flashinfer_decode.py`)
Agent B: Fused SwiGLU MLP (edits `graph_ops.py` + adds `kernels/cute_swiglu.py`)

Composition: merge both branches. The `graph_ops.py` changes are additive functions (git auto-merges). The `graph_blocks.py` changes are separate call sites (attention vs MLP — no overlap). The kernel directories are new files (no conflict). Build and validate the composition with E2E.

## Scaling Properties

| Dimension | How it scales |
|-----------|--------------|
| **N agents adding N kernels in parallel** | Each adds 1 new file (kernels/) + 1 new function (graph_ops.py, additive). Git auto-merges additive changes to the same file. The only conflict is the call site selection (one line per kernel). |
| **Any model type** | `graph_ops.py` serves all builders. An attention kernel added here benefits decoders, encoders, DiT, vision, speech — every model type that calls attention. |
| **Any op type** | Same pattern works for attention, MLP, norm, conv, timestep embedding. The contract is: match the input/output tensor shapes. |
| **Multiple kernels per model** | The bundle manifest supports any number of kernel `.so` sections. A model can use FlashInfer for attention + CuTe for MLP + custom for norm. |
| **Architecture portability** | Kernel `.so` is SM-specific, same as the TRT engine plan. One bundle per target device. |
| **C++ runtime deployment** | Self-contained `.trtfb` — one file has engine plan + kernel `.so` files + manifest. No Python needed at runtime. |

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `tensorrt_model_connect/tensorrt_model_connect/kernels/flashinfer_decode.py` | New file | ~25 |
| `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` | Add `add_decoder_attention_decomposed()` + `add_decoder_attention_ffi()` | ~100 added |
| `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py` | Replace inline attention with delegation calls | ~95 removed, ~10 added |
| `tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py` | Remove env var mechanism | ~5 removed |
| `tensorrt_model_connect/tensorrt_model_connect/bundle_writer.py` | No structural change — callers pass kernel sections via existing `BundleSection` list | 0 |
| `tensorrt_model_connect/tensorrt_model_connect/engine_builder.py` | Collect kernel artifacts, add to sections list | ~15 |
| `src/runtime/plugins/shared/plugin_helpers.h` | Add `load_ffi_kernels_from_bundle()` declaration | ~5 |
| `src/runtime/plugins/shared/plugin_helpers.cpp` | Implement `load_ffi_kernels_from_bundle()` | ~30 |
| `src/runtime/plugins/decoder_plugin.cpp` | Call `load_ffi_kernels_from_bundle()` before engine load | 1 |
| Other pipeline plugins (encoder, hybrid, etc.) | Same one-line addition | 1 each |

**Total: ~180 lines added, ~100 lines removed. No new abstractions.**

## What This Design Does NOT Include

- **No kernel registry or discovery mechanism.** List the `kernels/` directory to see what's available.
- **No config file for kernel selection.** The perf agent changes code, not config.
- **No abstract base class for kernel implementations.** The contract is tensor shapes, not a type hierarchy.
- **No kernel caching layer.** JIT frameworks (FlashInfer, TVM) manage their own caches.
- **No graph_ops.py file split.** One file, ~2200 lines. The user confirmed file size is not a problem.
- **No per-kernel test infrastructure.** Use existing test tiers (op-level, graph-op GPU, E2E).
- **No multi-arch `.so` bundling.** One bundle per target device, consistent with TRT engine plan behavior.

## C++ Plugin Infrastructure (Unchanged)

The TVM-FFI bridge plugin (`src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_plugin.cpp`) is stable infrastructure. It:

- Accepts any TVM-FFI function by name via `kernel_name` plugin field
- Marshals TRT tensors to DLTensors for the TVM-FFI call
- Passes scalar extra_args (int, float, none, ptr) from the `shape_spec` JSON
- Manages workspace allocation via TRT's workspace pool
- Serializes/deserializes kernel_name + shape_spec (survives engine plan round-trip)

No changes to the plugin are required for this design. Future improvements (bfloat16 dtype support, dynamic output shapes, serialization versioning) are independent infra work, not per-kernel.

## Known Limitations

1. **Plugin dtype support is binary (fp16/fp32).** Kernels requiring int8, bfloat16, or fp8 need a plugin extension first. This is a one-time infra change, not a per-kernel concern.

2. **Output shapes are static or `same_as_input_N`.** Kernels whose output shape is a function of input shape (e.g., `output = 2 * input.d[0]`) need a plugin extension for symbolic dimension expressions.

3. **Thread safety:** `cached_fn_` in the plugin is lazily populated without a mutex. Benign in practice (same pointer written) but technically undefined behavior. Fix: `std::call_once`.

4. **The call site line in `graph_blocks.py` is the irreducible merge conflict.** When N agents each want their kernel to be the default, the integrator must choose. This is a decision, not a bug — no architecture eliminates it.
