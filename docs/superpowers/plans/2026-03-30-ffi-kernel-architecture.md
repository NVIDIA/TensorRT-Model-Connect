# FFI Kernel Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple FFI kernel wiring from graph structure, enable kernel `.so` bundling for C++ runtime deployment, and remove env var kernel selection — so perf agents can add arbitrary FFI kernels with minimal merge conflict.

**Architecture:** Three changes: (1) extract kernel setup into `kernels/` directory, (2) extract attention implementations from `graph_blocks.py` into `graph_ops.py`, (3) add kernel `.so` bundling in `.trtfb` with C++ runtime loading. No new abstractions.

**Tech Stack:** Python (TensorRT API, TVM-FFI), C++ (TRT plugin, TVM-FFI C ABI), `.trtfb` bundle format.

**Spec:** `docs/superpowers/specs/2026-03-30-ffi-kernel-architecture-design.md`

---

### Task 1: Create `kernels/` package with FlashInfer decode setup

**Files:**
- Create: `tensorrt_model_connect/tensorrt_model_connect/kernels/__init__.py`
- Create: `tensorrt_model_connect/tensorrt_model_connect/kernels/flashinfer_decode.py`

- [ ] **Step 1: Create empty `__init__.py`**

```python
# tensorrt_model_connect/tensorrt_model_connect/kernels/__init__.py
```

Create the file with no content (empty package marker).

- [ ] **Step 2: Create `flashinfer_decode.py`**

```python
"""FlashInfer single-decode kernel — JIT compile, register, export .so."""


def setup(head_dim, dtype=None):
    """Prepare FlashInfer decode kernel for the given head_dim.

    JIT-compiles the kernel for the current GPU, registers it as a TVM-FFI
    global function, and exports the compiled module as a .so file for
    bundle packaging.

    Returns (kernel_name, so_path) where:
        kernel_name: TVM-FFI global function name (e.g. "flashinfer.decode_f16_d64")
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

- [ ] **Step 3: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/kernels/__init__.py tensorrt_model_connect/tensorrt_model_connect/kernels/flashinfer_decode.py
git commit -m "feat(ffi): add kernels/ package with FlashInfer decode setup"
```

---

### Task 2: Extract decomposed decoder attention into `graph_ops.py`

**Files:**
- Modify: `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` (add function before the TVM-FFI section at line ~2025)
- Modify: `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py` (replace lines 188-246 with delegation call)

- [ ] **Step 1: Add `add_decoder_attention_decomposed()` to `graph_ops.py`**

Insert before the `# TVM-FFI kernel bridge` comment (line 2025):

```python
# ---------------------------------------------------------------------------
# Decoder attention implementations
# ---------------------------------------------------------------------------

def add_decoder_attention_decomposed(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    all_k: trt.ITensor,
    all_v: trt.ITensor,
    attention_mask: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    attention_window: int,
    attn_scale_tensor: trt.ITensor,
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
    position_id: trt.ITensor | None = None,
) -> trt.ITensor:
    """Decomposed decoder attention: Q@K^T -> scale -> mask -> softmax -> @V.

    Inputs:
        q:              [1, attention_size]  (attention_size = num_heads * head_dim)
        all_k, all_v:   [attention_window, attention_size]
        attention_mask:  [1, attention_window]
    Returns:
        context:        [1, attention_size]
    """
    attention_size = num_heads * head_dim

    q_heads = network.add_shuffle(q)
    q_heads.reshape_dims = (num_heads, 1, head_dim)

    k_heads = network.add_shuffle(all_k)
    k_heads.reshape_dims = (attention_window, num_heads, head_dim)
    v_heads = network.add_shuffle(all_v)
    v_heads.reshape_dims = (attention_window, num_heads, head_dim)

    k_heads.second_transpose = trt.Permutation([1, 0, 2])
    v_heads.second_transpose = trt.Permutation([1, 0, 2])

    score = network.add_matrix_multiply(
        q_heads.get_output(0), trt.MatrixOperation.NONE,
        k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)

    scaled = network.add_elementwise(
        score.get_output(0), attn_scale_tensor,
        trt.ElementWiseOperation.PROD)

    if alibi_slopes_tensor is not None and alibi_indices_tensor is not None:
        pos_float = network.add_identity(position_id)
        pos_float.set_output_type(0, trt.float32)
        pos_1d = network.add_shuffle(pos_float.get_output(0))
        pos_1d.reshape_dims = (1,)
        full_indices = network.add_concatenation(
            [alibi_indices_tensor, pos_1d.get_output(0)])
        full_indices.axis = 0
        idx_3d = network.add_shuffle(full_indices.get_output(0))
        idx_3d.reshape_dims = (1, 1, attention_window)
        pos_reshaped = network.add_shuffle(pos_float.get_output(0))
        pos_reshaped.reshape_dims = (1, 1, 1)
        rel_pos = network.add_elementwise(
            idx_3d.get_output(0), pos_reshaped.get_output(0),
            trt.ElementWiseOperation.SUB)
        alibi_bias = network.add_elementwise(
            alibi_slopes_tensor, rel_pos.get_output(0),
            trt.ElementWiseOperation.PROD)
        scaled = network.add_elementwise(
            scaled.get_output(0), alibi_bias.get_output(0),
            trt.ElementWiseOperation.SUM)

    mask3d = network.add_shuffle(attention_mask)
    mask3d.reshape_dims = (1, 1, attention_window)

    masked = network.add_elementwise(
        scaled.get_output(0), mask3d.get_output(0),
        trt.ElementWiseOperation.SUM)

    softmax = network.add_softmax(masked.get_output(0))
    softmax.axes = 1 << 2

    context_heads = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_heads.get_output(0), trt.MatrixOperation.NONE)

    context_flat = network.add_shuffle(context_heads.get_output(0))
    context_flat.reshape_dims = (1, attention_size)
    return context_flat.get_output(0)
```

- [ ] **Step 2: Replace the decomposed path in `graph_blocks.py`**

In `graph_blocks.py`, replace lines 188-246 (the `else:` block of the attention core) with a delegation call. The full replacement target is:

Replace the entire block from `    else:` through `        context_flat.reshape_dims = (1, attention_size)` with:

```python
    else:
        # Standard decomposed attention path
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

Note: the lines after this (`# Output projection`, `attn_out = ...`) reference `context_flat.get_output(0)`. Since the new function returns an `ITensor` directly (not a shuffle layer), change the output projection call from `context_flat.get_output(0)` to just `context_flat`:

```python
    # Output projection
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat,
        attention_size, hidden_size,
        weights[f"{prefix}.w_o"])
```

- [ ] **Step 3: Verify C++ unit tests still pass**

Run inside the container:
```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && ctest --test-dir build --output-on-failure -j4 2>&1 | tail -5"
```

Expected: all tests pass (C++ tests are unaffected by Python changes).

- [ ] **Step 4: Verify Python builder unit tests still pass**

Run inside the container:
```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py 2>&1 | tail -20"
```

Expected: all tests pass. The decomposed attention path is exercised by `test_standard_decoder.py` and `test_graph_blocks.py` if they exist.

- [ ] **Step 5: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/graph_ops.py tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py
git commit -m "refactor(ffi): extract decomposed decoder attention into graph_ops"
```

---

### Task 3: Extract FFI decoder attention into `graph_ops.py`

**Files:**
- Modify: `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` (add function after `add_decoder_attention_decomposed`)
- Modify: `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py` (replace lines 154-187 with delegation call)

- [ ] **Step 1: Add `add_decoder_attention_ffi()` to `graph_ops.py`**

Insert after `add_decoder_attention_decomposed()`, before the `# TVM-FFI kernel bridge` section:

```python
def add_decoder_attention_ffi(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    all_k: trt.ITensor,
    all_v: trt.ITensor,
    *,
    kernel_name: str,
    num_heads: int,
    head_dim: int,
    attention_window: int,
) -> trt.ITensor:
    """Decoder attention via TVM-FFI kernel (FlashInfer, CuTe, etc).

    Same input/output contract as add_decoder_attention_decomposed.
    The kernel must be registered as a TVM-FFI global before engine build.

    Inputs:
        q:              [1, attention_size]
        all_k, all_v:   [attention_window, attention_size]
    Returns:
        context:        [1, attention_size]
    """
    attention_size = num_heads * head_dim

    q_2d = network.add_shuffle(q)
    q_2d.reshape_dims = (num_heads, head_dim)
    k_3d = network.add_shuffle(all_k)
    k_3d.reshape_dims = (attention_window, num_heads, head_dim)
    v_3d = network.add_shuffle(all_v)
    v_3d.reshape_dims = (attention_window, num_heads, head_dim)

    scale_val = 1.0 / (head_dim ** 0.5)
    ffi_outputs = add_tvm_ffi_kernel(
        network,
        kernel_name=kernel_name,
        inputs=[q_2d.get_output(0), k_3d.get_output(0),
                v_3d.get_output(0)],
        output_specs=[{"dims": [num_heads, head_dim], "dtype": "float16"}],
        workspace_bytes=32 * 1024 * 1024,  # 32MB for FlashInfer tmp
        extra_args=[
            {"type": "none"},              # maybe_lse
            {"type": "int", "value": 0},    # kv_layout_code (NHD)
            {"type": "int", "value": -1},   # window_left
            {"type": "none"},              # alibi_slopes
            {"type": "float", "value": 0.0},     # logits_soft_cap
            {"type": "float", "value": scale_val}, # sm_scale
            {"type": "float", "value": 1.0},      # rope_rcp_scale
            {"type": "float", "value": 0.0001},   # rope_rcp_theta
        ],
    )
    context_flat = network.add_shuffle(ffi_outputs[0])
    context_flat.reshape_dims = (1, attention_size)
    return context_flat.get_output(0)
```

Note: this function calls `add_tvm_ffi_kernel` which is defined later in the same file. Python allows forward references within a module since function bodies are only executed at call time, not at definition time. No import needed.

- [ ] **Step 2: Replace the FFI path in `graph_blocks.py`**

Replace lines 154-187 (the `if ffi_attention_kernel is not None:` block) with:

```python
    if ffi_attention_kernel is not None:
        # Fused attention kernel via TVM-FFI plugin
        context_flat = graph_ops.add_decoder_attention_ffi(
            network, q, all_k.get_output(0), all_v.get_output(0),
            kernel_name=ffi_attention_kernel,
            num_heads=num_heads, head_dim=head_dim,
            attention_window=attention_window)
```

- [ ] **Step 3: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/graph_ops.py tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py
git commit -m "refactor(ffi): extract FFI decoder attention into graph_ops"
```

---

### Task 4: Remove env var kernel selection from `standard_decoder_builder.py`

**Files:**
- Modify: `tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py`

- [ ] **Step 1: Remove the FP16 force-enable block (lines 110-112)**

Delete these lines:

```python
    # Enable FP16 if using FFI attention (FlashInfer requires fp16)
    if os.environ.get("TRTMC_FFI_ATTENTION_KERNEL"):
        trt_config.set_flag(trt.BuilderFlag.FP16)
```

- [ ] **Step 2: Remove the env var lookup (lines 249-255)**

Replace:

```python
    # ---------------------------------------------------------------
    # Optional: TVM-FFI attention kernel (FlashInfer, etc.)
    # ---------------------------------------------------------------
    ffi_attention_kernel = os.environ.get("TRTMC_FFI_ATTENTION_KERNEL")
    if ffi_attention_kernel:
        print(f"[standard_decoder_builder] Using FFI attention: {ffi_attention_kernel}",
              file=sys.stderr)
```

With:

```python
    # FFI attention kernel: set by the perf agent on their branch.
    # Default: None (use decomposed attention).
    ffi_attention_kernel = None
```

- [ ] **Step 3: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py
git commit -m "refactor(ffi): remove env var kernel selection, make explicit code"
```

---

### Task 5: Add kernel `.so` bundling in Python builder

**Files:**
- Modify: `tensorrt_model_connect/tensorrt_model_connect/engine_builder.py` (~line 509, before `write_bundle`)

- [ ] **Step 1: Add kernel artifact collection to `build_bundle()`**

In `engine_builder.py`, locate the `write_bundle(output_path, info, sections)` call at line 509. Before that call, add kernel manifest + `.so` section packaging. The build function needs a way to receive kernel artifacts. Add a `kernel_artifacts` parameter to `build_bundle()` and wire it through.

Find the `build_bundle` function signature and add the parameter:

```python
def build_bundle(
    model_dir: str | Path,
    output_path: str | Path,
    max_cache_length: int = 256,
    verbose: bool = False,
    kernel_artifacts: list[tuple[str, str]] | None = None,  # NEW: [(global_name, so_path)]
) -> None:
```

Then before `write_bundle(output_path, info, sections)` (line 509), insert:

```python
    # Package FFI kernel .so files into the bundle
    if kernel_artifacts:
        import json as _json
        manifest_entries = []
        for global_name, so_path in kernel_artifacts:
            section_name = f"kernel_{global_name.replace('.', '_')}.so"
            so_data = Path(so_path).read_bytes()
            sections.append(BundleSection(section_name, so_data))
            manifest_entries.append({
                "global_name": global_name,
                "func_name": "run",
                "section": section_name,
            })
        manifest_json = _json.dumps({"kernels": manifest_entries}).encode("utf-8")
        sections.append(BundleSection("kernel_manifest.json", manifest_json))
```

- [ ] **Step 2: Verify existing tests still pass**

Run inside the container:
```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/builder/test_bundle_writer.py -v 2>&1 | tail -10"
```

Expected: all pass (we added an optional parameter with a default of None — no existing callers are affected).

- [ ] **Step 3: Commit**

```bash
git add tensorrt_model_connect/tensorrt_model_connect/engine_builder.py
git commit -m "feat(ffi): add kernel .so bundling support to engine builder"
```

---

### Task 6: Add C++ kernel loading from bundle

**Files:**
- Modify: `src/runtime/plugins/shared/plugin_helpers.h` (add declaration)
- Modify: `src/runtime/plugins/shared/plugin_helpers.cpp` (add implementation)

- [ ] **Step 1: Add declaration to `plugin_helpers.h`**

Inside the `#if TRTMC_HAS_TRT` block, before `#endif`, add:

```cpp
// Load all TVM-FFI kernels listed in the bundle's kernel_manifest.json.
// Must be called BEFORE deserializing any TRT engine that uses FFI plugins.
// No-op if the bundle has no kernel_manifest.json section (non-FFI bundles).
void load_ffi_kernels_from_bundle(const BundleFile& bundle);
```

- [ ] **Step 2: Add implementation to `plugin_helpers.cpp`**

Find the end of the existing implementations in `plugin_helpers.cpp`. Add:

```cpp
void load_ffi_kernels_from_bundle(const BundleFile& bundle)
{
#if TRTMC_HAS_TVM_FFI
    const auto* manifest_sec = find_section(bundle, "kernel_manifest.json");
    if (!manifest_sec) return;

    std::string manifest_str(manifest_sec->data.begin(), manifest_sec->data.end());

    // Simple JSON array iteration — find each kernel entry and load its .so
    // Format: {"kernels": [{"global_name": "...", "func_name": "...", "section": "..."}]}
    auto pos = manifest_str.find("\"kernels\"");
    if (pos == std::string::npos) return;

    // Iterate kernel objects
    auto arr_start = manifest_str.find('[', pos);
    auto arr_end = manifest_str.find(']', arr_start);
    if (arr_start == std::string::npos || arr_end == std::string::npos) return;

    auto cur = arr_start + 1;
    while (cur < arr_end) {
        auto obj_start = manifest_str.find('{', cur);
        if (obj_start == std::string::npos || obj_start >= arr_end) break;
        auto obj_end = manifest_str.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        std::string obj = manifest_str.substr(obj_start, obj_end - obj_start + 1);

        std::string global_name = extract_json_string(obj, "global_name", "");
        std::string func_name = extract_json_string(obj, "func_name", "run");
        std::string section_name = extract_json_string(obj, "section", "");

        if (global_name.empty() || section_name.empty()) {
            cur = obj_end + 1;
            continue;
        }

        const auto* so_sec = find_section(bundle, section_name);
        if (!so_sec || so_sec->data.empty()) {
            std::cerr << "[ffi] Kernel .so section not found: " << section_name << '\n';
            cur = obj_end + 1;
            continue;
        }

        // Write .so to temp file and load via TVM-FFI module loader
        // Build safe temp path: replace dots in global_name with underscores
        std::string safe_name = global_name;
        for (auto& c : safe_name) { if (c == '.') c = '_'; }
        std::string tmp_path = "/tmp/trtmc_kernel_" + safe_name + ".so";
        {
            std::ofstream ofs(tmp_path, std::ios::binary);
            ofs.write(so_sec->data.data(), static_cast<std::streamsize>(so_sec->data.size()));
        }

        if (load_tvm_ffi_module_func(tmp_path, func_name, global_name)) {
            std::cerr << "[ffi] Loaded kernel: " << global_name << '\n';
        } else {
            std::cerr << "[ffi] Failed to load kernel: " << global_name
                      << " from " << section_name << '\n';
        }

        cur = obj_end + 1;
    }
#else
    (void)bundle;
#endif
}
```

Add the required includes at the top of plugin_helpers.cpp:

```cpp
#include <fstream>
```

And conditionally include the module loader:

```cpp
#if TRTMC_HAS_TVM_FFI
#include "runtime/plugins/tvm_ffi/tvm_ffi_module_loader.h"
#endif
```

Also add `#include "utils/json_helpers.h"` if not already included.

- [ ] **Step 3: Rebuild C++ runtime**

```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && cmake --build build -j 2>&1 | tail -5"
```

Expected: builds successfully.

- [ ] **Step 4: Verify C++ unit tests still pass**

```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && ctest --test-dir build --output-on-failure -j4 2>&1 | tail -5"
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/plugins/shared/plugin_helpers.h src/runtime/plugins/shared/plugin_helpers.cpp
git commit -m "feat(ffi): add load_ffi_kernels_from_bundle() for C++ runtime"
```

---

### Task 7: Wire kernel loading into pipeline plugins

**Files:**
- Modify: `src/runtime/plugins/decoder_plugin.cpp`
- Modify: `src/runtime/plugins/encoder_plugin.cpp`
- Modify: other pipeline plugins that call `load_trt_module_from_plan`

- [ ] **Step 1: Add `load_ffi_kernels_from_bundle()` call to `decoder_plugin.cpp`**

In `decoder_plugin.cpp`, add the call as the first line of `create()`:

```cpp
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);  // Load FFI kernels before engine
        auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
```

- [ ] **Step 2: Add the same call to all other pipeline plugins**

Add `load_ffi_kernels_from_bundle(ctx.bundle);` as the first line of `create()` in each plugin that calls `load_trt_module_from_plan`. Find them:

```bash
grep -rn 'load_trt_module_from_plan' src/runtime/plugins/ --include='*.cpp' -l
```

For each file found, add the call before the first `load_trt_module_from_plan`. This is safe because `load_ffi_kernels_from_bundle` is a no-op when no `kernel_manifest.json` section exists.

- [ ] **Step 3: Rebuild and test**

```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && cmake --build build -j 2>&1 | tail -5"
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && ctest --test-dir build --output-on-failure -j4 2>&1 | tail -5"
```

Expected: builds and all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/runtime/plugins/
git commit -m "feat(ffi): wire kernel loading into all pipeline plugins"
```

---

### Task 8: End-to-end validation

**Files:**
- Modify: `tests/test_qwen3_flashinfer_e2e.py` (update to use new architecture)

- [ ] **Step 1: Update the Qwen3 FlashInfer E2E test**

The test at `tests/test_qwen3_flashinfer_e2e.py` currently uses the env var. Update it to use the new `kernels/` setup and explicit kernel selection. The key changes:

1. Replace manual FlashInfer JIT + registration with `from tensorrt_model_connect.kernels import flashinfer_decode`
2. Remove `os.environ["TRTMC_FFI_ATTENTION_KERNEL"]` — instead, temporarily patch the builder to use the FFI kernel

For a quick E2E validation, the simplest approach is to directly test the extracted functions:

```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && python3 -c \"
from tensorrt_model_connect.kernels import flashinfer_decode
name, so_path = flashinfer_decode.setup(head_dim=64)
print(f'Kernel: {name}')
print(f'SO: {so_path}')
import os
print(f'SO exists: {os.path.exists(so_path)}')
print(f'SO size: {os.path.getsize(so_path)} bytes')
\""
```

Expected: prints kernel name, `.so` path, confirms file exists with non-zero size.

- [ ] **Step 2: Run the FlashInfer plugin E2E test**

This test (`tests/test_flashinfer_plugin_e2e.py`) tests the C++ plugin directly and doesn't use the env var or the builder, so it should still pass:

```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && python3 tests/test_flashinfer_plugin_e2e.py 2>&1"
```

Expected: `Correctness: max_diff=0.000000 PASS`

- [ ] **Step 3: Run the full builder unit test suite**

```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py 2>&1 | tail -20"
```

Expected: all pass.

- [ ] **Step 4: Run the E2E smoke test (Qwen3-0.6B, baseline decomposed attention)**

Verify the default decomposed path still works end-to-end:

```bash
docker exec trtmc-dev-gb300-agent-2 bash -c "cd /workspace/tensorrt-model-connect && /opt/venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python --rebuild-engines 2>&1 | tail -20"
```

Expected: PASS — the decomposed attention path builds and runs correctly through the refactored code.

- [ ] **Step 5: Commit any test updates**

```bash
git add tests/
git commit -m "test(ffi): update E2E tests for refactored FFI architecture"
```
