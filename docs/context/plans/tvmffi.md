# TASK-05: TVM-FFI Kernel Bridge — Universal TRT Plugin for External Kernels

## Branch: `agent-X-tvm-ffi-bridge`

## Goal

Build a single `IPluginV3` TRT plugin (`TvmFfiKernelPlugin`) that bridges the
TVM-FFI ABI, allowing any TVM-FFI-compatible kernel library (FlashInfer,
CuTe DSL, CUDA Tile, hand-written CUDA) to be called from inside a TRT engine
graph. This replaces per-kernel plugin boilerplate with one generic bridge.

## Motivation

- **Performance**: FlashInfer fused attention kernels eliminate multi-layer TRT
  attention composition overhead (fewer memory round-trips, better occupancy).
- **Cross-platform**: CUDA Tile kernels abstract over tensor cores across
  Ampere → Blackwell → Rubin without per-arch tuning.
- **Ecosystem access**: TVM-FFI is the converging ABI for FlashInfer (MLSys 2025
  best paper, NVIDIA-backed) and CUTLASS 4 CuTe DSL. One bridge = all kernels.
- **Maintainability**: No per-kernel IPluginV3 implementations. Add a kernel by
  registering it in the TVM-FFI global registry + wiring it in `graph_ops.py`.

## Architecture

```
TRT Engine Graph (built by graph_ops.py / graph_blocks.py)
  │
  └─ IPluginV3: TvmFfiKernelPlugin("flashinfer.decode_attention")
       │
       └─ tvm_ffi::Function lookup (global registry, cached after first call)
            │
            ├── FlashInfer kernels     (pip install flashinfer; ships with tvm-ffi)
            ├── CuTe DSL kernels       (cute.compile(..., options={"tvm_ffi": True}))
            ├── CUDA Tile / cuTile      (cuTile Python → tvm-ffi target)
            ├── Hand-written CUDA       (TVM_FFI_DLL_EXPORT_TYPED_FUNC)
            └── Future tvm-ffi libraries
```

### Data flow inside `enqueue()`

```
TRT void* device pointers + known shapes/dtypes
  → wrap as DLTensor structs (zero-copy, ~50ns each)
  → TVMFFIEnvSetStream(kDLCUDA, device_id, stream)
  → cached_fn_(dl_in_0, dl_in_1, ..., dl_out_0)
  → return 0
```

---

## Phase 1: Plugin Shell + Dummy Kernel Round-Trip

### Deliverables

- [ ] `src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_plugin.h` — IPluginV3 + Core + Build + Runtime
- [ ] `src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_plugin.cpp` — implementation
- [ ] `src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_creator.cpp` — IPluginCreatorV3One registration
- [ ] CMake: find/link `libtvm_ffi`, compile plugin sources, register plugin .so
- [ ] `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py`: `add_tvm_ffi_kernel()` helper
- [ ] `tests/cpp/test_tvm_ffi_plugin.cpp` — dummy kernel round-trip through TRT engine

### IPluginV3 Bridge Design

```cpp
class TvmFfiKernelPlugin : public IPluginV3,
                            public IPluginV3OneCore,
                            public IPluginV3OneBuild,
                            public IPluginV3OneRuntime {
    // --- State (serialized) ---
    std::string kernel_name_;   // TVM-FFI global function name
    std::string shape_spec_;    // JSON: I/O count, dims, dtypes

    // --- Runtime cache (not serialized) ---
    tvm::ffi::Function cached_fn_;  // Resolved on first enqueue()

    // --- Core ---
    const char* getPluginName() override { return "TvmFfiKernel"; }
    const char* getPluginVersion() override { return "1"; }

    // --- Build ---
    int32_t getNbOutputs() override;                    // from shape_spec_
    DimsExprs getOutputDimensions(...) override;         // from shape_spec_
    DataType getOutputDataType(...) override;            // from shape_spec_
    bool supportsFormatCombination(...) override;        // linear FP16/FP32

    // --- Runtime ---
    int32_t enqueue(..., cudaStream_t stream) override;  // DLTensor wrap + call
    size_t getWorkspaceSize(...) override;               // from shape_spec_ or 0

    // --- Serialization ---
    size_t getSerializationSize() override;
    void serialize(void* buffer) override;               // write kernel_name_ + shape_spec_
};
```

### Python-Side Helper (`graph_ops.py`)

```python
def add_tvm_ffi_kernel(network, kernel_name, inputs, output_specs):
    """Insert a TVM-FFI kernel as a TRT plugin layer.

    Args:
        network: TRT INetworkDefinition
        kernel_name: TVM-FFI global function name (e.g. "flashinfer.decode_attention")
        inputs: list of ITensor
        output_specs: list of {"dims": [...], "dtype": trt.DataType}

    Returns:
        list of ITensor outputs
    """
    creator = trt.get_plugin_registry().get_creator("TvmFfiKernel", "1")
    shape_spec = json.dumps({
        "num_inputs": len(inputs),
        "num_outputs": len(output_specs),
        "outputs": output_specs,
    })
    fields = [
        trt.PluginField("kernel_name", kernel_name.encode(), trt.PluginFieldType.CHAR),
        trt.PluginField("shape_spec", shape_spec.encode(), trt.PluginFieldType.CHAR),
    ]
    plugin = creator.create_plugin("TvmFfiKernel", trt.PluginFieldCollection(fields))
    layer = network.add_plugin_v2(inputs, plugin)
    return [layer.get_output(i) for i in range(len(output_specs))]
```

### Unit Test: Dummy Kernel Round-Trip

Register a trivial `tvm_ffi_test.add_one` kernel that adds 1.0 to each element.
Build a TRT engine with: `input → TvmFfiKernelPlugin("tvm_ffi_test.add_one") → output`.
Run inference, verify `output == input + 1.0`.

### CMake Changes

```cmake
# --- TVM-FFI dependency ---
find_path(TVM_FFI_INCLUDE_DIR tvm/ffi/c_api.h
  HINTS /opt/venv/include ${TVM_FFI_ROOT}/include)
find_library(TVM_FFI_LIBRARY tvm_ffi
  HINTS /opt/venv/lib ${TVM_FFI_ROOT}/lib)

if(TVM_FFI_INCLUDE_DIR AND TVM_FFI_LIBRARY)
  target_sources(trtmc_core PRIVATE
    src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_plugin.cpp
    src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_creator.cpp)
  target_include_directories(trtmc_core PRIVATE ${TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtmc_core PRIVATE ${TVM_FFI_LIBRARY})
  target_compile_definitions(trtmc_core PRIVATE TRTMC_HAS_TVM_FFI=1)
endif()
```

### Acceptance Criteria

- Dummy kernel round-trip passes (build engine → run → verify output).
- Plugin serializes/deserializes correctly (engine save → load → run).
- `TRTMC_HAS_TVM_FFI=0` builds cleanly without tvm-ffi installed.

---

## Phase 2: FlashInfer Attention

### Deliverables

- [ ] Container setup: `pip install flashinfer` (brings tvm-ffi dependency)
- [ ] `tensorrt_model_connect/tensorrt_model_connect/ffi_kernels/flashinfer_attention.py` — register
      FlashInfer decode + prefill attention as TVM-FFI functions with shape specs
- [ ] `graph_ops.py`: `add_fused_attention()` that uses `add_tvm_ffi_kernel()`
      instead of the current multi-layer attention composition
- [ ] `graph_blocks.py`: opt-in flag `use_flashinfer_attention=True` on
      `add_attention_block()`
- [ ] E2E validation: `test_e2e[qwen3-0.6b]` with FlashInfer attention
- [ ] Benchmark: native TRT attention vs FlashInfer plugin (prefill + decode latency)

### Target Kernels

| Kernel | TVM-FFI Name | Replaces | Expected Win |
|--------|-------------|----------|-------------|
| Decode attention | `flashinfer.single_decode_with_kv_cache` | Multi-layer Q×K^T → scale → mask → softmax → ×V | Fused, memory-efficient |
| Prefill attention | `flashinfer.single_prefill_with_kv_cache` | Same composition for seq_len > 1 | FlashAttention-class speedup |

### KV Cache Integration

FlashInfer expects contiguous KV cache tensors. The existing `DeviceKvCache`
already stores contiguous per-layer K/V buffers on device — compatible layout.
The plugin receives cache pointers as additional inputs alongside Q.

### Acceptance Criteria

- Qwen3-0.6B E2E passes with FlashInfer attention (logit cosine ≥ 0.999).
- Decode latency improvement measurable on GB300 (target: ≥ 10% on decode step).
- Prefill latency improvement on seq_len ≥ 128 (target: ≥ 20%).
- Fallback: if `flashinfer` not installed, `graph_ops.py` uses native TRT attention.

---

## Phase 3: CuTe DSL Fused Ops

### Deliverables

- [ ] Container setup: `pip install nvidia-cutlass-dsl` (CUTLASS 4.3+)
- [ ] `tensorrt_model_connect/tensorrt_model_connect/ffi_kernels/fused_rmsnorm.py` — CuTe DSL fused
      RMSNorm + residual add, compiled with `compile_with_tvm_ffi`
- [ ] `tensorrt_model_connect/tensorrt_model_connect/ffi_kernels/fused_swiglu.py` — CuTe DSL fused
      SwiGLU (gate_proj + up_proj + silu + elementwise mul)
- [ ] `tensorrt_model_connect/tensorrt_model_connect/ffi_kernels/fused_rope.py` — CuTe DSL fused
      RoPE application (sin/cos bake + rotate)
- [ ] `graph_blocks.py`: opt-in `use_cute_fused_ops=True` for fused paths
- [ ] Unit tests: each fused op vs PyTorch/NumPy reference
- [ ] E2E validation: `test_e2e[qwen3-0.6b]` with all fused ops enabled

### Kernel Specs

| Kernel | Replaces in graph_ops.py | TRT Layers Eliminated |
|--------|-------------------------|----------------------|
| Fused RMSNorm + residual | `add_rms_norm()` (5+ layers: sub, pow, reduce, mul, add) + residual add | ~6 layers → 1 plugin |
| Fused SwiGLU | `add_swiglu_mlp()` (2 matmuls + silu + mul) | Element-wise portion: 3 layers → 1 plugin |
| Fused RoPE | `add_rotary_embedding()` (~10 layers: slice, concat, mul, add) | ~10 layers → 1 plugin |

### CuTe DSL → TVM-FFI Compilation Pattern

```python
import cute
from cute import compile as cute_compile

@cute_compile(options={"tvm_ffi": True})
def fused_rmsnorm_residual(x, residual, weight, eps, out):
    # CuTe DSL kernel body
    # Each thread block handles a row
    # 1. Compute x + residual
    # 2. Compute RMS norm of sum
    # 3. Scale by weight
    # 4. Write to out
    ...

# Kernel automatically registered in TVM-FFI global registry
# Callable as: tvm_ffi.Function.get_global("fused_rmsnorm_residual")(...)
```

### Acceptance Criteria

- Each fused op matches PyTorch reference within atol=1e-3.
- E2E Qwen3-0.6B passes with all fused ops (logit cosine ≥ 0.999).
- Per-op latency improvement measurable (target: ≥ 15% per fused block).
- Fallback: if `nvidia-cutlass-dsl` not installed, uses native TRT composition.

---

## Phase 4: Cross-Platform + CUDA Tile

### Deliverables

- [ ] `tensorrt_model_connect/tensorrt_model_connect/ffi_kernels/tile_attention.py` — CUDA Tile
      attention kernel (cuTile Python), portable across Ampere/Ada/Blackwell/Rubin
- [ ] Arch-dispatch convention: `kernel_name.sm90` / `kernel_name.sm100` with
      fallback to generic
- [ ] Benchmark matrix: FlashInfer vs CuTe DSL vs CUDA Tile attention on
      Ampere (A100), Blackwell (GB300), and future arch if available
- [ ] Documentation: how to register a new external kernel (3-step guide)

### Kernel Name Convention for Arch Dispatch

```
trtmc.attention.decode          → generic (any arch)
trtmc.attention.decode.sm90     → Blackwell-optimized
trtmc.attention.decode.sm100    → Rubin-optimized
```

Plugin `enqueue()` resolves: try `name.smXX` first, fall back to `name`.

### Acceptance Criteria

- Same plugin bridge works for CUDA Tile kernels (no plugin code changes).
- Arch-specific dispatch demonstrated on GB300.
- Adding a new kernel documented as: (1) write kernel, (2) register in TVM-FFI,
  (3) call `add_tvm_ffi_kernel()` in graph_ops — no C++ changes.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Plugin overhead exceeds fusion gains (small ops) | Medium | Low | Start with attention (Phase 2) where ROI is guaranteed; benchmark before expanding |
| FlashInfer KV cache layout mismatch | Low | Medium | Existing DeviceKvCache is contiguous per-layer — compatible; add transpose if needed |
| TVM-FFI ABI instability (v0.x) | Low | Medium | Pin version; ABI is minimal + stable by design (DLPack core) |
| CuTe DSL tile config suboptimal on GB300 | Medium | Low | Use CuTe DSL autotuning; compare against FlashInfer as baseline |
| TRT version incompatibility with IPluginV3 | Low | High | Target TRT 10.x (already in container); IPluginV3 is the only non-deprecated API |

## Dependencies

- `apache-tvm-ffi` (pip, Apache 2.0) — required for all phases
- `flashinfer` (pip, Apache 2.0) — Phase 2
- `nvidia-cutlass-dsl` (pip) — Phase 3
- CUDA 13.1+ for CUDA Tile — Phase 4
- TensorRT 10.x with IPluginV3 — already in container

## Estimated Effort

| Phase | Effort | Prereqs |
|-------|--------|---------|
| Phase 1: Plugin shell | 2–3 days | None |
| Phase 2: FlashInfer attention | 3–5 days | Phase 1 |
| Phase 3: CuTe DSL fused ops | 3–5 days | Phase 1 |
| Phase 4: CUDA Tile + cross-platform | 3–5 days | Phase 1 |

Phases 2, 3, and 4 are independent after Phase 1 and can be parallelized.
