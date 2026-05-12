# FLUX.2-dev Performance Tuning Worklog

## Goal
- Image quality matching FP32 master ("a beautiful cat on windowsill at sunset")
- Beat torch.compile perf: 0.185s/step, 5.2s total for 28 steps

## Final Result

| Config | Per Step | 28 Steps | Engine | Quality |
|--------|---------|----------|--------|---------|
| torch.compile (bf16) | 185ms | 5.2s | — | Good |
| BF16 ONNX (monolithic Myelin) | 185ms | 5.2s | 31GB | Good |
| BF16 API (STRONGLY_TYPED) | 185ms | 5.2s | 65GB | Good |
| **FP8+BF16 API (final)** | **117ms** | **3.3s** | **46GB** | **Good** |
| **FP8+BF16 C++ E2E** | **133ms** | **3.7s** | — | **Good** |

**1.58x faster than torch.compile** with matching image quality.

## Architecture
- FLUX.2-dev DiT: 8 joint + 48 single transformer blocks
- dim=6144, 48 heads, head_dim=128, MLP ratio 3.0, 4D RoPE
- Engine I/O: hidden_states [4096,128], encoder_hidden_states [512,6144], temb [6144], rotary_cos/sin [4608,128] → output [4096,128]
- Baked x_embedder: raw patches [4096,128] → matmul in TRT engine → [4096,6144]
- Context embedder: CPU-side projection (15360→6144)

---

## Phase 1: BF16 Engine (Exp 1-17)

### Problem: Gray noise from all precision configurations
Every engine on the branch produced gray noise — FP32, FP16, BF16, TF32, STRONGLY_TYPED or not.

**Root cause**: Stale C++ binary not rebuilt after baking code changes. After `cmake --build build`, all engines produced correct images.

### Key results (BF16 API path)

| # | Config | Per Step | Quality |
|---|--------|---------|---------|
| 15 | STRONGLY_TYPED + BF16, non-baked | 243ms | Good |
| 16 | STRONGLY_TYPED + BF16, baked | 367ms | Good |
| 17 | + CUDA graph | 244ms | Good |

**Bottleneck**: 244ms GPU compute floor vs 185ms torch.compile target. The API-built BF16 engine uses explicit Q@K^T + scale + softmax + @V matmuls for attention, which TRT cannot fuse into FlashAttention kernels.

### ONNX path comparison (Exp 19)
- ONNX parser auto-fuses SDPA → FlashAttention → **185ms/step** (matching torch.compile)
- Compiles into 1 monolithic Myelin layer (828MB activation)
- API path compiles into multiple layers (no FMHA fusion) → 244ms

---

## Phase 2: FP8 Quantization Journey

### ModelOpt calibration
- `nvidia-modelopt` v0.42.0 with `FP8_DEFAULT_CFG`
- Quantizes **both weights AND activations** (not weights-only) of all `nn.Linear` layers
- Per-tensor scales (axis=None), FP8 E4M3 format, max calibration
- 202 linear layers quantized, 10 excluded (embedders, norms, modulations)
- Attention BMMs (Q@K^T, softmax@V) are **NOT quantized** by default
- Scale formula: `scale = amax / 448.0`
- Script: `scripts/_quantize_fp8.py`

### Attempt 1: API path with FP32 intermediates (FAILED)
- STRONGLY_TYPED + FP8 Q/DQ on linear layers, DQ output type = FP32
- **Result: 283ms/step, 12.9GB activation, 925 Myelin layers**
- Root cause: FP32 intermediate tensors between FP8 layers bloat activation memory 15x and explicit BF16 Cast nodes fragment Myelin compilation into 925 separate kernels

### Attempt 2: ONNX path with FP8 Q/DQ injection (FAILED)
- Injected FP8 Q/DQ into working BF16 ONNX via proto-level manipulation
- STRONGLY_TYPED: 249ms/step, 12.9GB activation, 1 Myelin node but unfused internally
- Non-STRONGLY_TYPED: FP8 ignored ("invalid precision FP8"), runs as BF16
- Root cause: same 12.9GB activation bloat — Myelin cannot efficiently fuse FP8 Q/DQ regardless of graph partitioning

### Deep dive: Myelin source code analysis
Examined TRT optimizer and Myelin compiler source code:

**`qdqGraphOptimizer.cpp`** — TRT's Q/DQ graph optimizer:
- `ConstQDQInitializersFusion` constant-folds Q→DQ on weights
- `QuantizeDoubleInputNodes` fuses DQ+DQ→MatMul into quantized FC
- Hopper+ FP8 MatMul requires TN layout (opA=NONE, opB=TRANSPOSE)
- STRONGLY_TYPED uses `typeConstraint` instead of `precision`

**`myelinNodesClusterer.cpp`** — Backend partitioning:
- Assigns each node `ChunkType`: kMYELIN, kTRT, or kBOTH
- Consecutive same-backend nodes become one ForeignNode (subgraph)
- Cast nodes are `kBOTH` — they don't create hard boundaries
- BF16 ONNX: all ops → kMYELIN → 1 ForeignNode → 1 monolithic kernel
- FP8 API with Casts: Cast nodes create transition points → 925 ForeignNodes

**`quantize_ppg.cpp`** — Myelin's internal Q/DQ fusion:
- `dequantize_fc()` (line 2670): fuses DQ(A8) + DQ(W8) → MatMul into FP8 FC
- Per-tensor scales: folded into FC alpha (efficient)
- "scale operand is not a tensor" warnings are benign (checking for double-DQ in FP4)
- 195 FC fusions succeed out of 202 linear layers
- 120 attention matmuls fail (no Q/DQ) — expected, ModelOpt doesn't quantize attention
- 64 SwiGLU mul operations: Q backward propagation blocked

### The breakthrough: BF16 base type + FP8 linear + BF16 FMHA

**Key insight**: Set `_ALL_BF16 = True` even in FP8 mode. The entire network runs in BF16 as the base type. Only calibrated linear layers get FP8 Q/DQ (with DQ output type = BF16, not FP32). Attention uses TRT's `add_attention` API for FMHA fusion.

This prevents:
1. FP32 intermediates that bloat activation 15x
2. Cast nodes that fragment Myelin compilation
3. Type boundaries that prevent memory reuse

### Three bugs fixed to achieve correct FP8 output

**Bug 1: FP8 scale key mismatch**
- The scales JSON used ONNX node names (`node_linear_10`) but the builder looks up HF layer names (`transformer_blocks.0.attn.to_q`)
- Zero scales matched → engine was pure BF16, not FP8
- Fix: extract scales from ModelOpt checkpoint with HF names (`amax / 448.0`)

**Bug 2: Weight not divided by scale before FP8 cast**
- Code did `w.astype(float8_e4m3fn)` instead of `(w / scale).astype(float8_e4m3fn)`
- TRT's DQ multiplies by scale: `recovered = fp8_val * scale`
- Without dividing first: `recovered = float8(0.14) * 0.000299 = 0.000042` (3000x too small)
- This caused velocity explosion ([-348, 368] instead of [-7, 7])

**Bug 3: `add_attention` doesn't apply `1/sqrt(head_dim)` scaling**
- `add_attention(Q, K, V, SOFTMAX, False)` computes `softmax(Q@K^T) @ V` without the `1/sqrt(d)` factor
- Outputs were 4x too large (std ratio 4.11 vs explicit SDPA)
- Fix: pre-scale Q by `1/sqrt(head_dim)` before passing to `add_attention`
- Verified with minimal repro: pre-scaled Q gives cosine=0.99999 vs explicit SDPA

### Final FP8+BF16 engine results

| Metric | BF16 baseline | FP8+BF16 |
|--------|--------------|----------|
| GPU per-step | 185ms | **117ms** |
| 28 steps GPU | 5.2s | **3.3s** |
| C++ E2E per-step | — | **133ms** |
| C++ E2E 28 steps | — | **3.7s** |
| Activation memory | 828MB | 1,158MB |
| Engine size | 65GB | 46GB |
| Myelin nodes | 1 | 1 |
| Speedup | 1.0x | **1.58x** |

---

## Phase 3: GPU-Resident Denoising Loop

### Problem: TrtModule::forward() overhead
- `forward()` does H2D upload → enqueue → D2H download per call
- For 28 denoising steps: 28 × (upload + download + sync) = ~290s total
- Raw GPU compute: 28 × 117ms = 3.3s

### Solution: Direct GPU buffer management
- Pre-allocate GPU buffers via `TrtModule::device_ptr()`
- Upload static inputs (encoder_hidden, RoPE cos/sin) once
- Per step: upload only temb (24KB) + `enqueue()` + GPU Euler step kernel
- Download final latents once at the end

### Additional finding: forward() corrupts TRT context
- Fresh context: 167ms/step
- After `forward()` call: 196ms/step (29ms slower)
- `forward()` calls `setTensorAddress` which invalidates TRT's optimized execution path
- Fix: never call `forward()` in the hot loop — use `enqueue()` with pre-bound addresses

### Result
- Before: ~290s for 28 steps (~10s/step via forward())
- After: **3.7s for 28 steps (133ms/step)**
- The 16ms gap vs 117ms raw benchmark is CPU temb computation + cudaMemcpyAsync + stream sync

---

## Lessons Learned

1. **Always rebuild the C++ binary** after code changes — stale binaries cause mysterious gray noise
2. **STRONGLY_TYPED is required for FP8** — without it, TRT ignores FP8 Q/DQ nodes
3. **DQ output type matters**: BF16 prevents activation bloat, FP32 causes 15x memory explosion
4. **add_attention does NOT scale** — must pre-scale Q by 1/sqrt(head_dim) externally
5. **Weight quantization**: must divide by scale before casting to FP8 (`(w/scale).astype(fp8)`)
6. **Scale key naming**: ModelOpt checkpoint uses HF names, not ONNX node names
7. **Never call forward() in hot loops** — use enqueue() with pre-bound GPU buffers
8. **BF16 as base type for FP8 networks** — prevents type boundaries that fragment compilation
9. **Myelin compiles monolithically** when all ops stay in one backend (kMYELIN) — avoid Cast boundaries
10. **ModelOpt FP8_DEFAULT_CFG** quantizes both weights AND activations, but NOT attention BMMs
