# FP8 Quantization via TensorRT API

Complete guide for building FP8 quantized TensorRT engines using the raw
network definition API. Covers the full workflow from calibration to
deployment, with pitfalls discovered during FLUX.2-dev optimization
(1.58x speedup over BF16).

**Reference**: `tensorrt_model_connect/tensorrt_model_connect/flux2_dit_builder.py` (FLUX.2-dev DiT builder).
The quantization framework is in `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` (add_matmul with precision support).

---

## Table of Contents

1. [End-to-End Workflow](#end-to-end-workflow)
2. [Step 1: Calibrate with ModelOpt](#step-1-calibrate-with-modelopt)
3. [Step 2: Extract Scales](#step-2-extract-scales-from-checkpoint)
4. [Step 3: Build TRT Engine](#step-3-build-the-trt-engine)
5. [Step 4: Verify Quality](#step-4-verify-quality)
6. [Step 5: Optimize Runtime](#step-5-optimize-runtime)
7. [Mixed Precision Strategy](#mixed-precision-strategy)
8. [TRT API Patterns](#trt-api-patterns)
9. [Common Mistakes](#common-mistakes)
10. [Debugging Checklist](#debugging-checklist)
11. [Architecture Requirements](#architecture-requirements)

---

## End-to-End Workflow

```
1. Calibrate     ModelOpt FP8_DEFAULT_CFG + calibration data
                 → quantized checkpoint with per-layer amax values
                           │
2. Extract       Parse checkpoint: amax → scale = amax / 448.0
   Scales        Apply exclusion filter (embedders, norms, output heads)
                 → {layer_name: {input_scale, weight_scale}} JSON
                           │
3. Build         STRONGLY_TYPED network, BF16 base type
   Engine        FP8 Q/DQ on calibrated linear layers
                 BF16 for attention (add_attention API), norms, residuals
                 → .engine file (or .trtfb bundle)
                           │
4. Verify        Compare FP8 vs BF16 engine output
                 Check velocity/latent stats, visual quality
                 → confirm cosine > 0.99 for single-step
                           │
5. Deploy        GPU-resident inference loop
                 Pre-allocated buffers, enqueue() not forward()
                 → production-ready pipeline
```

---

## Step 1: Calibrate with ModelOpt

### Install

```bash
pip install nvidia-modelopt
```

### Calibration script template

```python
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

# 1. Load model in BF16
model = load_your_model(torch_dtype=torch.bfloat16, device="cuda")
model.eval()

# 2. Define calibration loop
def calibration_loop(model):
    """Run representative inputs through the model.

    Guidelines:
    - Decoder models: 16-32 diverse prompts, 1 forward pass each
    - Diffusion models: 4-8 random inputs x 4-8 diverse timesteps
    - VL models: mix of text-only and image+text inputs
    - Use real data distribution, not random noise
    """
    for batch in calibration_data:
        with torch.no_grad():
            model(**batch)

# 3. Quantize
model = mtq.quantize(
    model,
    config=mtq.FP8_DEFAULT_CFG,
    forward_loop=calibration_loop,
)

# 4. Disable quantizers on sensitive layers
mtq.disable_quantizer(model, filter_func)

# 5. Save checkpoint
mto.save(model, "quantized.pt")

# 6. Print summary to verify
mtq.print_quant_summary(model)
```

### What FP8_DEFAULT_CFG does

```python
FP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},  # FP8 E4M3, per-tensor
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},   # FP8 E4M3, per-tensor
        # Disabled by default:
        "nn.BatchNorm*": {"*": {"enable": False}},
        "*lm_head*": {"enable": False},
        "*router*": {"enable": False},
        "*mlp.gate.*": {"enable": False},
        # ... other exclusions
    },
    "algorithm": "max",  # tracks max(|x|) across calibration
}
```

- **Quantizes**: Both weights AND activations of `nn.Linear`, `nn.Conv*`
- **Does NOT quantize**: Attention BMMs, BatchNorm, LM heads, MoE routers
- **Per-tensor** (axis=None): one scalar scale per tensor (not per-channel)
- **Max calibration**: scale = max(|x|) / 448.0

### Model-specific exclusion patterns

Choose which layers to exclude based on the model family:

| Model Type | Layers to Exclude | Reason |
|-----------|-------------------|--------|
| **All models** | Output head / lm_head | Final logits need full precision |
| **Diffusion** | Embedders (x_embed, ctx_embed), norm_out, time/guidance MLPs | Small layers, high sensitivity |
| **Decoder** | embed_tokens, lm_head | Vocabulary projections need precision |
| **MoE** | Router/gate projections | Routing decisions need exact values |
| **VL** | Vision patch embedder, cross-attention projections | Boundary layers between modalities |

### Exclusion filter pattern

```python
import re

def make_exclusion_filter(model_type):
    """Return a regex pattern for layers that should NOT be quantized."""
    patterns = {
        "diffusion": (
            r"(proj_out.*|.*(time_text_embed|context_embedder|x_embedder"
            r"|norm_out|time_guidance_embed|stream_modulation).*)"
        ),
        "decoder": r"(.*embed_tokens.*|.*lm_head.*|.*rotary_emb.*)",
        "vl": r"(.*patch_embed.*|.*visual_projection.*|.*lm_head.*)",
    }
    return re.compile(patterns.get(model_type, r"$^"))  # match nothing by default
```

---

## Step 2: Extract Scales from Checkpoint

The ModelOpt checkpoint stores `_amax` values (max absolute activation/weight
seen during calibration). Convert to TRT scales:

```python
import torch, json, re

def extract_fp8_scales(checkpoint_path, exclusion_pattern=None):
    """Extract FP8 scales from ModelOpt checkpoint.

    Returns: {layer_name: {input_scale: float, weight_scale: float}}
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    MAXBOUND = 448.0  # FP8 E4M3 max representable value

    scales = {}
    for key in sd:
        if "_amax" not in key:
            continue

        # Parse: "transformer_blocks.0.attn.to_q.input_quantizer._amax"
        match = re.match(r"(.+)\.(input_quantizer|weight_quantizer)\._amax", key)
        if not match:
            continue

        prefix = match.group(1)   # HF layer name
        qtype = match.group(2)    # input_quantizer or weight_quantizer
        amax = sd[key].item()
        scale = amax / MAXBOUND

        if prefix not in scales:
            scales[prefix] = {}
        if qtype == "input_quantizer":
            scales[prefix]["input_scale"] = scale
        else:
            scales[prefix]["weight_scale"] = scale

    # Keep only layers with BOTH input and weight scales
    complete = {k: v for k, v in scales.items()
                if "input_scale" in v and "weight_scale" in v}

    # Apply exclusion filter
    if exclusion_pattern:
        complete = {k: v for k, v in complete.items()
                    if not exclusion_pattern.match(k)}

    return complete

# Usage:
exclude = re.compile(r"(proj_out.*|.*context_embedder.*|.*x_embedder.*)")
scales = extract_fp8_scales("/tmp/quantized.pt", exclude)
with open("fp8_scales.json", "w") as f:
    json.dump(scales, f, indent=2)
```

### Key naming gotcha

The scale keys MUST match the weight keys used by your TRT engine builder.
ModelOpt uses HuggingFace-style names (e.g., `transformer_blocks.0.attn.to_q`).
If your builder uses different naming, you must create a mapping.

**This was bug #1 in FLUX.2 work**: ONNX node names (`node_linear_10`) were
used instead of HF names — zero scales matched, engine ran as pure BF16.

---

## Step 3: Build the TRT Engine

### Network setup: STRONGLY_TYPED + BF16 base type

```python
import tensorrt as trt

builder = trt.Builder(trt.Logger(trt.Logger.WARNING))

# STRONGLY_TYPED is REQUIRED for FP8
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 30)
# Do NOT set BuilderFlag.FP8 or BuilderFlag.BF16 with STRONGLY_TYPED —
# types come from the graph itself.
```

### FP8 linear layer pattern

```python
import numpy as np
import ml_dtypes

def fp8_linear(network, input_bf16, in_features, out_features,
               weight_fp32, input_scale, weight_scale):
    """FP8 quantized linear layer for STRONGLY_TYPED BF16 network.

    Both input activation and weight are quantized to FP8 E4M3.
    DQ output type is BF16 to match the base network type.
    Weight uses TN layout (required for FP8 fusion on Hopper+/Blackwell).
    """
    # --- Input activation: Q(BF16→FP8) then DQ(FP8→BF16) ---
    s_inp = network.add_constant(
        (), trt.Weights(np.array(input_scale, dtype=np.float32)))
    q = network.add_quantize(
        input_bf16, s_inp.get_output(0), trt.DataType.FP8)
    dq_inp = network.add_dequantize(
        q.get_output(0), s_inp.get_output(0), trt.DataType.BF16)

    # --- Weight: quantize offline, store as FP8 constant + DQ ---
    # CRITICAL: divide by scale BEFORE casting to FP8
    weight_tn = np.ascontiguousarray(weight_fp32.T)  # [out, in] for TN layout
    weight_fp8 = (weight_tn / weight_scale).astype(ml_dtypes.float8_e4m3fn)

    w_const = network.add_constant(
        (out_features, in_features),
        trt.Weights(trt.DataType.FP8, weight_fp8.ctypes.data, weight_fp8.size))
    # IMPORTANT: keep a reference to weight_fp8 to prevent GC during build
    _weight_refs.append(weight_fp8)

    s_wt = network.add_constant(
        (), trt.Weights(np.array(weight_scale, dtype=np.float32)))
    dq_wt = network.add_dequantize(
        w_const.get_output(0), s_wt.get_output(0), trt.DataType.BF16)

    # --- MatMul with TN layout ---
    mm = network.add_matrix_multiply(
        dq_inp.get_output(0), trt.MatrixOperation.NONE,
        dq_wt.get_output(0), trt.MatrixOperation.TRANSPOSE)

    return mm.get_output(0)
```

### Non-quantized layer fallback (BF16)

Layers without calibrated scales must use BF16:

```python
def bf16_linear(network, input_bf16, in_features, out_features, weight_fp32):
    """BF16 linear layer for non-quantized layers."""
    weight_bf16 = weight_fp32.astype(ml_dtypes.bfloat16)
    w_const = network.add_constant(
        (in_features, out_features),
        trt.Weights(trt.DataType.BF16, weight_bf16.ctypes.data, weight_bf16.size))
    _weight_refs.append(weight_bf16)

    mm = network.add_matrix_multiply(
        input_bf16, trt.MatrixOperation.NONE,
        w_const.get_output(0), trt.MatrixOperation.NONE)
    return mm.get_output(0)
```

## TRT API Patterns

### Dispatcher: FP8 if calibrated, BF16 otherwise

```python
def linear_layer(network, input_bf16, in_features, out_features,
                 weight_fp32, layer_name, fp8_scales):
    """Build a linear layer with FP8 if scales exist, BF16 otherwise."""
    scales = fp8_scales.get(layer_name, {})
    inp_scale = scales.get("input_scale")
    wt_scale = scales.get("weight_scale")

    if inp_scale is not None and wt_scale is not None:
        return fp8_linear(network, input_bf16, in_features, out_features,
                          weight_fp32, inp_scale, wt_scale)
    else:
        return bf16_linear(network, input_bf16, in_features, out_features,
                           weight_fp32)
```

### Attention: add_attention API (TRT 10.15+)

```python
def fmha_attention(network, q, k, v, num_heads, head_dim, seq_len):
    """Multi-head attention using TRT's fused FMHA kernel.

    CRITICAL: add_attention does NOT apply 1/sqrt(head_dim) scaling.
    You MUST pre-scale Q (or K) before calling.
    """
    # Pre-scale Q by 1/sqrt(head_dim)
    scale_val = 1.0 / np.sqrt(head_dim)
    scale_bf16 = np.array([scale_val], dtype=ml_dtypes.bfloat16)
    scale_const = network.add_constant(
        (1, 1), trt.Weights(trt.DataType.BF16, scale_bf16.ctypes.data, 1))
    q_scaled = network.add_elementwise(
        q, scale_const.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

    # Reshape [seq_len, dim] → [1, num_heads, seq_len, head_dim]
    def to_4d(tensor):
        s = network.add_shuffle(tensor)
        s.reshape_dims = (1, seq_len, num_heads, head_dim)
        s.second_transpose = (0, 2, 1, 3)
        return s.get_output(0)

    q_4d = to_4d(q_scaled)
    k_4d = to_4d(k)
    v_4d = to_4d(v)

    attn = network.add_attention(
        q_4d, k_4d, v_4d,
        trt.AttentionNormalizationOp.SOFTMAX, False)

    # Reshape back [1, num_heads, seq_len, head_dim] → [seq_len, dim]
    dim = num_heads * head_dim
    flat = network.add_shuffle(attn.get_output(0))
    flat.first_transpose = (0, 2, 1, 3)
    flat.reshape_dims = (seq_len, dim)
    return flat.get_output(0)
```

### I/O boundaries: FP32 ↔ BF16 casts

Cast only at network input/output edges. All internal ops stay BF16.

```python
# Inputs: FP32 from host → cast to BF16 at boundary
x_fp32 = network.add_input("hidden_states", trt.float32, shape)
x_bf16 = network.add_cast(x_fp32, trt.DataType.BF16).get_output(0)

# ... all internal ops in BF16 ...

# Output: BF16 → cast to FP32 for host download
output_fp32 = network.add_cast(output_bf16, trt.DataType.FLOAT).get_output(0)
output_fp32.name = "output"
network.mark_output(output_fp32)
```

---

## Step 4: Verify Quality

### Single-step numerical comparison

```python
# Build both BF16-only and FP8+BF16 engines
# Feed identical random inputs, compare outputs

output_bf16 = run_engine(bf16_engine, random_inputs)
output_fp8 = run_engine(fp8_engine, random_inputs)

diff = np.abs(output_bf16 - output_fp8)
cosine = np.dot(output_bf16.flat, output_fp8.flat) / (
    np.linalg.norm(output_bf16) * np.linalg.norm(output_fp8))
std_ratio = output_fp8.std() / output_bf16.std()

print(f"Cosine similarity: {cosine:.6f}")    # Should be > 0.999
print(f"Std ratio: {std_ratio:.4f}")          # Should be ~1.0 (0.9-1.1)
print(f"Max abs diff: {diff.max():.6f}")
```

**Red flags**:
- Cosine < 0.99 → scale mismatch or weight quantization bug
- Std ratio > 2 or < 0.5 → missing scaling factor (e.g., add_attention bug)
- NaN/Inf → overflow from bad scales

### Multi-step diffusion quality check

For diffusion models, compare denoising trajectory:

```
Step 1 velocity range:  FP8 should be within ~20% of BF16
Step 28 latent range:   Should not explode (stay within BF16's range)
Final image:            Visual comparison (no gray noise, recognizable content)
```

### Checklist for quality validation

- [ ] Single-step cosine > 0.999 vs BF16 baseline
- [ ] Velocity/logit magnitudes match BF16 within 20%
- [ ] No NaN/Inf in any step
- [ ] Final output (image/text) is visually/semantically correct
- [ ] Latent/hidden state ranges don't explode over multiple steps

---

## Step 5: Optimize Runtime

### GPU-resident inference loop (diffusion)

Avoid `TrtModule::forward()` in multi-step loops. Instead:

```cpp
// 1. Get pre-allocated device pointers (set up during engine init)
void* d_hidden = denoiser->device_ptr("hidden_states");
void* d_temb = denoiser->device_ptr("temb");
void* d_output = denoiser->device_ptr("output");

// 2. Upload static inputs ONCE
cudaMemcpyAsync(d_encoder, encoder_data, enc_bytes, H2D, stream);
cudaMemcpyAsync(d_cos, cos_data, cos_bytes, H2D, stream);

// 3. Upload initial latents
cudaMemcpyAsync(d_hidden, initial_latents, hidden_bytes, H2D, stream);

// 4. Hot loop: only upload changing inputs + enqueue + GPU Euler step
for (int step = 0; step < num_steps; ++step) {
    cudaMemcpyAsync(d_temb, temb_data[step], temb_bytes, H2D, stream);
    denoiser->enqueue(stream);  // NOT forward()!
    launch_euler_step(d_hidden, d_output, dt, n_elems, stream);
}
cudaStreamSynchronize(stream);

// 5. Download final result ONCE
cudaMemcpy(result, d_hidden, hidden_bytes, D2H);
```

### Why not forward()

`forward()` calls `setTensorAddress()` per invocation, which invalidates
TRT's internal execution optimization. Measured impact: **+29ms/step** on
FLUX.2-dev (167ms → 196ms). With `enqueue()` on pre-bound addresses, TRT
reuses its optimized kernel launch sequence.

### Pre-compute all timestep embeddings

For diffusion models, temb only depends on the timestep schedule (known
ahead of time). Pre-compute all N tembs on CPU before the loop:

```cpp
std::vector<std::vector<float>> all_tembs(num_steps);
for (int step = 0; step < num_steps; ++step) {
    compute_temb(timesteps[step], all_tembs[step]);
}
// In loop: just upload all_tembs[step] (24KB for dim=6144)
```

---

## Mixed Precision Strategy

### Which layers get FP8

| Layer Type | FP8? | Reason |
|-----------|------|--------|
| **Linear (QKV projections)** | Yes | Dominant compute, tolerates quantization |
| **Linear (FFN up/down)** | Yes | Large matmuls, good speedup |
| **Linear (output projection)** | Yes | Works well with calibrated scales |
| **Attention BMMs (Q@K^T, softmax@V)** | No | ModelOpt doesn't quantize; use FMHA fusion instead |
| **LayerNorm / RMSNorm** | No | Small ops, sensitive to precision |
| **SoftMax** | No | Exponential is precision-sensitive |
| **Residual additions** | No | Accumulation needs BF16 |
| **Embedders (token/patch)** | No | Boundary layers, small, sensitive |
| **Output head (lm_head/proj_out)** | No | Final logits need full precision |
| **Modulation projections** | No | Small, routing-sensitive |
| **SiLU/GELU activations** | No | Nonlinearities stay BF16 |

### Why this split works

- **FP8 on linear layers**: 80%+ of FLOPS in transformers are in linear
  projections. FP8 gives 2x throughput vs BF16 on Hopper+/Blackwell.
- **BF16 on attention**: `add_attention` fuses QKV matmul + softmax + output
  matmul into a single FMHA kernel. This fusion gives more speedup than
  FP8 would on the individual matmuls.
- **BF16 on norms/residuals**: These are memory-bound, not compute-bound.
  FP8 wouldn't speed them up and would hurt precision.

### Base type: always BF16

Set `_ALL_BF16 = True` for the entire STRONGLY_TYPED network. This ensures:
- All non-quantized ops run in BF16 (2 bytes, not 4)
- No FP32 intermediates between FP8 layers
- TensorRT keeps the graph in a compact compiler partition
- Activation memory stays reasonable (~1GB, not ~13GB)

---

## Common Mistakes

### 1. Weight not divided by scale before FP8 cast

```python
# WRONG — raw cast preserves value in FP8 range, DQ then multiplies by scale
weight_fp8 = weight.astype(ml_dtypes.float8_e4m3fn)
# Recovered: float8(0.14) * 0.0003 = 0.000042  (3000x too small!)

# CORRECT — quantize (divide by scale), then cast
weight_fp8 = (weight / scale).astype(ml_dtypes.float8_e4m3fn)
# Recovered: float8(0.14/0.0003) * 0.0003 ≈ 0.14  ✓
```

**Symptom**: Velocity/output values explode (100-1000x expected magnitude).

### 2. DQ output type FP32 (not BF16)

```python
# WRONG — FP32 intermediates bloat activation 15x
dq = network.add_dequantize(fp8_val, scale, trt.float32)

# CORRECT — BF16 matches base network type
dq = network.add_dequantize(fp8_val, scale, trt.DataType.BF16)
```

**Symptom**: 12.9GB activation memory instead of ~1GB. Engine may still run
but much slower due to memory pressure.

### 3. add_attention without pre-scaling Q

```python
# WRONG — add_attention(SOFTMAX) does NOT apply 1/sqrt(head_dim)
attn = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)

# CORRECT — pre-scale Q by 1/sqrt(head_dim)
q_scaled = elementwise_multiply(q, 1.0 / sqrt(head_dim))
attn = network.add_attention(q_scaled, k, v, ...)
```

**Symptom**: Output std is ~4x too large. Cosine similarity ~0.35 vs
explicit SDPA. Image quality is noisy/washed out.

### 4. Scale key mismatch (ONNX names vs HF names)

```python
# WRONG — ONNX node names don't match builder's weight name lookups
scales = {"node_linear_10": {...}, "node_linear_11": {...}}

# CORRECT — use HF layer names that match the builder
scales = {"transformer_blocks.0.attn.to_q": {...}, ...}
```

**Symptom**: Zero scales match → engine runs as pure BF16, no FP8 speedup.

### 5. FP8 on uncalibrated layers

```python
# WRONG — arbitrary defaults produce garbage
if scale is None: scale = 1.0

# CORRECT — fallback to BF16 for uncalibrated layers
if scale is None: return bf16_linear(...)
```

**Symptom**: Output is numerically garbage for uncalibrated layers.

### 6. Explicit Cast nodes between FP8 layers

```python
# WRONG — casts create backend boundaries and fragment compiler partitioning
x = add_cast(x, FP32)  →  FP8 Q/DQ matmul  →  add_cast(x, BF16)

# CORRECT — DQ outputs BF16 directly, no explicit casts needed
x_fp8 = quantize(x_bf16, scale)
x_bf16 = dequantize(x_fp8, scale, BF16)  # DQ already outputs BF16
```

**Symptom**: Engine has hundreds of layers instead of ~1 monolithic kernel.
Performance worse than BF16.

---

## Debugging Checklist

When the FP8 engine produces wrong output, check in this order:

1. **Are scales actually matching?** Print how many layers got FP8 vs BF16
   fallback. If zero got FP8, key naming is wrong.

2. **Is weight divided by scale?** Check `(weight / scale).astype(fp8)` not
   `weight.astype(fp8)`. Look for output values 100-1000x too large or small.

3. **Is DQ output type BF16?** Check activation memory — should be ~1-2GB
   for a 12B model, not 10+GB.

4. **Is add_attention pre-scaled?** Compare std of attention output vs explicit
   SDPA. Ratio should be ~1.0, not ~4.0.

5. **Are excluded layers staying BF16?** Embedders, norms, and output heads
   should not have FP8 Q/DQ.

6. **Is the C++ binary rebuilt?** Stale binary after code changes produces
   gray noise. Always `cmake --build build` before testing.

7. **Single-step comparison**: Feed identical inputs to FP8 and BF16 engines.
   Cosine should be > 0.999. If not, there's a quantization bug.

8. **Multi-step divergence**: Run 5-10 denoising steps. If latent values
   explode, check weight quantization and scale magnitudes.

---

## Architecture Requirements

### GPU architecture

| Feature | Hopper (SM90) | Blackwell (SM100+) |
|---------|--------------|-------------------|
| FP8 E4M3 compute | Yes | Yes |
| FP8 MatMul fusion | opA=NONE, opB=TRANSPOSE (TN layout) | Same |
| FMHA (add_attention) | BF16/FP16 | BF16/FP16 |
| Per-tensor FP8 scales | Yes | Yes |
| Per-channel FP8 scales | Yes (slower — intermediate FP32 mul) | Yes |

### TensorRT version

- **TRT 10.15+**: Required for `add_attention` API
- **TRT 10.x**: FP8 Q/DQ supported with STRONGLY_TYPED
- `ml_dtypes` package: Required for `float8_e4m3fn` numpy dtype

### Key TRT constraints

- `add_quantize` / `add_dequantize`: Scale must be FP32 scalar constant
  (not tensor, not per-channel for optimal fusion)
- `add_attention`: Requires 4D input `[B, H, S, D]`, BF16 or FP16 only
- STRONGLY_TYPED: Cannot use `BuilderFlag.FP8` or `BuilderFlag.BF16` —
  types declared explicitly in the graph
- Weight GC: Keep Python references to numpy arrays used in `trt.Weights()`
  until `build_serialized_network()` returns

### TensorRT compiler partitioning behavior

For maximum performance, the network should stay in a compact compiler
partition. This requires:
- All ops assigned to compatible TensorRT backends
- No type boundaries that force backend switches
- BF16 as uniform base type (no FP32 islands)
- FP8 Q/DQ properly fused into quantized FC ops

Check monolithic compilation via build log:
```
[BlockAssignment] Algorithm ShiftNTopDown took 0.03ms to assign 1 blocks to 1 nodes
```
`1 blocks to 1 nodes` = monolithic. Multiple nodes = fragmented (slower).
