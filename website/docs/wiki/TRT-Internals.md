# TRT Internals

This page explains how the TensorRT engine is built (Python) and how it runs (C++). The build and runtime phases are split across languages.

## Overview

The TRT pipeline has two phases:

1. **Build phase (Python)**: `python/tensorrt_model_connect/` reads HF model weights, constructs a TRT `INetworkDefinition` via the TensorRT Python API, compiles it to an `ICudaEngine`, and serializes the engine plan into a `.trtfb` bundle.
2. **Run phase (C++)**: The C++ runtime deserializes the engine plan from the bundle, creates an execution context, and runs the autoregressive generation loop on GPU with KV-cache management.

---

## Decoder Layer Anatomy

The standard decoder graph builder (in Python) implements the dominant modern LLM decoder pattern. Each layer performs:

### 1. Pre-Attention RMSNorm
```
norm1 = RMSNorm(hidden, input_norm_weights, eps)
```
RMS normalization: `x * gamma / sqrt(mean(x^2) + eps)`.

### 2. QKV Projections
```
Q = norm1 * W_q    [1, hidden] * [hidden, attention_size] -> [1, attention_size]
K = norm1 * W_k    [1, hidden] * [hidden, attention_size] -> [1, attention_size]
V = norm1 * W_v    [1, hidden] * [hidden, attention_size] -> [1, attention_size]
```

### 3. Optional Per-Head QK Norm (Qwen3)
```
Q = PerHeadRMSNorm(Q, q_norm_weights)
K = PerHeadRMSNorm(K, k_norm_weights)
```
LLaMA and Mistral skip this step. Qwen3 applies per-head RMS normalization before RoPE.

### 4. Rotary Position Embeddings (RoPE)
```
Q = ApplyRoPE(Q, position_id, cos_table, sin_table)
K = ApplyRoPE(K, position_id, cos_table, sin_table)
```
The cos/sin tables are precomputed for all positions up to `max_cache_length + 1` and stored as constant tensors in the TRT network.

### 5. KV-Cache Concatenation
```
all_K = Concat(cache_K, current_K)
all_V = Concat(cache_V, current_V)
```

### 6. Grouped Query Attention (GQA)
```
scores = Q_heads @ K_heads^T / sqrt(head_dim)
scores = scores + attention_mask
weights = softmax(scores)
context = weights @ V_heads
```
GQA is handled transparently: the checkpoint mapper expands K/V projections to match the number of query heads during loading.

### 7. Output Projection + Residual
```
attn_output = attn_output * W_o
hidden = hidden + attn_output
```

### 8. Post-Attention RMSNorm + SwiGLU MLP
```
norm2 = RMSNorm(hidden, post_attn_norm_weights, eps)
gate = norm2 * W_gate
up   = norm2 * W_up
swish = gate * sigmoid(gate)
gated = swish * up
down  = gated * W_down
hidden = hidden + down
```

### 9. Final Layer: Norm + LM Head
```
hidden = RMSNorm(hidden, final_norm_weights, eps)
logits = hidden * W_lm_head
```

---

## Graph Building (Python)

The Python `python/tensorrt_model_connect/` package builds the TRT network graph using the TensorRT Python API. Shared ops in `python/tensorrt_model_connect/graph_ops.py` provide reusable building blocks:

| Function | Description |
|----------|-------------|
| `add_constant_tensor()` | Creates a constant weights tensor in the TRT network |
| `add_matmul()` | Matrix multiply with constant weights |
| `add_bias_sum()` | Adds a bias vector to a tensor |
| `add_rms_norm()` | Full-hidden RMS normalization with gamma weights |
| `add_rms_norm_per_head()` | Per-head RMS normalization (for Qwen3 QK norms) |
| `add_rope()` | Apply RoPE to Q or K tensor using precomputed tables |
| `add_attention()` | Scaled dot-product attention with masking |
| `add_swiglu()` | SwiGLU MLP block (gate + up + SiLU + down) |

The standard decoder graph builder composes these ops into a full N-layer network:

```
create_decoder_step_network():
  1. Create IBuilder, INetworkDefinition, IBuilderConfig
  2. Add inputs: token_id[1], position_id[1], attention_mask[1, window],
     per-layer cache_k/cache_v [max_cache, attn_size]
  3. Add embedding gather: token_id -> hidden[1, H]
  4. Precompute RoPE tables as constants
  5. For each decoder layer:
     - add_rms_norm -> QKV proj -> optional QK norm -> add_rope
     - KV cache concat -> add_attention -> output proj -> residual
     - add_rms_norm -> add_swiglu -> residual
  6. Final RMSNorm + LM head matmul -> logits[1, vocab]
  7. Mark outputs: logits + per-layer present_k/present_v
  8. buildEngineWithConfig(network, config) -> ICudaEngine
  9. Serialize engine plan to bytes
```

Custom family graph builders can reuse these shared ops and compose them differently (e.g., MoE routing, parallel attention, vision encoder).

### VL Image Preprocessing (Non-TRT)

Vision-language models require image preprocessing before the vision TRT engine. This is handled by `image_preprocessor.cpp` (C++) and `debug_runner.py` (Python), both implementing the same 4 strategies:

| Strategy | Pipeline | Use Case |
|----------|----------|----------|
| `qwen_merge_group` | Load -> resize -> normalize -> merge-group patch permutation -> temporal duplication | Qwen2.5-VL |
| `simple_chw` | Load -> resize -> normalize | Standard ViT (LLaVA, InternVL, Phi-3-Vision) |
| `center_crop_chw` | Load -> center-crop to square -> resize -> normalize | CLIP, DINOv2-based models |
| `aspect_preserve_chw` | Load -> aspect-preserving resize -> zero-pad to square -> normalize | InternVL v2 |

Interpolation is configurable: `"bicubic"` (default, Catmull-Rom), `"bilinear"` (triangle), `"nearest"` (point sample). The mode is read from `config.json` (set by the engine builder), with fallback to the HF `preprocessor_config.json` `resample` integer.

---

## TRT Engine Lifecycle

### Engine Compilation (Python)

Engine compilation happens during `./build/trtmc build`:
- TensorRT compiles the network graph into optimized CUDA kernels
- Compilation takes 30-300 seconds depending on model size
- The serialized plan is written into the `.trtfb` bundle

### Engine Deserialization (C++)

Engine deserialization happens during `trtmc_create_pipeline()`:
- `ReadBundleFile()` extracts the engine plan bytes from the bundle
- `createInferRuntime(logger)` creates a TRT runtime
- `deserializeCudaEngine(plan_bytes)` recreates the `ICudaEngine` (~5s)
- `createExecutionContext()` creates the execution context

### TrtModule

The deserialized engine is wrapped in `TrtModule` (C++, `include/trtmc/runtime/trt_module.h`):

`TrtModule` wraps a TRT `ICudaEngine` + `IExecutionContext`. It pre-allocates
all device buffers at construction and provides forward pass modes:
- `forward(inputs) -> TensorMap` (CPU-to-CPU, synchronous)
- `forward_device(inputs) -> DeviceTensorMap` (GPU-to-GPU, no copies)
- `forward_async(inputs)` + `sync()` (split async path)
- `bind_external(name, ptr)` for injecting KV cache / recurrent state buffers

---

## Autoregressive Generation Loop (C++)

`TextGenerationPipeline::generate()` in `src/runtime/models/<family>/pipeline.cpp`:

```
generate(prompt, cfg):
  1. Tokenize: tokenizer->encode(prompt) -> input_ids
  2. Reset KvCache: zero all buffers, position = 0
  3. Bind KvCache to TrtModule: bind_external() for cache_k/v and present_k/v per layer

  4. Prefill phase:
     For each token in input_ids[0..N-2]:
       - build_attention_mask(mask)
       - module.forward({token_id, position_id, attention_mask})
       - cache.advance() -- D2D copy: present_k/v -> cache_k/v[position], position++

  5. Decode phase:
     For step = 0 to max_new_tokens:
       - Run one decode step with the previously generated token
       - Greedy sampling: argmax over logits -> next_token_id
       - If next_token == eos_token: break
       - Append to output, advance cache

  6. Decode output token IDs to text: tokenizer->decode(new_tokens)
  7. Return: TextResult{text, token_ids}
```

### KV-Cache Management

The cache uses a fixed-size buffer per layer, held in device memory (`KvCache` in `include/trtmc/runtime/kv_cache.h`):
- Size: `[max_cache_length, kv_dim]` per layer, per K and V, resident on GPU
- `bind_to(module)` injects cache pointers directly into the TrtModule's execution context
- `advance()` does D2D async copy of present K/V into cache slots, then increments position
- `build_attention_mask()` produces a causal mask: visible positions = 0.0, future = -1e4
- When cache is full, position clamps to `max_length - 1` (sliding window)

### CUDA Resource Management

All GPU resources use RAII wrappers from `trt_common.h`:
- `CudaStream` -- RAII `cudaStream_t`
- `CudaBuffer` -- RAII `cudaMalloc`/`cudaFree`
- `TrtUniquePtr<T>` -- Smart pointer for TRT objects
- `TrtLogger` -- Custom `ILogger` implementation with error tracking
