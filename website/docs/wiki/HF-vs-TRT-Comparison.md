# HuggingFace Transformers vs TensorRT-Model-Connect

This page provides a detailed comparison between HuggingFace's Python `transformers` library and this library. Understanding these similarities and differences helps when porting models or debugging parity issues.

## API Comparison

| Concept | HuggingFace Transformers | TensorRT-Model-Connect |
|---------|-------------------------|---------------------|
| Build engine | N/A (eager execution) | `trtmc-build build <model-dir> -o model.trtfb` (Python) |
| Run inference | `pipeline("Hello")` | `trtmc run model.trtfb --prompt "Hello"` (C++) |
| Programmatic API | `pipeline("text-generation", model=...)` | `trtmc_create_pipeline("model.trtfb", flags)` (C ABI) |
| Model loading | `AutoModelForCausalLM.from_pretrained()` | Python checkpoint mapper in `tensorrt_model_connect/` |
| Tokenizer | `AutoTokenizer.from_pretrained()` | `HfPythonTokenizer` (subprocess) or `VocabTokenizer` |
| Config | `AutoConfig.from_pretrained()` | Python `config.json` parsing in `tensorrt_model_connect/tensorrt_model_connect/config.py` |

## Architectural Parallels

### Model Discovery

**HuggingFace**: Uses `AutoModelForCausalLM` which reads `config.json`'s `architectures` field to dispatch to a specific Python model class (e.g., `Qwen2ForCausalLM`).

**TensorRT-Model-Connect**: The Python `tensorrt_model_connect/` package reads `config.json`'s `model_type` to dispatch to a registered family plugin (e.g., Qwen plugin). Same metadata, different dispatch mechanism.

```python
# HF: Dynamic class dispatch
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
# Internally: architectures=["Qwen2ForCausalLM"] -> Qwen2ForCausalLM class
```

```bash
# trtmc: Family plugin dispatch (Python build)
trtmc-build build path/to/Qwen3-0.6B -o qwen3.trtfb
# Internally: model_type="qwen3" -> QwenPlugin -> checkpoint mapper -> TRT graph
```

### Weight Loading

**HuggingFace**: `from_pretrained()` downloads safetensors, uses `safetensors` library to load tensors directly into PyTorch `nn.Parameter` objects using the exact HF tensor key names.

**TensorRT-Model-Connect**: The Python `tensorrt_model_connect/` package uses the same `safetensors` Python library, then the **checkpoint mapper** translates HF key names to canonical format with explicit transposition.

Key difference: **HF stores weights as `[out_features, in_features]`** (PyTorch convention). The checkpoint mapper transposes them to `[in_features, out_features]` during mapping for efficient right-side matmul in TRT.

### The Transformer Block

Both implement the same mathematical operations. The differences are in execution strategy:

| Operation | HuggingFace | TensorRT-Model-Connect |
|-----------|-------------|---------------------|
| RMSNorm | `Qwen2RMSNorm(nn.Module)` -- Python class, eager PyTorch op | TRT graph op, fused kernel |
| QKV Projection | `nn.Linear` (separate Q, K, V modules) | TRT `addMatrixMultiply` -- constant folded |
| RoPE | `apply_rotary_pos_emb()` -- dynamic Python function | Precomputed cos/sin tables as TRT constants |
| Attention | `torch.matmul` + masking + softmax | TRT `addMatrixMultiply` + `addSoftMax` -- fused |
| SwiGLU | `self.act_fn(gate) * up` in Python | TRT `addActivation(SIGMOID)` + `addElementWise(PROD)` -- fused |
| Residual | Python `+` operator | TRT `addElementWise(kSUM)` |

### GQA (Grouped Query Attention)

**HuggingFace**: Keeps K/V at their natural smaller size (`num_key_value_heads`), then `repeat_kv()` expands them at attention time.

**TensorRT-Model-Connect**: Expands K/V projections at **checkpoint loading time** (in the Python checkpoint mapper). The TRT graph builder always sees matching head counts.

### KV Cache

**HuggingFace**: `DynamicCache` class. K/V tensors grow dynamically as the sequence extends.

**TensorRT-Model-Connect**: Fixed-size `CudaBuffer` per layer (C++ runtime). Circular buffer with `max_cache_length` slots. The cache length is set at engine build time.

## Tokenization

**HuggingFace**: `tokenizers` library (Rust-backed, called from Python). BPE, WordPiece, SentencePiece, etc.

**TensorRT-Model-Connect**: Two paths:
1. **HfPythonTokenizer**: Spawns a Python subprocess that imports `transformers` and calls the same HF tokenizer. Exact parity guaranteed.
2. **VocabTokenizer**: Simple word-to-id lookup from vocabulary list.

## Numerical Parity

For the same model weights and input, TRT and HF should produce nearly identical logits. Sources of small differences:

| Source | Magnitude | Reason |
|--------|-----------|--------|
| FP32 accumulation order | ~1e-6 | TRT may reorder operations for GPU efficiency |
| TF32 (disabled by default) | ~1e-3 | We explicitly `clearFlag(BuilderFlag::kTF32)` |
| RoPE table precision | ~1e-7 | Precomputed vs on-the-fly computation |
| Softmax implementation | ~1e-6 | Different reduction algorithms |

Validation tools:
```bash
# E2E logit comparison
python3 tools/diff_logits.py --model <hf-model> --atol 1e-3 --battery

# Per-layer hidden state comparison
python3 tools/diff_layers.py --model <hf-model> --atol 0.05

# Vision-language feature comparison (TRT vision encoder vs HF)
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --model Qwen/Qwen2.5-VL-3B-Instruct --atol 0.1

# Debug preprocessor with override
python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
  --vision-only --preprocessor-type simple_chw

# Performance comparison (serial GPU execution — supports large models on 24GB)
python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3.trtfb \
  --prompt "Hello" --max-new-tokens 20
```

## What HF Has That We Don't (Yet)

| Feature | HF | TensorRT-Model-Connect |
|---------|-----|-----------------|
| Auto model download | Hub integration, `from_pretrained("Qwen/Qwen3-0.6B")` downloads automatically | Must pre-download weights to local directory |
| Sampling strategies | top-k, top-p, beam search, temperature, repetition penalty | Greedy argmax only (currently) |
| Dynamic batch size | Arbitrary batch sizes | Batch size = 1 |
| Quantization | GPTQ, AWQ, bitsandbytes | FP16, FP8, INT8, INT4, NVFP4, W4A8 via extensible quantization framework |
| Attention variants | Flash Attention 2, SDPA, PagedAttention | Standard scaled dot-product in TRT |
| Model formats | safetensors, PyTorch bin, GGUF | safetensors only |
| Architecture breadth | 200+ model architectures | 63 family plugins: decoders, MoE, SSM, VL, diffusion, audio, encoder-only, seq2seq, segmentation |

## What We Have That HF Doesn't

| Feature | Description |
|---------|-------------|
| TensorRT kernel fusion | Operations are fused into optimized CUDA kernels at compile time |
| Bundle distribution | `.trtfb` files: single-file deployment, no Python needed at runtime |
| Zero Python runtime | Core inference path is pure C++ (tokenizer bridge is optional) |
| Embeddable | Statically linked library, no interpreter needed |
| Instant startup | Bundle loading (~5s) vs model loading + compilation (~60-300s) |

## Workflow Comparison

```
HuggingFace:
  pip install transformers
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
  output = model.generate(input_ids)
  # Everything happens in Python, single process

TensorRT-Model-Connect:
  # Step 1: Build (Python, once per model per GPU)
  trtmc-build build path/to/Qwen3-0.6B -o qwen3.trtfb

  # Step 2: Run (C++, no Python needed except for tokenizer)
  trtmc run qwen3.trtfb --prompt "Hello" --max-new-tokens 30
```
