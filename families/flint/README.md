# Flint

This family builds Recursive Flint (`RecursiveSI/flint`, 39.8B parameters, about
2.5B active per token) for `text_generation` in bf16 on one GPU. The bundle has
one TensorRT engine per decoder layer and one head engine; the family runtime
runs them one token at a time and keeps the KV cache. 49 layers: layers 0 and 1
have a dense SwiGLU MLP, layers 2 to 48 have 256 routed experts (top 6, sigmoid
router) plus a shared expert. 14 layers use full attention without RoPE and 35
use sliding-window attention (513 tokens) with RoPE; attention is explicit
(fp32 scores, learned per-head sinks, 32 query and 4 KV heads). The routed
experts of a layer are one stacked constant that the engine gathers by the
router indices. `max_sequence_length` (default 1024) sets the cache profile.

```bash
trtmc build RecursiveSI/flint -o flint.bundle --max-sequence-length 1024
```

## Quantization

FP8 and NVFP4 were measured and are not shipped: the family builds bf16 only.

No ModelOpt FP8 or NVFP4 checkpoint of Flint exists on the Hub (the publisher
has GGUF and INT4 exports; the other FP8 / NVFP4 repositories are community
exports), so the study self-quantized the BF16 checkpoint at build time:

- FP8 W8A8 for dense projections: per-tensor weight scale (amax / 448), static
  per-tensor activation scale (max calibration over 512 texts: Nemotron
  post-training v1 and cnn_dailymail train), bf16 Dequantize. 12 inputs with an
  amax above 20x the median of their kind (shared-expert down input of layers
  42 to 48, routed-expert down input of layers 37 and 45 to 48) stay bf16.
- FP8 routed experts: the stacked expert constant holds e4m3 values and the
  graph runs `Gather -> Quantize(1.0) -> Dequantize`; TensorRT keeps the stack
  in FP8 in the plan. Activations stay bf16 (W8A16) or are quantized (W8A8).
- NVFP4 W4A4 (per-16 e4m3 block scales, `weight_scale_2 = amax / (6 * 448)`,
  activations through `add_dynamic_quantize`) for dense projections only. The
  gathered experts cannot be NVFP4: an FP4 constant can only feed a Dequantize
  and TensorRT has no static FP4 Quantize after a Gather, so NVFP4 builds keep
  the experts FP8.

Setup: TensorRT 11.4, B200 and B300, one job per GPU, all variants built in the
same job, `max_sequence_length` 1024. Speed is the family runtime with the
KV cache and hidden state kept on the GPU (`ITextGeneration::generate`,
greedy) on a 400-token prompt plus 128 new tokens, 3 interleaved rounds x 3
requests, median. Rows marked "old runtime" were timed only with the earlier
runtime that copied the cache between host and device on every layer call
(bf16 about 22 ms/token). Quality is teacher-forced
logits on 5,400 held-out tokens (3 wikitext-2 test and 3 cnn_dailymail test
texts of 900 tokens, so 2,322 tokens are past the sliding window) against a
PyTorch bf16 implementation of the model; KL is KL(PyTorch || engine).

| Variant (what is quantized) | Bundle GiB | Decode ms/token B200 (vs bf16) | Decode ms/token B300 (vs bf16) | KL mean | KL p99 | Top-1 vs PyTorch |
|---|---|---|---|---|---|---|
| bf16 (default) | 74.7 | 7.7 | 7.1 | 0.043 | 0.84 | 94.0% |
| FP8: all linears W8A8, experts W8A16 | 38.3 | 0.93x (old runtime) | 0.92x (old runtime) | 0.119 | 2.08 | 89.4% |
| FP8: all linears W8A8, experts W8A8 | 38.3 | 6.9 (1.12x) | 7.2 (0.97x) | 0.131 | 2.67 | 88.5% |
| FP8: experts only, W8A8 | 39.5 | 7.4 (1.05x) | 7.6 (0.93x) | 0.082 | 1.33 | 90.8% |
| FP8: experts only, W8A16 | 39.5 | not timed | not timed | 0.067 | 1.27 | 92.1% |
| FP8: dense linears only (experts bf16) | 73.5 | not timed | not timed | 0.111 | 2.54 | 89.9% |
| NVFP4: dense MLP and shared expert, attention FP8, experts FP8 W8A16 | 38.1 | 0.94x (old runtime) | 0.92x (old runtime) | 0.247 | 4.90 | 84.6% |
| NVFP4: all dense linears, experts FP8 W8A16 | 37.8 | 0.92x (old runtime) | 0.88x (old runtime) | 0.392 | 5.81 | 78.7% |

The bf16 engine is
already 0.043 KL away from PyTorch bf16; PyTorch bf16 against PyTorch fp32 is
0.037, so this floor is bf16 rounding that changes expert choices, not an
engine defect. No variant gave a non-finite logit on the held-out texts or on
stress inputs (600 repeated tokens, 600 random tokens, the 144 highest token ids three times,
a full 1024-token context). TensorRT 11.1 (B200) builds the bf16, FP8 and NVFP4
bundles with the same quality (KL 0.035 / 0.118 / 0.253).

Reasons:

- Speed: with the cache on the GPU, decode (about 7.4 ms/token) is the summed
  engine time. FP8 all linears W8A8 is 1.12x on B200 but 0.97x on B300, and
  FP8 experts-only W8A8 is 1.05x / 0.93x, so FP8 is not faster on both GPUs.
  The W8A16 expert gather is slower than the bf16 gather (0.92x to 0.93x with
  the old runtime).
- Quality: the dense projections are sensitive (FP8 on them alone adds 0.068
  KL over the bf16 floor), so every FP8 layout that is not slower adds 0.04 to
  0.09 KL and loses 3 to 6 points of top-1. FP8 halves the bundle (74.7 to
  38.3 to 39.5 GiB) but with that loss and no speed gain on both GPUs it is
  not shipped.
- NVFP4 is slower than bf16 (0.88x to 0.94x, old runtime), has 1.9x to 3.3x the KL of the
  FP8 layouts and no size gain over FP8, because the experts that hold 95% of
  the weights cannot be NVFP4.
