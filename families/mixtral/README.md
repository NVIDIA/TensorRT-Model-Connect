# Mixtral

This family builds Mixtral sparse mixture-of-experts decoders
(`ggml-org/stories15M_MOE` in the manifests, `Mixtral-8x7B-Instruct-v0.1` for
the quantized builds) for `text_generation`: grouped-query attention with RoPE,
RMSNorm, and a routed SwiGLU MoE block (8 experts, top-2 routing with
renormalized softmax weights). Every expert runs for every token and the
experts the router did not select get a routing weight of 0. Unquantized builds
use one engine with a prefill profile and a one-token decode profile; quantized
builds feed one token per step. Precision is fp32, fp16 or bf16, batch 1, with
optional tensor parallelism for unquantized builds.

## Quantization

`quantization="fp8"` reuses a ModelOpt FP8 checkpoint as is (#1916), for
example `nvidia/Mixtral-8x7B-Instruct-v0.1-FP8`: the attention q, k, v
projections and all expert projections (`w1`, `w3`, `w2` of every expert) run
as FP8 W8A8 GEMMs with the checkpoint weights and per-tensor scales. The
attention output projection runs fp16 with the dequantized checkpoint weight.
The router, embedding, `lm_head` and norms stay fp16. FP8 requires fp16 and no
tensor parallelism. No other quantization is accepted.

### NVFP4

NVFP4 was built and measured for Mixtral-8x7B-Instruct with the published
ModelOpt checkpoint `nvidia/Mixtral-8x7B-Instruct-v0.1-NVFP4`. It is faster
than FP8 in decode on both B200 and B300, but it is not shipped: its quality
loss is 4.6x to 4.9x the FP8 loss (KL against the PyTorch BF16 model), above
the 2x limit. The NVFP4 code was not merged.

Recipe: the checkpoint's packed E2M1 weights, per-16 e4m3 block scales and
per-tensor `weight_scale_2` are used bit-exact. Every expert projection runs
as an NVFP4 W4A4 GEMM; the activation goes through `add_dynamic_quantize`
(block 16) with the checkpoint `input_scale` as the global scale. The experts
stay separate GEMMs, as in the FP8 build. The attention q, k, v and o
projections run fp16 with the dequantized NVFP4 weights (see the TensorRT
problem below). The router, embedding, `lm_head` and norms stay fp16.

Results: fp16 engines, `max_sequence_length` 256, all three variants built and
timed in the same job (one job per GPU and TensorRT version). Decode is one
token at position 100, engine only, no CUDA graph, median of 9 interleaved
rounds. Quality is teacher-forced over 4096 wikitext-2 test tokens (32 chunks
of BOS + 128 tokens) against `mistralai/Mixtral-8x7B-Instruct-v0.1` in PyTorch
bf16: KL(bf16 || engine) per token, next-token top-1 agreement, perplexity.

| GPU | TensorRT | Decode fp16 | Decode FP8 | Decode NVFP4 (x FP8) | KL fp16 / FP8 / NVFP4 (x FP8) | Top-1 fp16 / FP8 / NVFP4 | Perplexity fp16 / FP8 / NVFP4 (bf16) |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 25.90 ms | 12.03 ms | 7.43 ms (1.62x) | 0.0026 / 0.0146 / 0.0720 (4.9x) | 97.6% / 93.8% / 87.7% | 10.98 / 10.95 / 11.39 (11.04) |
| B300 | 11.4 | 23.14 ms | 11.75 ms | 8.86 ms (1.33x) | 0.0030 / 0.0150 / 0.0684 (4.6x) | 96.9% / 94.1% / 87.3% | 10.98 / 10.94 / 11.38 (11.02) |
| B200 | 11.1 | 27.26 ms | 12.36 ms | 7.76 ms (1.59x) | 0.0026 / 0.0148 / 0.0708 (4.8x) | 97.4% / 93.9% / 87.3% | 10.98 / 10.93 / 11.31 (11.04) |

- Size: engine plan 87.0 GiB (fp16), 44.3 GiB (FP8), 26.6 GiB (NVFP4).
- NaN scan: 0 non-finite logits for every variant on both GPUs (the quality
  runs, 6 prompts with 24 greedy tokens each, and 255-token runs of random
  token ids, one repeated token and the largest token ids).
- TensorRT problem: with q, k and v also NVFP4 W4A4 (the full checkpoint
  recipe), the engine output is wrong (top-1 agreement 0%, KL 41.7) on
  TensorRT 11.4, with no NaN. Single NVFP4 GEMMs with the real weights are
  correct. A 2-layer engine of the real checkpoint reproduces it: the q/k/v
  activation quantization of layer 1 is wrong when the experts of layer 0 are
  NVFP4; any other pairing is correct, and q/k/v with NVFP4 weights but fp16
  activations is correct. A cast barrier or turning off the dual-GEMM fusion
  does not help. Part of the remaining NVFP4 quality gap may come from the same
  problem: engines with only the experts in NVFP4 also lose accuracy as the
  layer count grows, which was not separated from the quantization error.
- Revisit when TensorRT fixes the q/k/v problem above, or with an NVFP4
  checkpoint or recipe that keeps the KL within 2x of FP8.
