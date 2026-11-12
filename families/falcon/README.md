# Falcon

This family builds Falcon decoders (`tiiuae/falcon-rw-1b` in the manifests) for
`text_generation`: LayerNorm, a fused query/key/value projection, an attention
output projection, a two-projection GELU MLP and ALiBi (Falcon-RW) or RoPE
positions. It builds a prefill engine (`prefill.plan`, dynamic prompt length)
and a single-token decode engine (`engine.plan`) with a static KV cache of
`max_sequence_length` tokens (default 256). Precision is fp32, fp16 or bf16,
batch 1, with optional tensor parallelism for unquantized builds.

## Quantization

`quantization="fp8"` self-quantizes the original BF16 checkpoint at build time
(`quantization.py`): q, k, v (one shared input scale), the attention output
projection and both MLP projections of every layer run as FP8 W8A8 GEMMs
(per-tensor weight scale `amax / 448` from the loaded weights, static
per-tensor activation scale from `fp8_activation_scales.json`, ModelOpt
`FP8_DEFAULT_CFG` calibration, fp16 Dequantize). `lm_head`, the embedding and
the norms stay fp16. Only the model config `falcon-h2048-l24-i8192`
(Falcon-RW-1B) has calibrated scales. FP8 requires fp16 and no tensor
parallelism. No other quantization is accepted.

### NVFP4

NVFP4 was built and measured for Falcon-RW-1B with a quick screen (TensorRT
11.4, one job on B200 and one on B300, run twice). It is not shipped: with the
same projections as FP8, decode is slower than the FP8 build on both GPUs
(0.94x to 0.99x), and the quality loss is 5x the FP8 loss (KL against the
PyTorch model), above the 2x limit. The MLP-only variant (only the MLP up
projection in NVFP4) is 1.02x to 1.09x faster than FP8 in decode, but its KL is
3.1x the FP8 KL, also above the limit. The full study was not run. No ModelOpt
NVFP4 checkpoint of Falcon-RW exists, so the weights were self-quantized from
the BF16 checkpoint. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the committed FP8 activation scale divided by 6
(same calibration data and statistic). Projections that are not NVFP4 keep the
FP8 layout of `quantization.py`. The input of the MLP down projection
(`dense_4h_to_h`) stays FP8 in every variant: an NVFP4 input right after the
fused bias and GELU gives NaN in the last partial 128-row tile of the prefill
GEMM on TensorRT 11.4 (a known TensorRT problem seen in other GELU decoders).
The plan inspector shows FP4 GEMMs for every NVFP4 projection.

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| NVFP4 (FP8 projections, MLP down kept FP8) | q, k, v, attention output, MLP up | MLP down |
| MLP only | MLP up | q, k, v, attention output, MLP down |

Results: fp16 (manifest precision), engine only, batch 1. Decode is one token
at position 100 with CUDA graph replay (eager launch in brackets), median of 7
interleaved rounds, all variants built and timed in the same job; decode at
`max_sequence_length` 256 (manifest config) and 2048. KL is KL(PyTorch bf16 ||
engine) per token, teacher-forced over 514 wikitext-2 test tokens (2 windows of
320 tokens, 64-token prefill then one-token decode, 2048 build). NaN counts
non-finite logits and K/V values.

| GPU | TensorRT | `max_sequence_length` | Decode fp16 | Decode FP8 | Decode NVFP4 (x FP8) | Decode MLP only (x FP8) |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 256 | 1.346 ms | 1.057 ms | 1.120 ms, 0.94x (0.95x) | 1.003 ms, 1.05x (1.04x) |
| B200 | 11.4 | 2048 | 1.723 ms | 1.431 ms | 1.505 ms, 0.95x (0.95x) | 1.395 ms, 1.03x (1.02x) |
| B300 | 11.4 | 256 | 1.163 ms | 1.046 ms | 1.079 ms, 0.97x (0.95x) | 0.963 ms, 1.09x (1.08x) |
| B300 | 11.4 | 2048 | 1.550 ms | 1.426 ms | 1.449 ms, 0.98x (0.98x) | 1.340 ms, 1.06x (1.05x) |

| GPU | TensorRT | KL fp16 / FP8 / NVFP4 / MLP only | Top-1 agreement | NaN |
|---|---|---|---|---|
| B200 | 11.4 | 0.0019 / 0.0096 / 0.0474 (4.9x) / 0.0299 (3.1x) | 96.3% / 92.6% / 86.6% / 92.0% | 0 |
| B300 | 11.4 | 0.0018 / 0.0095 / 0.0476 (5.0x) / 0.0294 (3.1x) | 96.1% / 93.0% / 86.4% / 91.8% | 0 |

- Perplexity (PyTorch 9.72): fp16 9.72, FP8 9.81, NVFP4 10.17, MLP only 9.94.
- Prefill of 128 tokens (256 build): NVFP4 0.92x to 0.93x FP8, MLP only 0.95x
  to 0.97x.
- Plan size, prefill plus decode (256 build, B200): fp16 4.89 GiB, FP8 2.64
  GiB, NVFP4 1.99 GiB, MLP only 2.32 GiB. The size gain alone is not a reason
  to ship.
- NaN scan: 0 non-finite logits or K/V for every variant on both GPUs at 256
  and 2048 (random tokens, one repeated token and wikitext text through the
  prefill and decode plans, mixed chunk lengths, up to 1024 prompt tokens, 32
  greedy steps after each).
- The first screen job on each GPU gave the same picture (eager decode at 256:
  NVFP4 0.97x to 0.99x FP8, MLP only 1.07x to 1.09x).

Why it does not help: Falcon-RW-1B at batch 1 is only partly weight-bandwidth
bound. FP8 already removes most of the weight traffic (1.09x to 1.27x faster
decode than fp16); NVFP4 halves the remaining projection weights again but adds
a dynamic activation quantization per projection, which costs about as much as
it saves. The MLP-only gain is small and its quality cost is 3x the FP8 cost.
Earlier small-decoder studies (gpt2, opt, bloom, xglm, gpt_neox, codegen) found
the same. Revisit for `falcon-rw-7b` (FP8 gave 1.42x there), batched decode, or
a recipe that keeps the KL within 2x of FP8.
