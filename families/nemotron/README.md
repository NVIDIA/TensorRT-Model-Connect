# Nemotron

This family builds dense Nemotron-4 decoders (`nvidia/Nemotron-Mini-4B-Instruct`
and `nvidia/Nemotron-4-Mini-Hindi-4B-Base` in the manifests) for
`text_generation`: LayerNorm1P, separate q, k, v and o projections, a
squared-ReLU MLP with no gate projection (`up_proj`, `down_proj`), GQA, partial
RoPE and untied embedding and `lm_head`. It builds a prefill engine
(`prefill.plan`, dynamic prompt length) and a single-token decode engine
(`engine.plan`) with a static KV cache of `max_sequence_length` tokens
(default 256). Precision is fp32, fp16 or bf16, batch 1, no tensor
parallelism.

## Quantization

`quantization="fp8"` self-quantizes the original checkpoint at build time
(`quantization.py`): the q, k, v, up and down projections of every layer run
as FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` from the loaded
weights, static per-tensor activation scale from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG` calibration). The attention output projection,
`lm_head`, the embedding and the norms stay 16 bit. Only
Nemotron-Mini-4B-Instruct has calibrated scales (the key is the model config
plus the final norm sum); Nemotron-4-Mini-Hindi-4B-Base raises until it is
calibrated. FP8 requires fp16 or bf16. No other quantization is accepted.

### NVFP4

NVFP4 was built and measured for Nemotron-Mini-4B-Instruct with a quick screen
(TensorRT 11.4, B200 and B300). It is not shipped: decode is not faster than
the FP8 build in a way that counts (best variant 1.00x to 1.04x, all
projections 0.92x to 0.95x), and the quality loss is 6x to 9x the FP8 loss (KL
against the PyTorch model), far above the 2x limit. The full study was not
run. No ModelOpt NVFP4 checkpoint of Nemotron-Mini exists, so the weights were
self-quantized from the original checkpoint. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the committed FP8 activation scale divided by 6
(same calibration data and statistic). Projections that are not NVFP4 keep the
FP8 layout of `quantization.py`. The attention output projection, `lm_head`,
the embedding and the norms stay 16 bit in every variant. The plan inspector
shows FP4 GEMMs for every NVFP4 projection.

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| NVFP4 (same projections as FP8) | q, k, v, up, down | none |
| MLP only | up, down | q, k, v |

Results: fp16, engine only, batch 1. Decode is one token at position 100 with
CUDA graph replay, median of 7 interleaved rounds, all variants built and timed
in the same job. The 256-token cache (manifest config) was measured in two
jobs per GPU (both values shown). KL is KL(PyTorch bf16 || engine) per token,
teacher-forced over 1416 wikitext-2 test tokens (8 windows of 240 tokens,
64-token prefill then one-token decode), 256-token cache. NaN counts
non-finite logits and K/V values.

| GPU | TensorRT | Cache | Decode fp16 | Decode FP8 | Decode NVFP4 (x FP8) | Decode MLP only (x FP8) |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 256 | 2.382 / 2.485 ms | 2.164 / 2.261 ms | 2.278 / 2.431 ms, 0.95x / 0.93x | 2.097 / 2.209 ms, 1.03x / 1.02x |
| B200 | 11.4 | 4096 | 3.210 ms | 2.870 ms | 3.058 ms, 0.94x | 2.850 ms, 1.01x |
| B300 | 11.4 | 256 | 2.454 / 2.425 ms | 2.034 / 2.038 ms | 2.141 / 2.152 ms, 0.95x / 0.95x | 1.987 / 1.968 ms, 1.02x / 1.04x |
| B300 | 11.4 | 4096 | 3.190 ms | 2.689 ms | 2.925 ms, 0.92x | 2.693 ms, 1.00x |

| GPU | KL fp16 / FP8 / NVFP4 / MLP only | Top-1 agreement fp16 / FP8 / NVFP4 / MLP only | Perplexity fp16 / FP8 / NVFP4 / MLP only | NaN |
|---|---|---|---|---|
| B200 | 0.0008 / 0.0137 / 0.1192 (8.7x) / 0.0804 (5.9x) | 98.4% / 93.7% / 83.2% / 85.7% | 15.39 / 15.57 / 16.51 / 16.00 (PyTorch 15.46) | 0 |
| B300 | 0.0010 / 0.0136 / 0.1256 (9.2x) / 0.0838 (6.2x) | 97.7% / 93.1% / 82.8% / 85.6% | 15.41 / 15.55 / 16.73 / 16.05 (PyTorch 15.48) | 0 |

- Prefill of 128 tokens (256-token cache): NVFP4 0.96x to 0.99x FP8, MLP only
  0.98x to 1.01x.
- Decode plan size: fp16 7.81 GiB, FP8 5.65 GiB, NVFP4 4.71 GiB, MLP only
  4.92 GiB. The untied embedding and `lm_head` (256000 x 3072 each) stay 16
  bit. The size gain alone is not a reason to ship.
- NaN scan: 0 non-finite logits or K/V for every variant on both GPUs, with the
  256- and 4096-token cache (random tokens, one repeated token and wikitext text
  through the prefill and decode plans, mixed chunk lengths from 1 to 128
  tokens, 32 greedy steps after each). The NVFP4 input of `down_proj` after the
  squared ReLU gave no NaN in prefill.

Why it does not help: the 4B decoder at batch 1 is only partly
weight-bandwidth bound. FP8 already gives 1.10x to 1.21x faster decode than
fp16; a large part of the step is the 16-bit `lm_head`, the attention output
projection and attention. NVFP4 halves the projection weights again but adds a
dynamic activation quantization per projection, which costs about as much as
it saves (the all-projection variant is slower than FP8). Revisit for batched
decode or a recipe that keeps the KL within 2x of FP8.

The engines of this family give wrong logits when `max_sequence_length` is 512
or more on TensorRT 11.4, also without quantization (unquantized fp16 KL 1.87
with a 512-token cache and 5.33 with a 4096-token cache, against 0.0008 with
the 256-token cache; TensorRT 11.1 does not build these engines). Quality is
therefore reported for the 256-token cache only; the 4096-token decode times
are kernel timings of those engines.
