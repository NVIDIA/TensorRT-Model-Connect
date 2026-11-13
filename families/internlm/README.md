# InternLM

This family builds InternLM2 decoders for `text_generation`: pre-RMSNorm, a
fused group-interleaved `wqkv` projection (split into q, k and v at load),
SwiGLU MLP, RoPE and GQA. The manifests cover
`internlm/internlm2-math-plus-1_8b` (tokenizer from
`internlm/internlm2-step-prover`). It builds a prefill engine (`prefill.plan`,
dynamic prompt length) and a single-token decode engine (`engine.plan`) with a
static KV cache of `max_sequence_length` tokens (default 256). Precision is
fp32, fp16 or bf16, batch 1, with optional tensor parallelism for unquantized
builds.

## Quantization

`quantization="fp8"` self-quantizes the original BF16 checkpoint at build time
(`quantization.py`): the q, k, v, gate, up and down projections of every layer
run as FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` from the loaded
weights, static per-tensor activation scale from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG` calibration, fp32 Dequantize). The `down_proj` of
layers 6, 22 and 23 stays 16 bit (massive activation at its input, `null` in
the scale file). The attention output projection, `lm_head`, the embedding and
the norms stay 16 bit. Only the model config `internlm2-h2048-l24-i8192`
(internlm2-math-plus-1_8b) has calibrated scales. FP8 requires fp16 or bf16
and no tensor parallelism. No other quantization is accepted.

### NVFP4

NVFP4 was built and measured for internlm2-math-plus-1_8b with a quick screen
(TensorRT 11.4, B200 and B300, fp16 and bf16). It is not shipped: decode is at
most 1.02x the FP8 build on B200 and 0.81x to 1.00x on B300, and the quality
loss is 7x to 9x the FP8 loss (KL against the PyTorch model), far above the 2x
limit. The full study was not run. No ModelOpt NVFP4 checkpoint of InternLM2
exists, so the weights were self-quantized from the BF16 checkpoint. The NVFP4
code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the committed FP8 activation scale divided by 6
(same calibration data and statistic). Projections that are not NVFP4 keep the
FP8 layout of `quantization.py`; the three excluded `down_proj` layers stay 16
bit in every variant. The fused gate/up dual-GEMM kernel stays on for NVFP4
(the FP8 build turns it off). The plan inspector shows FP4 GEMMs for every
NVFP4 projection.

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| NVFP4 (same projections as FP8) | q, k, v, gate, up, down | none |
| MLP only | gate, up, down | q, k, v |

Results: `max_sequence_length` 256 (manifest config), engine only, batch 1.
Decode is one token at position 100 with CUDA graph replay (eager launch in
brackets), median of 7 interleaved rounds, all variants built and timed in the
same job. KL is KL(PyTorch bf16 || engine) per token, teacher-forced over 531
wikitext-2 test tokens (3 windows of 240 tokens, 64-token prefill then
one-token decode), measured on the bf16 builds (see the fp16 note below). NaN
counts non-finite logits and K/V values in the NaN scan.

| GPU | TensorRT | Precision | Decode unquantized | Decode FP8 | Decode NVFP4 (x FP8) | Decode MLP only (x FP8) | KL unquantized / FP8 / NVFP4 / MLP only | NaN |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | fp16 | 1.396 ms | 1.224 ms | 1.263 ms, 0.97x (0.93x) | 1.204 ms, 1.02x (1.01x) | see bf16 | 0 |
| B200 | 11.4 | bf16 | 1.489 ms | 1.272 ms | 1.282 ms, 0.99x (0.97x) | 1.255 ms, 1.01x (1.02x) | 0.0010 / 0.0173 / 0.1544 (8.9x) / 0.1163 (6.7x) | 0 |
| B300 | 11.4 | fp16 | 1.283 ms | 1.210 ms | 1.395 ms, 0.87x (0.81x) | 1.222 ms, 0.99x (0.97x) | see bf16 | 0 |
| B300 | 11.4 | bf16 | 1.394 ms | 1.237 ms | 1.415 ms, 0.87x (0.81x) | 1.238 ms, 1.00x (0.93x) | 0.0011 / 0.0169 / 0.1604 (9.5x) / 0.1171 (6.9x) | 0 |

- Next-token top-1 agreement with PyTorch (B200, bf16): unquantized 98.1%, FP8
  92.7%, NVFP4 80.2%, MLP only 83.2%. Perplexity (PyTorch 35.24): 35.25,
  35.55, 38.78, 38.26.
- Prefill of 128 tokens (fp16): NVFP4 1.02x FP8 on B200, 0.92x on B300.
- Plan size, prefill plus decode (fp16, B200): unquantized 7.05 GiB, FP8 4.52
  GiB, NVFP4 3.42 GiB, MLP only 3.58 GiB. The size gain alone is not a reason
  to ship.
- NaN scan: 0 non-finite logits or K/V for every variant, both GPUs, fp16 and
  bf16 (random tokens, one repeated token and wikitext text from the start of
  the test split through the prefill and decode plans, mixed chunk lengths, 32
  greedy steps after each).
- A first fp16 screen job on each GPU gave the same result (eager decode NVFP4
  0.92x FP8 on B200, 0.81x on B300).

fp16 note: this checkpoint has a massive activation in the residual stream
after layer 6 for the first token of a sequence. When a sequence starts with an
ordinary text token instead of BOS, its value can exceed the fp16 maximum
(65504; up to 74240 measured in the bf16 PyTorch model), and both the PyTorch
model in fp16 and the fp16 engines (unquantized and FP8) return NaN for the
whole sequence (2 of 3 KL windows here). This is why the KL above is measured
with bf16 builds. The NaN scan inputs above stayed finite in fp16.

Why it does not help: internlm2-1.8B at batch 1 is only partly
weight-bandwidth bound. FP8 already removes most of the weight traffic (1.06x
to 1.17x faster decode than the unquantized engine with CUDA graph); NVFP4
halves the remaining projection weights again but adds a dynamic activation
quantization per projection, which costs as much as it saves or more. Earlier
small-decoder studies (gpt2, opt, bloom, xglm, gpt_neox, codegen) found the
same. Revisit for a larger InternLM2 model, batched decode, or a recipe that
keeps the KL within 2x of FP8.

The decode plan of this family gives wrong logits when `max_sequence_length`
is 512 or more on TensorRT 11.4, also without quantization (prefill plan
correct). Long-context results are therefore not reported here.
