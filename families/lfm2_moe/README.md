# LFM2 MoE

This family builds `LiquidAI/LFM2.5-24B-A2B` (`model_type` `lfm2_moe`) for
`text_continuation`: a hybrid decoder with 40 layers, 30 short-convolution
blocks and 10 grouped-query attention blocks, two dense SwiGLU layers and 38
routed MoE layers (64 experts, top-4 with sigmoid scores and an expert bias,
about 2B active of 24B parameters). The engine decodes one token per step with
a fixed state profile (`max_sequence_length` 2 to 1024, default 128); the
native runtime feeds the prompt token by token and decodes greedily with the
checkpoint's repetition penalty. Precision is fp32, batch 1, one GPU.

## Quantization

`quantization="fp8"` quantizes the routed experts of the original BF16
checkpoint at build time. The build stays fp32 everywhere else; the default
(`none`) build is unchanged.

```python
BuildRequest(model_dir=..., output_path="lfm2.5-24b-a2b-fp8.bundle", family="lfm2_moe",
             task="text_continuation", precision="fp32", max_sequence_length=128,
             quantization="fp8")
```

Recipe (W8A8, static per-tensor scales):

- LiquidAI's `LFM2.5-24B-A2B-FP8` uses 128x128 block weight scales and dynamic
  per-group activations (compressed-tensors), which TensorRT cannot run as an
  FP8 GEMM, so the weights are quantized from the BF16 checkpoint instead.
- Quantized: the fused gate/up (`w1`, `w3`) and down (`w2`) projections of
  all experts of every MoE layer (about 95 % of the weights). One weight
  scale per layer and projection, `max(abs(w)) / 448`. Activation scales
  `amax / 448` were calibrated once with `calibrate_fp8_scales.py` (BF16 model
  in PyTorch, 512 cnn_dailymail chat samples of up to 512 and 1024 tokens,
  the largest expert input per layer) and are stored in
  `fp8_activation_scales.json`. The down inputs of layers 9, 21, 34 and 37
  are more than 20 times the median and stay fp32.
- Not quantized: attention, the conv blocks, the two dense MLP layers, the
  router, the embedding, `lm_head` and the norms. The official FP8 layer list
  (attention q/k/v/o, dense MLP and experts) and adding the conv in/out
  projections were measured: 1 % to 3 % faster decode, larger error (below).
- Graph: the four selected experts are picked with dynamic slices from a
  constant of FP8 grid values, then `Quantize(1.0)` and `Dequantize(scale)`;
  TensorRT stores the experts as FP8 and runs the batched expert matmuls as
  FP8 GEMMs. Dequantize and matmul run in fp32.

Results: TensorRT 11.4, `max_sequence_length` 128 (the manifest config). Decode
is one engine step at position 64 plus the state copies, median of 9
interleaved rounds; request is a 5-token prompt and 64 new tokens. Quality is
teacher-forced over 6096 held-out tokens (32 wikitext-2 test windows and 16
cnn_dailymail test chat samples of 128 tokens) against the PyTorch model:
KL(reference || engine) per token, top-1 agreement and perplexity of the
held-out text.

| GPU | Decode fp32 | Decode FP8 | Request fp32 / FP8 |
|---|---|---|---|
| B200 | 7.94 ms | 4.88 ms (1.63x) | 556 / 349 ms (1.59x) |
| B300 | 8.02 ms | 5.05 ms (1.59x) | 561 / 359 ms (1.56x) |

| Engine | KL vs BF16 mean / p99 | Top-1 vs BF16 | KL vs FP32 | Perplexity |
|---|---|---|---|---|
| fp32 (default) | 0.081 / 0.98 | 87.8 % | 0.0045 | 81.07 |
| FP8 (experts) | 0.142 / 1.58 | 84.4 % | 0.136 | 81.84 |
| FP8, official layer list | 0.188 / 1.92 | 81.3 % | 0.191 | 81.6 to 81.7 |
| FP8, official list + conv in/out | 0.206 / 2.41 | 80.3 % | 0.205 | n/a |

PyTorch BF16 against PyTorch FP32 is KL 0.081 with 87.7 % top-1 (perplexity
82.70 and 80.93): the model reacts strongly to any change of precision, so the
KL of every quantized variant starts near 0.1 (only the two dense MLP layers in
FP8: 0.128), while perplexity stays within 1 %.

- Size: plan 89.3 GiB (fp32) and 27.4 GiB (FP8).
- At `max_sequence_length` 1024 (5337 tokens, windows of up to 1024):
  decode 8.27 -> 5.19 ms (B200) and 8.10 -> 5.13 ms (B300); KL vs BF16 0.063
  (fp32) and 0.098 to 0.099 (FP8), perplexity 30.60 and 29.6 to 29.8.
- TensorRT 11.1 (B200): the FP8 engine builds and runs, decode 8.25 -> 4.94 ms,
  KL vs BF16 0.139, perplexity 80.41.
- NaN scan: no non-finite logit for fp32 or FP8 on either GPU (quality runs,
  4 greedy prompts to the full profile, 128-token runs of random, the
  largest, one repeated and repeated-sentence token ids).
- E2E (`lfm2.5-24b-a2b-fp8`, premerge): every greedy FP8 token must be within
  `near_tie_logit_tolerance` 1.5 of the largest FP32 logit at its step. Measured
  0.68 on B200 and B300; activation scales x0.1 give 7.49 and x3 give 3.12.

Where the FP8 gain comes from (finding, not changed here): in the fp32 engine
the selected experts are picked with one Gather per projection, a slow kernel
(0.12 ms per MoE layer, 4.6 of 10.1 ms in a profiled decode step). Picking
them with dynamic slices, as the FP8 build does, gives the same logits and
decode 7.94 -> 6.66 ms (B200) and 8.02 -> 6.68 ms (B300). Against that fp32
engine FP8 is 1.36x (B200) and 1.38x (B300). A 16-bit build (bf16 weights and
GEMMs, fp32 norms, attention and state) decodes in 6.68 / 6.73 ms with the
Gather (1.19x) and 4.86 / 4.96 ms with slices, the same speed as FP8, at the
quality of the fp32 engine (KL vs BF16 0.084, perplexity 81.51, plan 44.7 GiB).

### NVFP4 (measured, not shipped)

NVFP4 W4A4 was built for the routed experts (block-16 e4m3 scales,
`weight_scale_2 = amax / (6 x 448)`, dynamic activation quantization with the
FP8 calibration amax / 6 as global scale). It does not meet the NVFP4 rules
(at least 0.8x FP8 speed on B200 and B300 and at most about 2x the FP8 loss):

| Layout | Decode vs FP8 (B200 / B300) | KL vs BF16 (x FP8) | Top-1 | Perplexity | Plan |
|---|---|---|---|---|---|
| W4A4, every expert in one batched GEMM, top-4 rows picked | 1.02x / 1.02x | 0.399 (2.8x) | 72.5 % | 82.75 | 15.8 GiB |
| W4A16, selected experts gathered and dequantized | 0.69x / 0.70x | 0.279 (2.0x) | 76.3 % | 84.4 | 17.2 GiB |

The W4A4 engine reads all 64 experts per token, which cancels the 4-bit
saving; picking FP4 experts before the GEMM is not possible (an FP4 constant
can only feed a Dequantize, and a static FP4 Quantize after a slice or gather
does not build). Its greedy text repeats words ("Paris. is is is ..."). With
the other projections FP8 as well (official list + conv) it is 1.06x faster
than that FP8 recipe on B300 at 2.0x its KL (0.412 vs 0.206), and 2.9x the KL
of the shipped FP8 build. Revisit with a TensorRT MoE path that runs block-scaled
FP4 GEMMs on the selected experts only.
