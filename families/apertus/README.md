# Apertus

`ApertusForCausalLM` (for example `swiss-ai/Apertus-8B-Instruct-2509`) for the
`text_continuation` task: one engine with a prefill profile (up to 64 tokens per launch) and a
one-token decode profile over a fixed-capacity KV cache, built in `bf16` (default) or `fp32`.

## Quantization

`--precision bf16 --quantization fp8` builds an FP8 (W8A8) engine from the original BF16
checkpoint. The default stays `none`; unquantized builds are unchanged.

```
trtmc build swiss-ai/Apertus-8B-Instruct-2509 -o apertus-8b-fp8.bundle \
    --revision b946d40447b2b597999b9c86d44bee0b452c919f \
    --task text_continuation --precision bf16 --quantization fp8 --max-sequence-length 16384
```

Recipe (no ModelOpt FP8 or NVFP4 checkpoint of Apertus exists; the public quantized
repositories are compressed-tensors exports, which TensorRT cannot consume):

| Part | Format |
|---|---|
| q, k, v, attention output and MLP up and down projections of every layer | FP8 weight (`amax / 448` per projection, computed at build time) and static per-tensor FP8 activation scale (`fp8_activation_scales.json`) |
| Down projections of layers 2, 28, 29 and 31, output projections of layers 30 and 31 | BF16: their input scale is more than 10 times the median of the same projection (massive activations) |
| Embedding, norms, attention core (FP32 scores and softmax), xIELU, `lm_head` | BF16 / FP32 as in the unquantized engine |

The activation scales were calibrated once with `calibrate_fp8_scales.py` (ModelOpt
`FP8_DEFAULT_CFG`, max calibration, 544 texts of `cnn_dailymail` and
`nvidia/Nemotron-Post-Training-Dataset-v1` up to 1024 tokens and 16 sequences of 4096 tokens).
Only Apertus-8B (`apertus-h4096-l32-i21504-kv8-v131072`) is calibrated; another model config
raises and names the tool. A checkpoint that is already quantized is refused.

Measured on Apertus-8B-Instruct-2509, `max_sequence_length` 16384, TensorRT 11.4, batch 1,
all variants built and timed in one job per GPU (decode with a CUDA graph, median of 7
interleaved rounds):

| | B200 BF16 | B200 FP8 | B300 BF16 | B300 FP8 |
|---|---|---|---|---|
| Plan (GiB) | 15.02 | 8.89 | 15.02 | 8.89 |
| Decode at position 128 (ms) | 3.71 | 2.92 (1.27x) | 3.79 | 2.84 (1.33x) |
| Decode at position 2048 (ms) | 4.37 | 3.67 (1.19x) | 4.45 | 3.50 (1.27x) |
| Decode at position 8192 (ms) | 6.38 | 5.97 (1.07x) | 6.49 | 5.53 (1.17x) |
| Prefill of 64 tokens at 0 (ms) | 4.56 | 3.66 (1.24x) | 4.40 | 3.42 (1.29x) |

Quality against the BF16 model in PyTorch (`transformers`), teacher-forced on 5193 WikiText-2
test tokens (B200 / B300; perplexity of the PyTorch model 6.00 / 6.02):

| Engine | KL mean | KL p99 | Top-1 agreement | Perplexity |
|---|---|---|---|---|
| BF16 | 0.0057 / 0.0058 | 0.041 / 0.051 | 0.973 / 0.973 | 5.96 / 5.96 |
| FP8 | 0.0292 / 0.0308 | 0.34 / 0.38 | 0.931 / 0.934 | 5.99 / 5.99 |

No non-finite logit or KV-cache value in any run, including 2048 random tokens, 1024 repeats
of one token, 4096 WikiText tokens and the manifest prompts (9.9k-token long context), each
followed by 48 decode steps.

### NVFP4: measured, not shipped

NVFP4 (W4A4: E2M1 weights with per-16 e4m3 block scales, `weight_scale_2 = amax / (6 * 448)`,
`add_dynamic_quantize` activations with the FP8 calibration amax / 6 as global scale) was
measured with the same harness. Decode is bound partly by the many small FP32 kernels of the
explicit attention, xIELU and norms, so 4-bit weights do not make it faster than FP8:

| NVFP4 layout (others FP8) | Plan (GiB) | Decode vs FP8 (B200 / B300) | KL vs FP8 KL |
|---|---|---|---|
| Down projections (best) | 7.88 | 0.98x / 0.97x | 2.22x / 2.11x |
| Up projections | 7.46 | 1.01x / 1.02x | 4.5x to 4.8x |
| Up and down projections | 6.35 | 0.98x / 0.98x | 6.1x to 6.5x |
| All projections | 5.80 | 0.94x / 0.93x | 9.7x to 10.4x |

The KL ratio is against the shipped FP8 build (KL 0.029 / 0.031); the rows other than the
first come from an earlier job with the same harness. The best layout is at the 2.2x limit for
the loss and gives no speed, only an 11% smaller plan, and drifts far from the reference on the
E2E text prompt (largest teacher-forced gap 9.5 to 11.8 logits against 0.30 to 1.45 for FP8).
Down or up layouts with the first and last 4 layers in FP8 were 2.7x to 3.8x of FP8.
