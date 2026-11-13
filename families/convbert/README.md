# ConvBERT

This family builds ConvBERT encoders (`YituTech/conv-bert-base` in the
manifests) as one TensorRT engine for the `encoding` task. Each layer mixes
self-attention on half of the heads with span-based dynamic convolution on the
other half, followed by a GELU MLP (post-norm). Batch is 1. The manifests build
fp16, and fp32 with tensor parallelism (2 GPUs); the family default is fp32.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2077). Query, key, value, the convolution value projection
`conv_out_layer`, `intermediate.dense` and `output.dense` of every layer get
per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, WikiText). The
attention output projection, the dynamic-kernel projection, the convolutions,
embeddings, layer norms, attention and GELU stay fp16. B200 and B300, TensorRT
11.4, batch 1: FP8 is 1.01x to 1.04x faster than fp16, the plan is 33% smaller
(202.9 to 135.8 MiB), and the SST-2 probe and STS-B scores stay at the fp16
level.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. ConvBERT at batch 1 is not bound
by weight reads (numbers from #2077, TensorRT 11.4, batch 1):

| Item | Value |
|---|---|
| FP8 projection weights | 71 MB |
| Time to read them at about 8 TB/s | about 0.01 ms |
| fp16 engine call, seq 128 to 512 (B200, B300) | 0.71 to 1.21 ms |
| Weight reads as share of the call | about 1% |
| FP8 vs fp16 | 1.01x to 1.04x |

4-bit weights could save at most the weight-read time, and NVFP4 adds dynamic
activation quantize kernels.
