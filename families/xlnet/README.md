# XLNet

This family builds XLNet encoders (content stream only) as one TensorRT engine
for the `text_to_pooled_features` and `text_to_embedding` tasks
(`xlnet/xlnet-base-cased` in the manifests). Batch is 1. The default precision
is fp32; the manifests build fp16, FP8 and fp32 with tensor parallelism (4
GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2076). Both feed-forward projections of every layer get per-tensor FP8
weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (max calibration, WikiText-2 and SST-2 train
texts). Query, key and value stay fp16 (their input has a heavy tail; FP8 there
raised the KL from 0.008 to 0.045 with no speed gain), as do the relative
position projection, embeddings, layer norms, GELU and attention matmuls. The
attention output projection stays fp16 too: with one per-tensor scale on its
input, 10 of the 351 sentences of the AIPerf encoder-parity qualification were
outside cosine 0.99 / relative L2 0.15 against the float32 model; without it
all are within, and an FP8 engine call is 1.08x faster than fp16 instead of
1.09x (GB300, TensorRT 11.4, 256 tokens). FP8 with `--precision fp32` or `bf16`
raises. Scales ship for `xlnet-base-cased` and
`textattack/xlnet-base-cased-SST-2`. Measured in #2076 with the attention output
projection in FP8 (TensorRT 11.4, batch 1): FP8 was 1.02x to 1.13x faster than
fp16, the plan 27% smaller, and SST-2 accuracy 94.27% against 94.38% for fp16.
The current FP8 bundle of `xlnet-base-cased` is 24% smaller than the fp16 one.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. XLNet at batch 1 is not bound
by weight reads (numbers from #2076, B200 and B300, TensorRT 11.4):

| Item | xlnet-base-cased |
|---|---|
| FP8 projection weights | 64 MB |
| Time to read them at about 8 TB/s | about 0.01 ms |
| fp16 / FP8 engine call, batch 1, seq 128 to 512 | 0.510 to 0.960 / 0.489 to 0.918 ms |
| Weight reads as share of the call | 1% to 2% |
| FP8 vs fp16, batch 1, seq 128 to 512 | 1.02x to 1.13x |

4-bit weights could save at most the weight-read time, and NVFP4 adds one
dynamic quantize kernel per projection. On another small encoder family
(timm_swin), NVFP4 was slower than FP8 at batch 1 to 32 because of those
kernels (#2076).
