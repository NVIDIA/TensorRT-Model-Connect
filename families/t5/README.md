# T5

This family builds the original T5 encoder-decoder models for the
`text_generation` task as an encoder engine and a decoder engine
(`google-t5/t5-small` in the manifests; `t5-base`, `t5-large` and `t5-3b` load
through the same code path). Batch is 1. The decoder recomputes the
cross-attention K/V projections of the encoder output at every step. The
manifests build fp16 and fp32 with tensor parallelism (4 GPUs). Only t5-small
is finite in fp16; from t5-base on, the fp16 engine overflows, so those sizes
build in fp32.

## Quantization

`--quantization fp8` with `--precision fp16` (t5-small) or `--precision fp32`
(every size) quantizes the original checkpoint at build time (#1998). Encoder
and decoder q, k, v, cross-attention q, k, v and both feed-forward projections
of every layer get per-tensor FP8 weights (amax / 448) and static activation
scales from `fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`,
CNN/DailyMail and WMT16 prompts). Dequantize and the matmul run in the build
precision. The attention output projections, shared embedding, relative
position bias, RMSNorms and `lm_head` stay unquantized. Scales ship for
t5-small, t5-base, t5-large and t5-3b. BLEU and ROUGE stay within 0.8 point of
the unquantized engine of the same precision.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. The large FP8 speedups in #1998
(t5-3b decoder step 2.59x on B200 and 2.30x on B300) are against fp32 CUDA-core
GEMMs, the only usable unquantized precision for t5-base and larger, not against
fp16. The FP8 engine is not bound by weight reads (numbers from #1998, t5-3b,
B200, TensorRT 11.4, batch 1):

| Item | t5-3b |
|---|---|
| FP8 decoder plan | 2.37 GB |
| Time to read it at about 8 TB/s | 0.30 ms |
| FP8 decoder step | 2.66 ms |
| Weight reads as share of the step | 11% |
| FP8 GEMM and GEMV time, decoder step / encoder call | about 1.1 ms of 2.66 ms / 0.87 ms of 4.2 ms |

Halving the quantized GEMM time would save at most about 0.55 ms per decoder
step (20%) and 0.45 ms per encoder call (10%). The rest of the time is the fp32
attention output projections, attention and small kernels. Revisit together
with a TensorRT release that can quantize the attention output projections.
