# MPNet

This family builds MPNet encoders as one TensorRT engine for the
`text_to_pooled_features` and `text_to_embedding` tasks. The manifests use
`sentence-transformers/all-mpnet-base-v2` with a 384-token window and build
fp16, bf16 and fp32 with tensor parallelism (4 GPUs). Batch is 1. The window is
static, so every call computes the full window.

## Tokenizer validation

The CPU CTest target `mpnet_tokenizer` checks Unicode cleaning, lowercasing,
accent stripping, punctuation boundaries, normalizer flags and special-token
framing through the native tokenizer factory. The tables are generated from
`tokenizers==0.23.3`; regenerate them with
`python families/mpnet/tests/generate_lowercase.py`,
`python families/mpnet/tests/generate_punctuation.py` and
`python families/mpnet/tests/generate_accents.py` in an environment with that
version installed. Accent data preserves removed starter boundaries for
canonical ordering; Hangul decomposes algorithmically. The runtime does not
depend on Python or `tokenizers`.
Unicode-derived mapping data retains the upstream copyright and permission
notice in [UNICODE_LICENSE.txt](UNICODE_LICENSE.txt).

## Scoped checkpoint validation

At `11315b08`, the pinned all-mpnet-base-v2 revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130` passed its original input and the
Unicode clean-text E2E. The real FP16 TensorRT CLI was compared to the official
FP32 CLS reference, retaining the 0.999 cosine gate. Observed cosine values were
0.999988 and 0.999994, respectively. Public C/C++ consumers also passed task,
dimension, finiteness and C/C++ agreement checks; those consumer checks do not
compare their vectors numerically to the CLI or reference.

Validation used one H200, driver 595.71.05, CUDA 13.4.3, TensorRT 11.4.0.106,
PyTorch 2.12.0+cu130, Transformers 5.2.0 and Tokenizers 0.22.2 on Linux x86_64.
Run `families/mpnet/tests/test_e2e.py` with `--e2e-testcase
all-mpnet-base-v2,all-mpnet-base-v2-unicode-clean-text` and the required native
runtime paths. This does not qualify other precisions, quantization, TP4,
checkpoints, platforms, long inputs, concurrency or performance.

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2072). Query, key, value, `intermediate.dense` and `output.dense` of every
layer get per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, WikiText). The
attention output projection, embeddings, relative position bias, layer norms,
attention and GELU stay 16-bit. Scales ship for `all-mpnet-base-v2` and
`microsoft/mpnet-base`. TensorRT 11.4, batch 1, seq 128 to 512: FP8 is 1.03x to
1.07x faster than fp16, the plan is 34% to 35% smaller, and STS-B Spearman is
83.41 to 83.43 (PyTorch 83.42).

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. MPNet at batch 1 is not bound by
weight reads (numbers from #2072, B200 and B300, TensorRT 11.4):

| Item | Value |
|---|---|
| FP8 projection weights | 78 MB |
| Time to read them at about 8 TB/s | about 0.01 ms |
| FP8 engine call, batch 1, seq 384 to 512 | 0.41 to 0.54 ms |
| Weight reads as share of the call | about 2% |
| FP8 vs fp16, batch 1, seq 128 to 512 | 1.03x to 1.07x |
| Five quantized projections, fp16 to FP8 (B300, seq 384, profiler) | 0.240 to 0.209 ms |
| Extra Quantize and element-wise work in the FP8 engine | +0.030 ms |

4-bit weights could save at most the weight-read time. The GEMMs at 384 to 512
rows are compute bound, and NVFP4 adds one dynamic quantize kernel per
projection.
