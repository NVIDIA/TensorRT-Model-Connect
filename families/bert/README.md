# BERT runtime tasks

The family implements `encoding` (first-token hidden features) and `embedding`
(mean-pooled, L2-normalized hidden features). Its CLI exposes `encode` and `embed`.

Reranking builds, legacy reranking bundles and the old `rerank` CLI command are
rejected. The encoder graph has no trained scoring head; the first hidden-state
coordinate is not a relevance score. Real reranking support requires the
checkpoint scoring head and matching query/document pair tokenization.

Encoding and embedding bundles keep their existing tasks and results. The CPU
capability and CLI tests use deterministic tensors; they do not establish
pretrained TensorRT/GPU parity or performance.

This family builds BERT encoders as one TensorRT engine for the `encoding` and
`embedding` tasks: `google-bert/bert-base-uncased` and the embedders
`sentence-transformers/all-MiniLM-L6-v2`,
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
`BAAI/bge-small-en-v1.5` and `intfloat/multilingual-e5-small` in the manifests.
Batch is 1. The manifests build fp16, bf16 (`bge-small-en-v1.5`) and fp32 with
tensor parallelism (`bert-base-uncased`, 4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original checkpoint at build
time (#2075). Query, key, value, `intermediate.dense` and `output.dense` of every
layer get per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, WikiText). The
attention output projection, embeddings, layer norms, attention and GELU stay
16-bit. Scales ship for `bert-base-uncased` and `bert-large-uncased`. TensorRT
11.4, batch 1, seq 512: FP8 is 1.04x to 1.07x faster than fp16, the plan is 35%
(base) and 41% (large) smaller, and SST-2 accuracy stays within 0.12 point of
PyTorch.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. BERT at batch 1 is not bound by
weight reads (numbers from #2075, B200 and B300, TensorRT 11.4):

| Item | base | large |
|---|---|---|
| FP8 projection weights | 78 MB | 277 MB |
| Time to read them at about 8 TB/s | about 0.01 ms | about 0.035 ms |
| fp16 engine call, batch 1, seq 128 to 512 | 0.36 to 0.56 ms | 0.77 to 1.22 ms |
| Weight reads as share of the call | 2% to 3% | 2% to 3% |
| FP8 vs fp16, batch 1, seq 512 | 1.04x to 1.07x | 1.04x to 1.06x |

4-bit weights could save at most the weight-read time. The GEMMs at 128 to 512
rows are compute bound, and NVFP4 adds one dynamic quantize kernel per
projection.
