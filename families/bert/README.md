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
