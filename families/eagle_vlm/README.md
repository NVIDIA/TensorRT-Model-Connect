# Eagle VLM

This family builds the Llama-Nemotron embed-vl and rerank-vl models
(`nvidia/llama-nemotron-embed-vl-1b-v2` for `text_to_embedding`,
`nvidia/llama-nemotron-rerank-vl-1b-v2` for
`text_query_documents_to_relevance` in the manifests). The text backbone is
Llama 3.2 1B (16 layers, hidden size 2048); the bundle holds the text engine.
Embedding mode pools and L2-normalizes the hidden states; reranking mode scores
a query and document pair. Each call is one forward pass, with no token-by-token
decode. The manifests build fp16 and fp32 with tensor parallelism (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` builds from the ModelOpt FP8 checkpoints
`nvidia/llama-nemotron-embed-vl-1b-v2-fp8` and
`nvidia/llama-nemotron-rerank-vl-1b-v2-fp8` (#1871). The checkpoint's FP8
weight bytes, weight scales and input scales of q/k/v and gate/up/down of every
layer feed the engine as they are. The attention output projection stays fp16
(its FP8 weight is dequantized at build time), and the vision tower, projector,
embedding, norms and score head stay unquantized. The FP8 engine is 1.18x to
1.19x faster than fp16 and the bundle is 37% to 44% smaller.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. Reasons (numbers from #1871,
B200, fp16 baseline built from the original checkpoint):

- No ModelOpt NVFP4 checkpoint exists for these models. The family uses the
  published checkpoint scales as they are.
- Each call is one forward pass over the whole input (512 tokens for embed,
  batch 2 x 256 tokens for rerank in the table), not a one-token decode, and
  the FP8 gain is already modest because attention, the attention output
  projection, norms and the score head stay fp16 or fp32.

| Model | TensorRT | Shape | fp16 ms | FP8 ms | FP8 vs fp16 | Bundle fp16 / FP8 |
|---|---|---|---|---|---|---|
| embed | 11.1 | 512 tokens | 1.478 | 1.244 | 1.19x | 2.32 / 1.48 GiB |
| embed | 11.4 | 512 tokens | 1.548 | 1.314 | 1.18x | 2.32 / 1.48 GiB |
| rerank | 11.1 | batch 2 x 256 tokens | 2.586 | 2.174 | 1.19x | 2.63 / 1.48 GiB |
