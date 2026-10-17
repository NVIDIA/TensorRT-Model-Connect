# RoBERTa encoder tasks

The family supports `encoding` (the first token's hidden state) and `embedding`
(mean pooling followed by L2 normalization). The graph emits encoder hidden
states; it does not load or execute a sequence-classification scoring head.

`reranking` is therefore unsupported. Builds reject that task before checkpoint
loading, and the native factory rejects older bundles labeled `reranking`
before engine deserialization. A hidden-state coordinate is not a reranking
logit. Existing encoding and embedding bundles keep their current task IDs.

Implementing a reranker requires its checkpoint's scoring head, query/document
tokenization and official-reference validation. Loading an encoder backbone
from a sequence-classification checkpoint does not provide those behaviors.
