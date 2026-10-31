# RoBERTa runtime input capacity

A fixed-sequence bundle accepts at most the number of token IDs declared by its
`input_ids` engine binding. The limit includes tokenizer-added special tokens.

Native `embed`, `encode`, `rerank`, and token-ID encoding reject inputs above
that capacity with an invalid-argument error before submitting work to the
engine. Shorten or split longer text before invoking the bundle. Dynamic-input
engines retain their existing profile validation.
