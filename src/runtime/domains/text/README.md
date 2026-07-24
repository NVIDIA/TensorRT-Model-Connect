# Text Generation Runtime Domain

Shared autoregressive text-generation runtime contracts live here. Model
families keep graph names, configuration defaults, and pipeline behavior under
`src/runtime/models/<family>/`; cross-family dynamic KV budgeting, planning,
allocation, binding, and qualification support lives in `dynamic_memory/`.

Qwen and Llama use the same dynamic-memory implementation. New text-generation
families should reuse that implementation instead of copying it into a model
folder or moving it into the model-agnostic runtime core.
