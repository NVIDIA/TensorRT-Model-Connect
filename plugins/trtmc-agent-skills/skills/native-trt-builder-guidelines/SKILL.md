---
name: native-trt-builder-guidelines
description: >-
  Use when modifying TensorRT builders, family-owned graph construction
  helpers, or runtime strategy graph code in TensorRT-Model-Connect. Enforces
  strongly typed TensorRT networks, reusable family-local primitives, and
  compact GQA/MQA cache handling while preserving model ownership.
---

# Native TRT Builder Guidelines

## Principles

- Builders must create strongly typed networks with
  `builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))`.
- Basic TensorRT primitives should use the owning family's
  `python/tensorrt_model_connect/families/<family>/graph_ops.py` or
  `graph_blocks.py` when the tensor contract and semantics match. There is no
  supported root-level shared graph-helper module.
- TensorRT `IAttentionLayer` supports GQA/MQA. Do not expand K/V projections,
  K/V bias, or cache tensors to query-head width solely for attention; prefer
  compact K/V width: `num_key_value_heads * head_dim`.
- Keep model-specific dataflow local when it is truly architecture-specific:
  ordering, cache handling, bias terms, multimodal/task-specific behavior, and
  other non-shared variants do not need forced abstraction.
- If a native TensorRT primitive is insufficient for a basic operation, keep
  the fallback narrow and document the missing TensorRT capability near the
  implementation.

## Workflow

1. Check network creation with `rg -n "create_network\\("`; every builder path
   should use `NetworkDefinitionCreationFlag.STRONGLY_TYPED`.
2. Search the owning family for duplicated primitive logic before adding new
   helpers. Keep new graph helpers family-local unless a separate,
   model-independent shared contract and its cross-family tests justify a
   shared abstraction.
3. For decoder GQA/MQA paths, verify K/V projection weights, K/V bias tensors,
   and cache tensors remain at compact K/V width unless a non-attention
   primitive explicitly requires a different layout.
4. Read the affected builder and helper code before refactoring; do not infer
   semantics from grep hits alone.
5. Validate with `git diff --check` and the narrowest compile or test command
   that exercises the changed builder path.

## Review Checks

- Strongly typed network creation is preserved.
- Elementwise inputs have matching dtypes.
- New helper signatures describe tensor shape, dtype, and layout assumptions.
- Runtime metadata and E2E manifests still describe the selected
  `runtime_strategy` and precision accurately.
- Any model-specific local code has a concrete reason to remain local.
