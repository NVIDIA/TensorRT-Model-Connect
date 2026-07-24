# TensorRT-Model-Connect Wiki Archive

:::info Archive boundary

The Wiki section preserves design history and deep background. Current user
and contributor contracts live in Getting Started, Architecture, Features,
Reference, Extend, and Operations. A Wiki page must not override a live
`MODEL.toml`, CLI parser, workflow, or test.

:::

## Current architecture in one view

TensorRT-Model-Connect has two phases:

1. The Python package resolves a Hugging Face checkpoint, selects a
   family-owned builder, creates TensorRT engines, and writes a `.trtfb`
   bundle.
2. The C++ runtime reads the bundle's exact `runtime_strategy`, loads the
   owning model DSO, resolves its registered `IPipelinePlugin`, and serves the
   operation through `IPipeline`.

Model ownership is expressed by three matching descriptors:

- `python/tensorrt_model_connect/families/<family>/MODEL.toml`
- `src/runtime/models/<family>/MODEL.toml`
- `tests/e2e/models/<family>/MODEL.toml`

The generic operation contract is `task_strategy`; the implementation-specific
runtime contract is normally a family-qualified `runtime_strategy`.

## Use these current pages

- [Architecture Overview](../architecture/overview.md)
- [Runtime Plugins](../architecture/runtime-plugins.md)
- [Bundle Format](../architecture/bundle-format.md)
- [Source Layout](../reference/source-layout.md)
- [Testing Reference](../reference/testing.md)
- [Add a Model Family](../extend/add-model-family.md)
- [Add a Runtime Strategy](../extend/add-runtime-strategy.md)

The remaining pages in this section are retained for context. Each page now
states whether it is current guidance, a point-in-time snapshot, or a proposal.
