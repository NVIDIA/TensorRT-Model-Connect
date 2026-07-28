# TensorRT-Model-Connect Wiki Archive

:::info Archive boundary

The Wiki section preserves design history and deep background. Current user
and contributor contracts live in Getting Started, Architecture, Features,
Reference, Extend, and Operations. A Wiki page must not override a live
`MODEL.toml`, CLI parser, workflow, or test.

:::

## Current architecture in one view

TensorRT-Model-Connect has two phases and two implementation paths:

1. The Python package resolves a Hugging Face checkpoint and family. An exact
   qualified optimized-runtime profile may delegate artifact construction to
   that family's adapter; otherwise the native family plugin creates TensorRT
   plans.
2. The C++ runtime recognizes the resulting bundle shape. A native bundle uses
   `runtime_strategy`, its owning model DSO, `IPipelinePlugin`, and a backend
   DSO. A bundle with `optimized_runtime.json` loads its exact embedded
   implementation DSO and bypasses native strategy/registry/backend dispatch.
   Both paths return `IPipeline`.

Native model ownership is expressed by three linked descriptors:

- `python/tensorrt_model_connect/families/<builder-family>/MODEL.toml`
- `src/runtime/models/<runtime-owner>/MODEL.toml`
- `tests/e2e/models/<e2e-family>/MODEL.toml`

Each descriptor `id` matches its own directory. The three names normally
match, but the exact `runtime_strategy` is the cross-tree link. For example,
builder/E2E owner `magpie_tts` maps to runtime owner `magpie`, and
`wan_t2v` maps to `wan`.

The generic operation contract is `task_strategy`. The native implementation
contract is normally a family-qualified `runtime_strategy`; an optimized
implementation instead uses an exact family-owned provider/profile contract.

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
