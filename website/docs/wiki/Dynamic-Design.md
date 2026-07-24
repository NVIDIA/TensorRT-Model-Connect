# Dynamic Design

Status: current contract-level sequences.

## Build sequence

```mermaid
sequenceDiagram
  participant User
  participant CLI as build_cli.py
  participant Families as families registry
  participant Plugin as family plugin
  participant TRT as TensorRT builder
  participant Bundle as bundle_writer.py

  User->>CLI: build checkpoint
  CLI->>Families: resolve model metadata
  Families-->>CLI: family package
  CLI->>Plugin: load config and weights
  Plugin->>TRT: construct and compile network
  TRT-->>Plugin: serialized engine plan
  Plugin->>Bundle: engine sections plus config
  Bundle-->>User: .trtfb
```

Family resolution is descriptor-driven. A family may own multiple engines or
special build phases; the diagram shows the common control boundary, not an
assumption that every model is a decoder.

## Pipeline creation sequence

```mermaid
sequenceDiagram
  participant Caller
  participant Factory as PipelineFactory
  participant Bundle
  participant Loader as Plugin loader
  participant Registry as PipelineRegistry
  participant Plugin as Model plugin

  Caller->>Factory: from_bundle(path, options)
  Factory->>Bundle: read metadata and config
  Bundle-->>Factory: runtime_strategy plus sections
  Factory->>Loader: load DSO for strategy
  Loader->>Registry: registration symbol registers plugin
  Factory->>Registry: lookup strategy
  Registry-->>Factory: IPipelinePlugin
  Factory->>Plugin: validate and create(context)
  Plugin-->>Caller: IPipeline
```

If the bundle omits `runtime_strategy`, creation succeeds only when generated
runtime-manifest data defines a default. An unknown strategy, missing DSO, or
missing registration fails explicitly.

## Request sequence

The caller invokes one operation from `IPipeline`. The family pipeline
validates request support, owns model state and GPU execution, and returns the
operation's typed result. Text generation may have prefill/decode and KV-cache
state; encoder, diffusion, audio, vision, segmentation, and time-series
pipelines follow their own model-owned sequences.

## Config sequence

Before plugin creation, schema defaults and available bundle/file/CLI/session
layers are resolved into a `ConfigBundle`. Platform-wide sinks are applied by
the factory; model schemas are consumed by their owning plugin. The C++ CLI
rejects invalid explicit `--config`/`--set` input before dispatch. A direct
`PipelineFactory` caller can currently receive only a warning and a fallback
to plugin-local behavior after resolution fails, so library callers must inspect
diagnostics or enforce fail-fast behavior themselves.
