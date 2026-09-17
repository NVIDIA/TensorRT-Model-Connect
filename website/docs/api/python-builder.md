---
title: Python Build API
---

`BuildRequest` is the resolved low-level build contract used after CLI model
discovery. It contains the family and non-empty task selected from family-owned
support. The shared Python API also exposes `build()`, `BundleWriter`, and one
optional build-time graph transform.

```python
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build

build(BuildRequest(
    model_dir=Path("/models/gpt2"),
    output_path=Path("gpt2.bundle"),
    family="gpt2",
    task="text_generation",
    precision="fp16",
))
```

The core loads one exact family module and calls its plain
`build(request, writer)` function. It does not retry another family. Normal
users use the CLI with a Hugging Face ID; family authors and tests may call this
resolved API directly.

`model_dir` is already local at this boundary. The selected family alone
decides whether that directory is a Hugging Face snapshot or a prepared
checkpoint; `BuildRequest` does not perform another discovery pass.

## Optional execution inputs

`build(request, execution=...)` accepts an optional, frozen
`BuildExecutionInputs` descriptor. It contains a family-owned `variant` string
and a tuple of `NamedCheckpoint(role, model_dir)` descriptors. Import these
public types from `tensorrt_model_connect`. Companion directories must already
exist locally; the core does not download them or infer compatible model pairs.
Roles must be unique. Variant and role names are lowercase identifiers.

Providing execution inputs requires the selected family to implement
`build_with_inputs(request, writer, execution)`. The family validates the
variant, checkpoint roles, compatibility and execution semantics. A missing
hook fails before bundle creation; the core never substitutes ordinary
base-only generation or another variant. With no execution inputs, the
existing `build(request, writer)` family call is unchanged.

The build CLI exposes the same optional contract:

```text
trtmc build LOCAL_TARGET -o model.bundle \
  --execution-variant FAMILY_VARIANT \
  --companion ROLE=LOCAL_COMPANION_DIR
```

Replace the uppercase placeholders with values from the selected family's
recipe; they are not literal supported identifiers. Repeat `--companion` only
for distinct roles. A companion requires `--execution-variant`; URLs and
implicit companion downloads are unsupported. The generic API does not itself
qualify any speculative algorithm or checkpoint pair.

## Optional graph transform

`BuildRequest.graph_transform` is an in-place callback invoked on the completed
TensorRT network immediately before each engine is serialized. The second
argument is a zero-based engine index, so a multi-engine family can be targeted
without adding family names or roles to core.

```python
def replace_subgraph(network, engine_index):
    if engine_index != 0:
        return

    # Inspect network.get_layer(...), add replacement layers, then reconnect
    # the consumers with consumer.set_input(...).

build(BuildRequest(
    model_dir=Path("/models/gpt2"),
    output_path=Path("gpt2.bundle"),
    family="gpt2",
    task="text_generation",
    precision="fp16",
    graph_transform=replace_subgraph,
))
```

The callback receives the live TensorRT object and may select and replace any
subgraph TensorRT can express. It must reconnect the replacement in place and
raise on an invalid graph; a failure stops serialization and aborts bundle
publication. Normal builds do not install the hook. There is no graph IR,
registry, fingerprint, hash, fallback, or runtime Python path.

`tensor_parallel_size` and `context_parallel_size` are direct request fields,
not an options bag. Every family must either implement the requested value or
reject it. Cosmos3 accepts context parallel size 1 or 2. FLUX and Wan accept
1, 2, 4, or 8; their family-owned builders and runtimes reject simultaneous
tensor and context parallelism. Other families require the default value 1.
