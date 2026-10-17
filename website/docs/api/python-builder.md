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

## Bundle output staging

By default, `BundleWriter` stages separate section files and copies them into
the final atomic output. A family that emits sections in order can opt into
single-file staging before opening its first section:

```python
writer.enable_direct_staging(header_reserve_bytes=64 * 1024)
```

This reserves a bounded, whitespace-padded JSON header in one temporary
format-v1 bundle. Each section is appended once; `finish()` fills the header
and atomically renames that same file. Section bytes, offsets relative to the
payload, core provenance and the reader contract are unchanged. Reserved
capacity must be a positive integer no larger than the existing 100-MiB
header limit; a header that does not fit fails before publication.

Direct staging permits one open section at a time. Section handles support
forward writes and section-relative `tell()`, not seeking, reading or exposing
the underlying descriptor. Existing builds keep their default behavior unless
their family explicitly selects this mode. `abort()` discards only owned
staging files, preserving any previously published destination.

For a fresh output, payload disk occupancy no longer includes a separate
staged copy plus a final copy. Replacing an existing bundle still requires
retaining that old file until atomic publication. Checkpoints, compiler
scratch, serialized-plan host buffers and GPU residency are separate costs;
single-file staging does not claim to reduce them.

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
