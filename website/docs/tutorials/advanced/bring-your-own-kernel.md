---
title: "Bring Your Own Kernel with TVM FFI"
---

import Diagram from '@site/src/components/Diagram';

This lab uses the existing optional BYOK bridge without introducing a graph IR,
Recipe registry, generated binding format, or hash check.

## Learning objectives

By the end, you should be able to:

- build and run the checked-in TVM-FFI identity kernel;
- add an explicitly named kernel from one family-owned TensorRT graph;
- apply the same API through the one public pre-serialization graph transform;
- explain why the family or application owns the boundary and validation.

<Diagram
  src="/img/diagrams/tutorials/advanced/byok-workflow.svg"
  alt="A family or application adds an explicit TVM-FFI plugin layer before TensorRT serialization, then the runtime loads the named kernel DSO"
  caption="BYOK is a one-way extension over public build and runtime APIs. Core never depends on an example or kernel implementation."
/>

## Before you start

Use a development environment with TensorRT, CUDA, and
`apache-tvm-ffi==0.1.12`. Start at the repository root.

```bash
python -m pip install -e .
cmake -S . -B build \
  -DTRTMC_BUILD_EXAMPLES=ON \
  -DTRTMC_ENABLE_BYOK=ON
cmake --build build --target trtmc_byok_identity_copy test_byok_tvm_ffi
```

## Level 1: run the existing identity-kernel proof

The checked-in test loads `identity_copy_kernel.so`, builds a TensorRT engine
containing `TvmFfiKernel`, deserializes it, executes it on CUDA, and compares the
output.

```bash
ctest --test-dir build -R '^byok_tvm_ffi$' --output-on-failure
```

Read these files together:

```text
examples/byok/identity_copy_kernel.cpp
examples/byok/test_tvm_ffi_plugin.cpp
core/runtime/include/trtmc/byok.h
core/runtime/byok/
```

The example depends on the public BYOK API. No core or family target links the
example.

## Level 2: add a family-owned kernel

When a model requires a custom operation, add it directly while that family's
`model.py` constructs the TensorRT graph:

```python
from tensorrt_model_connect.byok import add_kernel

replacement, = add_kernel(
    network,
    plugin_library="/absolute/path/libtrtmc_byok_tvm_ffi.so",
    kernel_name="my_family.residual_add",
    inputs=[hidden, residual],
    output_specs=[{"dims": [256, 768], "dtype": "float16"}],
)
```

The family owns the ordered arguments, shapes, dtypes, engine wiring, kernel
DSO, and focused tests. Keep all model-specific code below
`families/<family>/`; do not create a shared model-kernel layer.

## Level 3: replace an existing region from an application

An application can pass one in-place callback through `BuildRequest`:

```python
from tensorrt_model_connect import BuildRequest, build
from tensorrt_model_connect.byok import add_kernel

def replace(network, engine_index):
    if engine_index != 0:
        return

    producer = network.get_layer(4)
    consumer = network.get_layer(7)
    replacement, = add_kernel(
        network,
        plugin_library="/absolute/path/libtrtmc_byok_tvm_ffi.so",
        kernel_name="my_family.replacement",
    inputs=[producer.get_input(0)],
        output_specs=[{"dims": [256, 768], "dtype": "float16"}],
    )
    consumer.set_input(0, replacement)

build(BuildRequest(
    model_dir=MODEL_DIR,
    output_path=BUNDLE,
    family="my_family",
    task="text_generation",
    precision="fp16",
    graph_transform=replace,
))
```

The callback runs immediately before each engine is serialized. It must select
and reconnect legal live TensorRT objects. If it raises, bundle publication
stops; the build does not silently continue with the original graph.

## Build a CuTe DSL example

Install the optional CuTe DSL dependencies and use the checked-in exporter:

```bash
python -m pip install -e '.[cutedsl]'
python examples/byok/export_cutedsl_residual_add.py \
  --output build/residual_add.so
```

Use the shape and dtype encoded by that example. A real family contribution
must replace those constants with the exact contract of its selected graph
boundary and prove it with a focused test.

## Load the function explicitly

Install the runtime, backend, family DSO, and BYOK bridge into one runtime root,
then name the external function at load time:

```bash
cmake --build build --parallel
cmake --install build --prefix "$PWD/build/install"

trtmc run model.bundle \
  --runtime-root "$PWD/build/install/lib" \
  --byok-library "$PWD/build/residual_add.so" \
  --byok-function run \
  --byok-name my_family.residual_add \
  --prompt "Hello"
```

All three BYOK arguments are required. There is no library search or generated
binding manifest.

## Validate the replacement

Use two controls:

1. run the original family graph and record its semantic output;
2. run the BYOK graph with the same model, request, and hardware.

Then add a focused unit test for the tensor contract and a family-owned E2E for
the user-visible task. A load success alone does not prove numerical parity or
performance.

## Current graph-hook limits

- The hook is one callback over the live TensorRT network.
- It receives only the network and zero-based engine index.
- Selection, connectivity, and output replacement are the caller's job.
- Graph transforms cannot run concurrently in one Python process.
- Runtime uses an explicitly named DSO and exported function.
- Model Connect does not calculate a graph, ABI, source, or DSO digest.

## Self-check

1. Which directory owns a kernel required by only one model family?
2. Why must an application reconnect every consumer before serialization?
3. What does the checked-in identity test prove, and what does it not prove?
4. Why is the external function named explicitly at runtime?

See [TVM FFI Kernel Bridge](../../features/tvm-ffi.md) and the
[checked-in BYOK example](https://github.com/NVIDIA/TensorRT-Model-Connect/tree/main/examples/byok).
