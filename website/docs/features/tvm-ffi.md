---
title: TVM FFI Kernel Bridge
---

TensorRT-Model-Connect can call an explicitly named
[Apache TVM FFI](https://tvm.apache.org/ffi/) kernel from a TensorRT plugin
layer. The current implementation is intentionally direct: there is no graph
snapshot format, Recipe registry, automatic region selector, binding manifest,
or ABI/content hash.

The canonical implementation and runnable examples live in
[`examples/byok/`](https://github.com/NVIDIA/TensorRT-Model-Connect/tree/main/examples/byok).

## Path 1: add a kernel in a family

A family that owns a custom operation can add the layer while constructing its
TensorRT graph:

```python
from tensorrt_model_connect.byok import add_kernel

output, = add_kernel(
    network,
    plugin_library="/absolute/path/libtrtmc_byok_tvm_ffi.so",
    kernel_name="my_family.residual_add",
    inputs=[hidden, residual],
    output_specs=[{"dims": [256, 768], "dtype": "float16"}],
)
```

The call belongs in `families/<family>/**`. The family owns the TensorRT graph,
the external kernel contract, and the tests. It does not make the kernel or
model helper shared.

## Path 2: use the direct graph transform

An application can replace a region before serialization through the one public
`graph_transform` callback:

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
        inputs=[producer.get_output(0)],
        output_specs=[{"dims": [256, 768], "dtype": "float16"}],
    )
    consumer.set_input(0, replacement)

build(BuildRequest(..., graph_transform=replace))
```

The application chooses the live TensorRT layers and reconnects every consumer.
An invalid transform raises and aborts bundle publication; core does not retry
the unmodified graph.

## Load the external kernel

The serialized engine records the plugin layer, while the callable TVM-FFI
function remains in an external DSO. Load it explicitly before loading the
bundle:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --byok-library ./residual_add.so \
  --byok-function run \
  --byok-name my_family.residual_add \
  --prompt "Hello"
```

All three BYOK arguments are required together. The runtime opens
`libtrtmc_byok_tvm_ffi.so` from the explicit runtime root, then loads the named
function from the requested DSO.

## Kernel contract

The family or application supplies ordered inputs, output shapes and dtypes,
optional workspace bytes, and optional fixed arguments. The kernel author must
implement that exact order. Model Connect validates the explicit shape
description needed to create the plugin; it does not infer a signature from the
DSO or calculate a hash.

## Current limits

- The optional TVM-FFI bridge must be enabled at build time.
- Region selection is manual application or family code over the live TensorRT
  network.
- The transform runs before each engine serialization and receives a zero-based
  engine index.
- A transform must reconnect the graph in place and is not a portable graph IR.
- Runtime loads only explicitly named DSOs and functions.

For a worked lab, see
[Bring Your Own Kernel with TVM FFI](../tutorials/advanced/bring-your-own-kernel.md).
