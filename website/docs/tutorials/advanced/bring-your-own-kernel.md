---
title: Bring Your Own Kernel
---

# Bring your own kernel with TVM-FFI

BYOK is an optional one-way extension: an application or family builder can use
TRTMC's public TVM-FFI bridge, while shared TRTMC code never depends on an
example or a family-specific kernel.

## Build the bridge and identity example

```bash
python -m pip install -e .
cmake -S . -B build -DTRTMC_BUILD_EXAMPLES=ON -DTRTMC_ENABLE_BYOK=ON
cmake --build build --target trtmc_byok_identity_copy test_byok_tvm_ffi
ctest --test-dir build -R '^byok_tvm_ffi$' --output-on-failure
cmake --install build --prefix "$PWD/build/install"
```

The test loads an external kernel DSO, builds and serializes a TensorRT engine
containing `TvmFfiKernel`, deserializes it, executes it on CUDA, and verifies
the result.

## Add a kernel while building a family graph

```python
from tensorrt_model_connect.byok import add_kernel

output, = add_kernel(
    network,
    plugin_library="/absolute/path/build/install/lib/libtrtmc_byok_tvm_ffi.so",
    kernel_name="my_family.residual_add",
    inputs=[hidden, attention_projection],
    output_specs=[{"dims": [256, 768], "dtype": "float16"}],
)
```

The owning family chooses the mathematical region, boundary tensors, shapes,
dtypes, and validation. TensorRT still owns the rest of graph lowering and
execution planning.

## Application-provided graph transform

An application can pass an in-place transform through the public build request:

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

build(BuildRequest(..., graph_transform=replace))
```

The hook runs only during build. It must reconnect every external consumer
before serialization.

## Load the external kernel at runtime

Provide the DSO, exported function, and registered kernel name together:

```bash
trtmc run model.bundle \
  --runtime-root build/install/lib \
  --byok-library ./residual_add.so \
  --byok-function run \
  --byok-name my_family.residual_add \
  --prompt "Hello"
```

The current interface does not provide the former graph-discovery,
fingerprinting, selection-receipt, or `--graph-patch` CLI workflow. Kernel name,
function, tensor shape, dtype, DSO compatibility, and numerical evidence must
be explicit and versioned by the integrating application or family.

For the working identity and CuTe DSL residual-add examples, see
`examples/byok/README.md` in the repository.
