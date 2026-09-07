---
title: TVM-FFI Kernel Bridge
---

TVM-FFI BYOK is an explicit one-way extension. A family builder or application
uses the public BYOK API; the core never depends on an example or model-specific
kernel.

## Build the bridge

The project dependency supplies TVM-FFI. Configure with BYOK enabled and build
the bridge plus example test:

```bash
python -m pip install -e .
cmake -S . -B build -DTRTMC_ENABLE_BYOK=ON -DTRTMC_BUILD_EXAMPLES=ON
cmake --build build --target trtmc_byok_identity_copy test_byok_tvm_ffi
ctest --test-dir build -R '^byok_tvm_ffi$' --output-on-failure
```

## Add a kernel to a family graph

The selected family can add a named TVM-FFI TensorRT plugin layer directly:

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

The family reconnects the returned tensor into its graph before serialization.
The kernel name accepts only letters, digits, `_`, `.`, `@`, and `-`; inputs,
outputs, workspace, shapes, dtypes, and optional scalar/null arguments are
explicit.

## Application graph transform

An application can pass `BuildRequest.graph_transform`, an in-place
`transform(network, engine_index)` callback. The callback inspects the live
TensorRT network, adds the BYOK layer, and reconnects all external consumers.
It runs immediately before each engine is serialized. Transform failures abort
the build; transforms cannot run concurrently in one process.

There is no graph snapshot/recipe/selection CLI, graph IR, node fingerprint,
ABI hash, compatibility fallback, or runtime graph patch in the current API.

## Bind the runtime function

Load the external function before the bundle Task runs:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --byok-library ./residual_add.so \
  --byok-function run \
  --byok-name my_family.residual_add \
  --prompt "Hello"
```

All three BYOK options are required together. The runtime loads
`libtrtmc_byok_tvm_ffi.so` from the runtime root, then binds the explicit
external DSO/function/name. The serialized engine contains the plugin layer but
not the external kernel DSO.

TVM-FFI does not let Model Connect verify the external function signature or
library contents. The kernel author must implement the exact ordered tensor,
workspace, and extra-argument contract used at build time.

For complete examples, see `examples/byok/` and
[Bring Your Own Kernel](../tutorials/advanced/bring-your-own-kernel.md).
