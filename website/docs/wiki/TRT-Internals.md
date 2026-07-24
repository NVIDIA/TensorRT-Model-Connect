# TensorRT Internals

## Build side

The Python phase:

1. resolves the checkpoint and family descriptor;
2. loads family-specific config and weights;
3. creates an explicit-batch TensorRT network;
4. asks the family builder to add inputs, layers, outputs, and optimization
   profiles;
5. compiles one or more serialized engine plans; and
6. writes engine, config, tokenizer/metadata, and optional auxiliary sections
   to a `.trtfb` bundle.

Graph semantics are family-owned below
`python/tensorrt_model_connect/families/<family>/`. There is no supported
root-level shared `graph_ops.py` contract.

## Runtime side

The C++ phase:

1. reads the bundle and resolves `runtime_strategy`;
2. loads the owning model DSO;
3. creates the model plugin/pipeline;
4. deserializes the required TensorRT engines;
5. creates execution contexts and model-owned state;
6. binds tensors for the requested operation; and
7. executes the family pipeline on CUDA.

Common TensorRT wrappers and device primitives are in `src/runtime/core/` and
public runtime headers. KV caches, samplers, schedulers, recurrent state,
pre/postprocessing, and tensor-name contracts stay model-owned when their
behavior is model-specific.

## Profiles and shapes

Build-time optimization profiles constrain legal runtime shapes. Decoder
families may use split prefill/decode engines or a dual-profile engine when
their plugin supports it. Diffusion and other modalities may use different
component engines and batch policies. Inspect the bundle and owning descriptor
instead of assuming one universal engine layout.

```bash
./build/trtmc inspect /path/to/model.trtfb
```

This requires a built CLI. Bundle creation and inference additionally require
the appropriate TensorRT/CUDA environment.

## Precision and parity

Precision is selected by the CLI and family policy. FP16/BF16/quantized builds
can change numerics, kernel selection, and memory. Validate the selected
family's output contract against its reference before interpreting performance
results.
