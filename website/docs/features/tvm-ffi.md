---
title: TVM FFI Kernel Bridge
---

Model Connect can replace one selected TensorRT graph region with a TVM-FFI
kernel. The TensorRT engine fixes the region boundary; the external kernel DSO
is chosen when a new pipeline loads.

There are two ways to select the region:

| Path | Who selects the TensorRT nodes | When to use it |
| --- | --- | --- |
| Family Recipe | The model family records an exact, versioned region and instance while building the graph. | Use this first when a matching Recipe exists. |
| Manual selection | The user inspects the raw graph and supplies exact node IDs. | Use this when no Recipe matches the required boundary. |

Both paths call the same region validator, graph replacement, TensorRT plugin,
and load-time binding implementation. A Recipe is only a shortcut for a known
manual selection.

## Path 1: use a family Recipe

List the Recipes recorded for a build:

```bash
trtmc graph inspect \
  --snapshot graph.json \
  --engine-role decode \
  MODEL [build options...]

trtmc graph recipes graph.json
```

Build one exact Recipe instance:

```bash
trtmc build MODEL [the same build options...] \
  --recipe RECIPE_ID INSTANCE_ID \
  -o model-slot.trtfb
```

The command captures the graph, resolves the Recipe, validates its nodes, and
applies the ordinary graph patch. It also writes
`model-slot.selection.json`, which is the boundary and ABI receipt.

## Path 2: select a region manually

Capture and inspect the same graph:

```bash
trtmc graph inspect \
  --snapshot graph.json \
  --engine-role decode \
  MODEL [build options...]

trtmc graph list graph.json --match '*attention*'
```

Select exact node IDs copied from that snapshot:

```bash
trtmc graph select graph.json \
  --nodes node:120 node:121 node:122 \
  --binding-id my.attention@1 \
  --workspace-bytes 0 \
  --output-shape-like-input 0 \
  -o attention.selection.json
```

`--output-shape-like-input` is required only for a dynamic output. Fixed scalar
or null arguments can be appended in call order with repeatable
`--extra-arg JSON`.

Build the selected replacement:

```bash
trtmc build MODEL [the same build options...] \
  --graph-patch attention.selection.json \
  -o model-slot.trtfb
```

Selections are tied to the captured graph fingerprint and engine role. Reuse
the same model revision and graph-producing build options, and recapture after
either changes.

## Bind a kernel when the pipeline loads

A slot-ready bundle contains `kernel_slots.json` with one binding ID and its
ABI SHA-256. It does not contain the external kernel DSO.

Create a strict JSON binding manifest beside the DSO:

```json
{
  "schema_version": 1,
  "bindings": [
    {
      "id": "my.attention@1",
      "abi_sha256": "<abi_sha256 from attention.selection.json>",
      "library": "./attention-kernel.so",
      "function": "run"
    }
  ]
}
```

The library path must be relative to the manifest. Load and run:

```bash
trtmc run model-slot.trtfb \
  --kernel-bindings kernel-bindings.json \
  --prompt "Hello" \
  --max-new-tokens 32
```

The manifest carries the ABI hash only. That hash describes the selected call
contract; it is not a DSO or library-content hash, and the runtime does not pin
the DSO bytes. To use another implementation, point a new manifest at another
DSO with the same binding ID and ABI, then construct a new pipeline from the
same bundle. A running pipeline is not rebound in place.

This flow uses JSON selections and bindings. There is no kernel YAML
configuration.

## Kernel ABI

The selection receipt records the ordered tensor boundary, workspace size,
fixed arguments, output-shape rule, and ABI SHA-256. The hash covers:

- input and output tensor dtypes and declared shapes;
- workspace size;
- fixed scalar or null arguments;
- the dynamic-output shape rule.

The exported TVM-FFI function receives arguments in this order:

1. boundary input DLTensors;
2. one CUDA `uint8` workspace DLTensor when workspace is nonzero;
3. boundary output DLTensors;
4. fixed arguments from the selection.

Matching the ABI hash confirms that the manifest targets the engine's recorded
contract. TVM FFI does not provide a signature that Model Connect can compare
with the DSO, so the kernel author must implement the ordered contract exactly.

## Current limits

- Only native TensorRT bundles built with TVM-FFI support are accepted.
  TensorRT-RTX and optimized-runtime bundles are rejected.
- Graph-slot builds currently require tensor parallel size 1 and reject
  quantized or FP8 builds.
- V1 replaces one connected, convex region with exactly one output in one
  engine role.
- A selected region cannot include a network output, shape or host tensors,
  control-flow or collective layers, or an existing plugin layer.
- Boundary tensors must be linear, rank 8 or lower, and use BF16, FP16, FP32,
  or INT32.
- The selected engine must deserialize while the pipeline loads so the runtime
  can capture the bound function.

For a complete worked example, see
[Bring Your Own Kernel with TVM FFI](../tutorials/advanced/bring-your-own-kernel.md).
