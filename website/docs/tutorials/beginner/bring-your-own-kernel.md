---
title: "Bring Your Own Kernel"
---

This tutorial replaces one explicitly selected region of a native TensorRT
graph with a load-time TVM-FFI kernel slot.

The workflow is:

```text
graph inspect -> graph list -> graph select -> build --graph-patch
              -> kernel-bindings.json -> load a pipeline
```

There is no semantic lowering map. You inspect the TensorRT graph that Model
Connect is about to compile, copy the exact node IDs that you want to replace,
and bring a compatible TVM-FFI DSO. The model family does not need a new slot
or a code change.

This is a raw TensorRT graph workflow. Layer names and operations help you
navigate, but they are not a stable model-level semantic API.

## Before you start

You need:

- a native TensorRT build of Model Connect with TVM-FFI enabled;
- a trusted TVM-FFI DSO for the target GPU;
- enough resources to build the selected model;
- one pinned model revision and one fixed set of build options.

Run the commands from the repository root. This example uses Qwen3-8B only to
make the commands concrete; it does not assume a particular attention
implementation or promise a speedup.

```bash
export MODEL=Qwen/Qwen3-8B
export REVISION=b968826d9c46dd6066d109eabc6255188de91218
export WORK="$PWD/artifacts/qwen3-graph-slot"
mkdir -p "$WORK"

BUILD_ARGS=(
  "$MODEL"
  --model-revision "$REVISION"
  --precision bf16
  --max-cache-length 40960
  --decoder-engine-layout split
)
```

Use the same `BUILD_ARGS` for inspection, the patched build, and the native
comparison build. A different revision, precision, cache length, layout, or
graph-producing implementation invalidates the selection.

## 1. Capture the TensorRT graph

Capture the decode graph immediately before TensorRT serialization:

```bash
trtmc graph inspect \
  --engine-role decode \
  --snapshot "$WORK/decode.graph.json" \
  "${BUILD_ARGS[@]}"
```

Inspection stops before engine compilation. The snapshot contains the ordered
TensorRT layers, tensors, engine role, build metadata, and a graph fingerprint.
For a split decoder, use `--engine-role prefill` instead to inspect the prefill
engine. `dual_profile` is also available for a dual-profile decoder build.

## 2. List nodes and choose a region

Filter the display by node ID, operation, or layer name:

```bash
trtmc graph list "$WORK/decode.graph.json" \
  --match '*attention*' | tee "$WORK/decode.nodes.txt"
```

The columns are:

```text
ID        OP        NAME        INPUTS        OUTPUTS
node:...  ...       ...         tensor:...    tensor:...
```

Use `OP`, `NAME`, and tensor edges to find the region, then copy its node IDs.
Start with the smallest region that matches your kernel. The selected nodes
must form one connected, convex region: a graph path cannot leave the region
and later re-enter it.

TensorRT 11 displays one `IAttention` as adjacent `ATTENTION_INPUT` and
`ATTENTION_OUTPUT` nodes. Select both when replacing the complete attention
operation. Model Connect includes typed attention ports such as
`key_value_lengths` in the ordered boundary even though TensorRT does not
expose those ports through ordinary `get_input()` calls.

`--match` only filters what `list` displays; it does not select anything. Node
IDs are explicit on purpose. `graph select` does not accept a model semantic
name, wildcard, regular expression, or lowering map.

## 3. Lock the selection and ABI

The following IDs are examples. Replace them with IDs copied from your own
`graph list` output:

```bash
export BINDING_ID=my.decode_attention@1
NODES=(node:120 node:121)

trtmc graph select "$WORK/decode.graph.json" \
  --nodes "${NODES[@]}" \
  --binding-id "$BINDING_ID" \
  --workspace-bytes 0 \
  --output-shape-like-input 0 \
  -o "$WORK/decode-attention.selection.json"
```

`graph select` prints each ordered input and output tensor ID, TensorRT name,
dtype, and shape, followed by `abi_sha256`. It also writes a selection JSON
containing:

- the graph fingerprint and engine role;
- `abi_sha256`, derived from the ordered tensor contracts, workspace, and
  extra arguments;
- the exact selected node and boundary tensor IDs.

The Qwen decode attention output has the same dynamic dimensions as boundary
`input[0]`, so the example names that relationship explicitly. For a different
region, choose the matching index from the snapshot; the input and output dtype
and declared shape must match. Remove the option for a fixed-shape output.
Model Connect never guesses a relationship from two matching `-1`
placeholders.

If the kernel needs fixed scalar or null arguments, add one strict JSON object
per argument, in call order:

```bash
trtmc graph select "$WORK/decode.graph.json" \
  --nodes "${NODES[@]}" \
  --binding-id "$BINDING_ID" \
  --workspace-bytes 0 \
  --output-shape-like-input 0 \
  --extra-arg '{"type":"int","value":32}' \
  --extra-arg '{"type":"float","value":0.5}' \
  -o "$WORK/decode-attention.selection.json"
```

Allowed types are `none`, signed 32-bit `int`, finite `float`, and null `ptr`.
`none` and `ptr` have no `value` field.

## 4. Implement the TVM-FFI function

Use the boundary records printed by `graph select` to export a TVM-FFI module
function, such as `run`, with this fixed call order:

1. boundary input DLTensors in `input_tensor_ids` order;
2. when `workspace_bytes` is nonzero, one CUDA `uint8` workspace DLTensor;
3. boundary output DLTensors in `output_tensor_ids` order;
4. extra arguments in their declared order.

The function must write the declared outputs. A raw third-party kernel is not
automatically compatible; wrap or export it through TVM-FFI with this exact
contract. Changing a boundary dtype, shape, argument order, workspace size, or
extra argument changes the engine ABI and requires a new selection and build.

## 5. Build a slot-ready bundle

Build with the same model revision and options used for inspection:

```bash
trtmc build "${BUILD_ARGS[@]}" \
  --graph-patch "$WORK/decode-attention.selection.json" \
  -o "$WORK/qwen3-slot-ready.trtfb"
```

Build reconstructs the graph, verifies its fingerprint, rewires the selected
region to a `TvmFfiKernel` layer, and stores `kernel_slots.json` in the bundle.
That section contains the binding ID and ABI SHA-256. The plugin name is
derived as `trtmc.slot.<binding-id>` instead of being repeated in the schema.

The DSO is not embedded in this bundle. The TensorRT engine and its slot ABI
are now fixed; the compatible DSO is chosen when a pipeline is loaded.

## 6. Write the strict runtime binding manifest

Copy your trusted DSO beside the manifest under an immutable, content-addressed
name:

```bash
export SOURCE_DSO=/path/to/your/kernel.so
export DSO_SHA256="$(sha256sum "$SOURCE_DSO" | awk '{print $1}')"
export DSO="$WORK/kernel.$DSO_SHA256.so"
cp "$SOURCE_DSO" "$DSO"
export SELECTION="$WORK/decode-attention.selection.json"

export ABI_SHA256="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["abi_sha256"])' \
    "$SELECTION"
)"

cat > "$WORK/kernel-bindings.json" <<EOF
{
  "schema_version": 1,
  "bindings": [
    {
      "id": "$BINDING_ID",
      "abi_sha256": "$ABI_SHA256",
      "library": "./kernel.$DSO_SHA256.so",
      "sha256": "$DSO_SHA256",
      "function": "run"
    }
  ]
}
EOF
```

The schema is exact: missing or unknown fields fail. The library path must be
relative to the manifest, each DSO digest must match, and every slot in the
bundle must be bound exactly once. A digest verifies identity, not safety; load
only trusted native code.

## 7. Load and run

A slot-ready bundle requires `--kernel-bindings`:

```bash
trtmc run "$WORK/qwen3-slot-ready.trtfb" \
  --kernel-bindings "$WORK/kernel-bindings.json" \
  --prompt "Explain grouped-query attention in one sentence." \
  --max-new-tokens 32 \
  --greedy
```

The runtime checks that the manifest-declared ABI hash matches the bundle, then
validates the DSO hash and loads the named module function while it constructs
the pipeline. TVM-FFI does not expose a function signature for the runtime to
compare with the tensor contract, so the kernel author must implement the ABI
printed by `graph select`. A slot-ready load fails if that model defers
TensorRT engine deserialization until first inference; v1 binds only during
pipeline loading.

You can load the same bundle into a new pipeline with another DSO when that DSO
uses the same binding ID and ABI hash. Existing pipelines keep the function
they loaded: editing the manifest or loading another pipeline does not rebind a
running pipeline. Use a distinct immutable filename for each DSO version;
overwriting a loaded `.so` path is not a supported swap mechanism. Destroy and
load a new pipeline to switch kernels.

## 8. Check correctness and performance

Build a native baseline with the same `BUILD_ARGS`, compare deterministic
outputs, and benchmark both bundles on the same idle GPU:

```bash
trtmc build "${BUILD_ARGS[@]}" -o "$WORK/qwen3-native.trtfb"

trtmc run "$WORK/qwen3-native.trtfb" \
  --prompt "Explain grouped-query attention in one sentence." \
  --max-new-tokens 32 --greedy > "$WORK/native.txt"

trtmc run "$WORK/qwen3-slot-ready.trtfb" \
  --kernel-bindings "$WORK/kernel-bindings.json" \
  --prompt "Explain grouped-query attention in one sentence." \
  --max-new-tokens 32 --greedy > "$WORK/external.txt"

diff -u "$WORK/native.txt" "$WORK/external.txt"
```

An empty diff is a useful smoke test, not a complete numerical qualification.
Use the same prompt and idle GPU for a simple no-regression gate:

```bash
PERF_PROMPT='Explain grouped-query attention, KV caching, and decode latency.'

trtmc run "$WORK/qwen3-native.trtfb" \
  --prompt "$PERF_PROMPT" --max-new-tokens 64 --greedy \
  --warmup 3 --benchmark 10 > "$WORK/native.perf.log" 2>&1

trtmc run "$WORK/qwen3-slot-ready.trtfb" \
  --kernel-bindings "$WORK/kernel-bindings.json" \
  --prompt "$PERF_PROMPT" --max-new-tokens 64 --greedy \
  --warmup 3 --benchmark 10 > "$WORK/external.perf.log" 2>&1

python - "$WORK/native.perf.log" "$WORK/external.perf.log" <<'PY'
import re
import sys

def throughput(path):
    text = open(path, encoding="utf-8").read()
    values = [float(x) for x in re.findall(r"tokens_per_sec=([0-9.]+)", text)]
    if not values:
        raise SystemExit(f"{path}: tokens_per_sec was not reported")
    return values[-1]

native = throughput(sys.argv[1])
external = throughput(sys.argv[2])
print(f"native={native:.2f} external={external:.2f} tokens/s")
if external < native:
    raise SystemExit("FAIL: the external kernel is slower")
print("PASS: the external kernel is not slower")
PY
```

Accept the kernel only if it meets your numerical tolerance and this gate.
This tutorial makes no performance claim for a particular external DSO.

## Current limits

- Only native TensorRT bundles with TVM-FFI enabled are supported. Explicit
  graph slots currently reject TensorRT-RTX, optimized-runtime, quantized, and
  tensor-parallel builds.
- One build can replace one connected, convex region with exactly one output
  boundary in one engine role.
- A region cannot include a network output, shape or host tensor, control-flow
  or collective layer, or an existing plugin layer.
- A selected output must have fixed positive dimensions or the same dtype and
  shape as a boundary input.
- Boundary tensors must be linear, rank 8 or lower, and use BF16, FP16, FP32,
  or INT32.
- The selection is tied to the exact captured graph and build settings.
- Kernel binding happens only while a new pipeline is loaded; there is no
  in-place rebind API for a running pipeline.
- v1 rejects a model that defers its selected engine's TensorRT deserialization
  until inference instead of completing it during pipeline load.
- Slot-ready bundles load through the CLI or the C++ binding overload; the
  current C-linkage API has no kernel-binding argument.

## Existing family-owned Direct Slots

The earlier Direct Slot workflow remains supported. When a model family
already publishes the boundary you need, use `trtmc kernel slots` and
`trtmc build --kernel <manifest.yaml>`. That YAML flow is simpler and keeps the
semantic call site family-owned. Use explicit graph selection only when you
need a raw TensorRT region that the family does not publish.
