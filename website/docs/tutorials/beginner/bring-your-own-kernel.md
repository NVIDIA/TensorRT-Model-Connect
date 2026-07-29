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

Run the commands from the repository root. This example uses Qwen3-8B and the
small identity-copy TVM-FFI kernel shipped in `examples/byok/`. It first
replaces a Qwen `logits + zero bias` region. That deliberately simple region
makes the complete mechanism testable before you adapt the same steps to a real
attention kernel. It is a load-and-correctness example, not a promised speedup.

```bash
export MODEL=Qwen/Qwen3-8B
export REVISION=b968826d9c46dd6066d109eabc6255188de91218
export WORK="$PWD/artifacts/qwen3-graph-slot"
export BUILD_DIR="$PWD/build-runtime-trt"
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

The POC region is at the end of the decode graph, so list its last layers:

```bash
trtmc graph list "$WORK/decode.graph.json" \
  | tee "$WORK/decode.nodes.txt" \
  | tail -n 10
```

The columns are:

```text
ID        OP        NAME        INPUTS        OUTPUTS
node:...  ...       ...         tensor:...    tensor:...
```

For the pinned revision above, the tail includes this chain:

```text
node:3598  CONSTANT
node:3599  CAST
node:3600  ELEMENTWISE
node:3601  CAST          -> logits
```

Qwen has no LM-head bias in this checkpoint, so Model Connect emits a zero
constant, casts it to BF16, and adds it to the BF16 matrix-multiply result.
Select the first three nodes and leave the final FP32 logits cast in the graph.
If your IDs differ, use the chain shown by your own snapshot instead of copying
these numbers.

For another kernel, use `--match GLOB` to filter by node ID, operation, or layer
name, then follow the tensor edges and copy the exact IDs. Start with the
smallest region that matches the kernel. The selected nodes must form one
connected, convex region: a graph path cannot leave the region and later
re-enter it.

`--match` only filters what `list` displays; it does not select anything. Node
IDs are explicit on purpose. `graph select` does not accept a model semantic
name, wildcard, regular expression, or lowering map.

## 3. Lock the selection and ABI

Use the IDs verified in the previous step:

```bash
export BINDING_ID=qwen3.decode.logits_copy@1
NODES=(node:3598 node:3599 node:3600)

trtmc graph select "$WORK/decode.graph.json" \
  --nodes "${NODES[@]}" \
  --binding-id "$BINDING_ID" \
  --workspace-bytes 0 \
  -o "$WORK/logits-copy.selection.json"
```

`graph select` prints each ordered input and output tensor ID, TensorRT name,
dtype, and shape, followed by `abi_sha256`. It also writes a selection JSON
containing:

- the graph fingerprint and engine role;
- `abi_sha256`, derived from the ordered tensor contracts, workspace, and
  extra arguments;
- the exact selected node and boundary tensor IDs.

This POC prints one BF16 `[1, 151936]` input and one BF16 `[1, 151936]`
output. Both shapes are fixed, so it does not use
`--output-shape-like-input`. A dynamic output requires that option with a
boundary input of the same dtype and declared shape. A `-1` means your DSO must
handle every runtime shape allowed by the built engine profile; Model Connect
never guesses a relationship from matching `-1` placeholders.

If the kernel needs fixed scalar or null arguments, add one strict JSON object
per argument, in call order. Add lines like these immediately before `-o` in
the `graph select` command that already succeeded above; keep or omit
`--output-shape-like-input` according to that region's output:

```text
Before:
  -o "$WORK/logits-copy.selection.json"

After:
  --extra-arg '{"type":"int","value":32}' \
  --extra-arg '{"type":"float","value":0.5}' \
  -o "$WORK/logits-copy.selection.json"
```

Allowed types are `none`, signed 32-bit `int`, finite `float`, and null `ptr`.
`none` and `ptr` have no `value` field.

## 4. Implement the TVM-FFI function

Build the supplied POC DSO from the same configured native build tree:

```bash
cmake --build "$BUILD_DIR" --target trtmc_byok_identity_copy -j
export SOURCE_DSO="$BUILD_DIR/identity_copy_kernel.so"
test -f "$SOURCE_DSO"
```

Its exported function is equivalent to:

```cpp
run(TensorView input, TensorView output)
```

It performs an asynchronous device-to-device copy on TensorRT's current CUDA
stream. That is equivalent to the selected Qwen `logits + zero` region.

For your own region, use the boundary records printed by `graph select` to
export a TVM-FFI module function, such as `run`, with this fixed call order:

1. boundary input DLTensors in `input_tensor_ids` order;
2. when `workspace_bytes` is nonzero, one CUDA `uint8` workspace DLTensor;
3. boundary output DLTensors in `output_tensor_ids` order;
4. extra arguments in their declared order.

The function must write the declared outputs. A raw third-party kernel is not
automatically compatible; wrap or export it through TVM-FFI with this exact
contract. Changing a boundary dtype, shape, argument order, workspace size, or
extra argument changes the engine ABI and requires a new selection and build.
The identity-copy DSO is only compatible with a one-input, one-output
same-size contiguous boundary; it is not an attention implementation.

## 5. Build a slot-ready bundle

Build with the same model revision and options used for inspection:

```bash
trtmc build "${BUILD_ARGS[@]}" \
  --graph-patch "$WORK/logits-copy.selection.json" \
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
export DSO_SHA256="$(sha256sum "$SOURCE_DSO" | awk '{print $1}')"
export DSO="$WORK/kernel.$DSO_SHA256.so"
cp "$SOURCE_DSO" "$DSO"
export SELECTION="$WORK/logits-copy.selection.json"

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

TEST_PROMPT='Explain grouped-query attention, KV caching, and decode latency.'
MAX_NEW_TOKENS=64

trtmc run "$WORK/qwen3-native.trtfb" \
  --prompt "$TEST_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --output "$WORK/native.jsonl"

trtmc run "$WORK/qwen3-slot-ready.trtfb" \
  --kernel-bindings "$WORK/kernel-bindings.json" \
  --prompt "$TEST_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --output "$WORK/external.jsonl"

diff -u "$WORK/native.jsonl" "$WORK/external.jsonl"
```

The JSONL includes both decoded text and exact token IDs. An empty diff is a
useful smoke test, not a complete numerical qualification. Use the same prompt
and idle GPU for a simple no-regression gate:

```bash
trtmc run "$WORK/qwen3-native.trtfb" \
  --prompt "$TEST_PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --warmup 3 --benchmark 10 > "$WORK/native.perf.log" 2>&1

trtmc run "$WORK/qwen3-slot-ready.trtfb" \
  --kernel-bindings "$WORK/kernel-bindings.json" \
  --prompt "$TEST_PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --warmup 3 --benchmark 10 > "$WORK/external.perf.log" 2>&1

python - "$WORK/native.perf.log" "$WORK/external.perf.log" <<'PY'
import re
import sys

def result(path):
    text = open(path, encoding="utf-8").read()
    matches = re.findall(
        r"generated_tokens_mean=([0-9.]+).*tokens_per_sec=([0-9.]+)", text
    )
    if not matches:
        raise SystemExit(f"{path}: benchmark token count and throughput were not reported")
    generated_tokens, throughput = matches[-1]
    return float(generated_tokens), float(throughput)

native_tokens, native_throughput = result(sys.argv[1])
external_tokens, external_throughput = result(sys.argv[2])
print(
    f"native={native_throughput:.2f} external={external_throughput:.2f} tokens/s; "
    f"mean generated tokens={native_tokens:.2f}"
)
if external_tokens != native_tokens:
    raise SystemExit(
        "FAIL: generated-token means differ "
        f"(native={native_tokens:.2f}, external={external_tokens:.2f})"
    )
if native_throughput <= 0 or external_throughput <= 0:
    raise SystemExit("FAIL: throughput must be positive")
ratio = external_throughput / native_throughput
if ratio < 0.98:
    raise SystemExit(f"FAIL: external/native throughput is {ratio:.4f}, below 0.98")
print(f"PASS: external/native throughput is {ratio:.4f}")
PY
```

The `0.98` threshold allows a 2% shortfall for normal measurement noise; it is
not a license for a known regression. Re-run on an idle GPU if the result is
close to the boundary. Accept the kernel only if it meets your numerical
tolerance and this gate. This tutorial makes no performance claim for a
particular external DSO.

## 9. Replace attention with your kernel

After the POC passes, repeat the same workflow for attention:

```bash
trtmc graph list "$WORK/decode.graph.json" --match '*attention*'
```

TensorRT 11 displays one `IAttention` as adjacent `ATTENTION_INPUT` and
`ATTENTION_OUTPUT` nodes. Select both for the complete attention operation, and
include surrounding scale or layout nodes only when that matches your kernel's
contract. Model Connect includes typed attention ports such as
`key_value_lengths` in the ordered boundary even though TensorRT does not
expose those ports through ordinary `get_input()` calls.

Run `graph select` with those IDs, implement the exact printed TVM-FFI argument
order, and rebuild the slot-ready bundle. A FlashInfer or CUDA-DSL export with a
different rank, page-table argument, scale argument, or output layout needs a
small TVM-FFI wrapper; a DSO built for a family-owned Direct Slot is not
automatically compatible with a raw graph region. Keep the new kernel only
after the same correctness and no-regression checks pass.

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
- The selection is tied to the captured graph fingerprint and recorded build
  metadata. Reuse identical `BUILD_ARGS`.
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
