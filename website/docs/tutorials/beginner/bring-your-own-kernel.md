---
title: "Bring Your Own Kernel"
---

This tutorial replaces part of a Qwen3-8B TensorRT graph with a TVM-FFI
kernel. You do not write TensorRT C++.

Start with the simplest graph workflow and move to the escape hatch only when
you need it:

| Level | How the region is chosen | Who should use it |
| --- | --- | --- |
| Recommended | The model family publishes a versioned recipe and you choose one exact instance. | Most kernel authors. |
| Advanced | You inspect the raw TensorRT graph and type every node ID yourself. | Authors whose region has no recipe. |

A recipe is only a family-owned shortcut for a known manual selection. It
records exact TRT node IDs, workspace, scalar arguments, and output-shape rule
while the family constructs the graph. `build --recipe` automatically captures
the graph, passes those values to the same `select_region()` validator used by
the advanced path, and then runs the ordinary graph-patch build. It adds no
semantic graph, matching language, plugin schema, or runtime behavior.

The two levels use the same backend:

```text
Recipe: build --recipe -----------------------> slot-ready bundle + selection receipt
Manual: graph inspect -> select -> graph-patch -> slot-ready bundle + selection receipt
                                                  |
                                                  v
                                      kernel-bindings.json -> new pipeline
```

The DSO is not stored in a graph-slot bundle. You may bind another
ABI-compatible DSO when constructing another pipeline without rebuilding the
bundle. A running pipeline is never rebound in place.

## Before you start

Complete [Installation](../../getting-started/installation.md), clone this
repository, and run the commands below from its root. You need:

- a native TensorRT build with TVM-FFI enabled;
- enough memory and disk to build Qwen3-8B;
- a trusted TVM-FFI DSO for the region you select;
- one pinned model revision and one fixed set of build options.

This first pass uses the identity-copy kernel in `examples/byok/` and Qwen's
prebuilt `logits + zero` recipe. The operation is intentionally simple: it
lets a first-time user test recipe selection, graph replacement, load-time
binding, correctness, and performance before adapting a real kernel.

This tutorial builds its example DSO from the source tree, so use the native
CLI from that same configured build.

```bash
export MODEL=Qwen/Qwen3-8B
export REVISION=b968826d9c46dd6066d109eabc6255188de91218
export WORK="$PWD/artifacts/qwen3-graph-slot"
export BUILD_DIR="${BUILD_DIR:-$PWD/build}"
export TRTMC="${TRTMC:-$BUILD_DIR/trtmc}"
mkdir -p "$WORK"

test -f "$BUILD_DIR/CMakeCache.txt" || {
  echo "BUILD_DIR must name an already configured native TVM-FFI build"
  exit 1
}
test -x "$TRTMC" || {
  echo "TRTMC must name the native CLI from BUILD_DIR"
  exit 1
}

BUILD_ARGS=(
  "$MODEL"
  --model-revision "$REVISION"
  --precision bf16
  --max-cache-length 40960
  --decoder-engine-layout split
)
```

Use the identical `BUILD_ARGS` for the Recipe build, any manual graph
inspection, and the native comparison. Changing the revision, precision, cache
length, engine layout, or graph-producing code invalidates a saved selection.

:::warning Trusted native code only
A DSO executes native code when loaded. Use only a library that you trust.
:::

## Level 1: use a family recipe

### 1. Build a slot-ready bundle in one command

This tutorial already tells you the versioned Recipe ID and its exact instance.
Build the slot-ready bundle directly:

```bash
export BINDING_ID=qwen.decode_logits_copy@1

"$TRTMC" build "${BUILD_ARGS[@]}" \
  --recipe "$BINDING_ID" decoder.logits_zero_bias \
  -o "$WORK/qwen3-slot-ready.trtfb"
```

The command internally performs the existing capture, Recipe resolution,
`select_region()`, and `--graph-patch` steps. It writes two files:

```text
qwen3-slot-ready.trtfb
qwen3-slot-ready.selection.json
```

The selection receipt contains the exact boundary tensor IDs, graph
fingerprint, binding ID, ABI SHA-256, workspace, scalar arguments, and
output-shape rule. Keep it next to your kernel integration records.

The Recipe does not bypass validation. The selected region must still be
connected and convex, expose exactly one output, use supported device tensors,
and satisfy the same output-shape rules as a manual selection.

If a family tutorial does not give you the Recipe and instance names, inspect
them without compiling an engine:

```bash
"$TRTMC" graph inspect \
  --engine-role decode \
  --snapshot "$WORK/decode.graph.json" \
  "${BUILD_ARGS[@]}"

"$TRTMC" graph recipes "$WORK/decode.graph.json" \
  | tee "$WORK/decode.recipes.txt"
```

For the pinned Qwen3-8B revision, the output includes:

```text
RECIPE                               INSTANCE                           NODES
qwen.decode_attention_region@1       decoder.layers.0.decode_attention  node:79,...,node:84
...
qwen.decode_attention_region@1       decoder.layers.35.decode_attention node:3544,...,node:3549
qwen.decode_logits_copy@1            decoder.logits_zero_bias           node:3598,node:3599,node:3600
```

Recipe IDs are versioned. Instances are explicit because a model may contain
many copies of the same pattern. The command never silently chooses the first
match or every match.

### 2. Build the example TVM-FFI DSO

Build the supplied DSO from the configured native tree:

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
stream, which is equivalent to adding Qwen's zero LM-head bias.

For another Recipe, implement the exact call order printed by `build --recipe`
and stored in the selection receipt:

1. boundary input DLTensors in `input_tensor_ids` order;
2. a CUDA `uint8` workspace DLTensor when `workspace_bytes` is nonzero;
3. boundary output DLTensors in `output_tensor_ids` order;
4. fixed scalar or null arguments in `extra_args` order.

TVM-FFI does not expose a signature that Model Connect can compare with this
contract at load time. The kernel author must implement that ABI
exactly.

### 3. Bind the DSO when a pipeline loads

Copy the DSO next to the runtime manifest:

```bash
export DSO="$WORK/kernel.so"
cp "$SOURCE_DSO" "$DSO"

export ABI_SHA256="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["abi_sha256"])' \
    "$WORK/qwen3-slot-ready.selection.json"
)"

cat > "$WORK/kernel-bindings.json" <<EOF
{
  "schema_version": 1,
  "bindings": [
    {
      "id": "$BINDING_ID",
      "abi_sha256": "$ABI_SHA256",
      "library": "./kernel.so",
      "function": "run"
    }
  ]
}
EOF
```

Missing or unknown fields fail. The library path is relative to the manifest,
and every slot in the bundle must be bound exactly once.

Load and run:

```bash
"$TRTMC" run "$WORK/qwen3-slot-ready.trtfb" \
  --kernel-bindings "$WORK/kernel-bindings.json" \
  --prompt "Explain grouped-query attention in one sentence." \
  --max-new-tokens 32 \
  --greedy
```

To switch kernels, create another manifest naming a different DSO with the same
binding ID and ABI hash, destroy the old pipeline, and construct a new one.
Editing a manifest does not affect an existing pipeline.

### 4. Check correctness and no regression

Build the native comparison with the same arguments:

```bash
"$TRTMC" build "${BUILD_ARGS[@]}" -o "$WORK/qwen3-native.trtfb"
```

Compare deterministic output:

```bash
TEST_PROMPT='Explain grouped-query attention, KV caching, and decode latency.'
MAX_NEW_TOKENS=64

"$TRTMC" run "$WORK/qwen3-native.trtfb" \
  --prompt "$TEST_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --output "$WORK/native.jsonl"

"$TRTMC" run "$WORK/qwen3-slot-ready.trtfb" \
  --kernel-bindings "$WORK/kernel-bindings.json" \
  --prompt "$TEST_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --output "$WORK/external.jsonl"

diff -u "$WORK/native.jsonl" "$WORK/external.jsonl"
```

An empty diff proves exact token equality for this smoke input. It is not a
complete numerical qualification.

Benchmark both bundles on the same idle GPU:

```bash
"$TRTMC" run "$WORK/qwen3-native.trtfb" \
  --prompt "$TEST_PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --warmup 3 --benchmark 10 > "$WORK/native.perf.log" 2>&1

"$TRTMC" run "$WORK/qwen3-slot-ready.trtfb" \
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
        raise SystemExit(f"{path}: benchmark result was not reported")
    generated_tokens, throughput = matches[-1]
    return float(generated_tokens), float(throughput)

native_tokens, native_throughput = result(sys.argv[1])
external_tokens, external_throughput = result(sys.argv[2])
print(f"native={native_throughput:.2f} external={external_throughput:.2f} tokens/s")
if external_tokens != native_tokens:
    raise SystemExit("FAIL: generated-token means differ")
if native_throughput <= 0 or external_throughput <= 0:
    raise SystemExit("FAIL: throughput must be positive")
ratio = external_throughput / native_throughput
if ratio < 0.98:
    raise SystemExit(f"FAIL: external/native throughput is {ratio:.4f}, below 0.98")
print(f"PASS: external/native throughput is {ratio:.4f}")
PY
```

The 2% margin covers ordinary measurement noise; it is not permission to
accept a known regression. Repeat an edge result on an idle GPU.

### 5. Apply the same Recipe flow to attention

Qwen also records one raw decode-attention recipe per layer:

```bash
"$TRTMC" build "${BUILD_ARGS[@]}" \
  --recipe qwen.decode_attention_region@1 \
           decoder.layers.0.decode_attention \
  -o "$WORK/qwen3-attention-slot.trtfb"
```

For Qwen3-8B layer 0, the printed boundary plus `extra_args` in the selection
receipt `qwen3-attention-slot.selection.json` define this ordered contract:

```text
input[0]  query              BF16  [1, 32, -1, 128]
input[1]  key cache          BF16  [1, 8, 40960, 128]
input[2]  value cache        BF16  [1, 8, 40960, 128]
input[3]  key/value lengths  INT32 [1]
output[0] context            BF16  [1, 32, -1, 128]
extra[0]  softmax scale      float
```

Export any CUDA DSL, FlashInfer, or custom CUDA kernel through TVM-FFI with
that exact ABI and then reuse the slot-ready build and load-time binding steps.
No Model Connect change is required for another DSO with this contract.

The repository does not yet ship a verified FlashInfer wrapper for this raw
boundary. The FlashInfer exporter already shipped under
`python/tensorrt_model_connect/families/qwen/examples/byok_flashinfer/` targets
the older, richer `qwen.decode_attention@1` Direct Slot. It uses 2-D
query/output tensors plus page offsets and a page table created by Qwen family
glue. Those tensors do not exist at the raw TRT attention boundary, so that DSO
is **not** compatible with `qwen.decode_attention_region@1`.

#### Working FlashInfer reference: Direct Slot

If you want to run that shipped FlashInfer kernel on an SM 10.3 GPU, use its
published Direct Slot flow:

```bash
"$TRTMC" kernel slots "$MODEL" --model-revision "$REVISION"

python -m pip install \
  "flashinfer-python==0.6.15" \
  "nvidia-cutlass-dsl==4.5.0" \
  "apache-tvm-ffi==0.1.12"

export FI_WORK="$PWD/artifacts/qwen3-flashinfer"
mkdir -p "$FI_WORK"
test ! -e "$FI_WORK/flashinfer-qwen3.so" || {
  echo "Choose a fresh FI_WORK; the exporter will not overwrite a DSO"
  exit 1
}

python python/tensorrt_model_connect/families/qwen/examples/byok_flashinfer/export_flashinfer_kernel.py \
  --output "$FI_WORK/flashinfer-qwen3.so"

sed "s/@SHA256@/$(sha256sum "$FI_WORK/flashinfer-qwen3.so" | awk '{print $1}')/" \
  python/tensorrt_model_connect/families/qwen/examples/byok_flashinfer/qwen3-flashinfer.yaml.in \
  > "$FI_WORK/qwen3-flashinfer.yaml"

"$TRTMC" build "${BUILD_ARGS[@]}" \
  --kernel "$FI_WORK/qwen3-flashinfer.yaml" \
  -o "$FI_WORK/qwen3-flashinfer.trtfb"

"$TRTMC" run "$FI_WORK/qwen3-flashinfer.trtfb" \
  --prompt "Explain grouped-query attention in one sentence." \
  --max-new-tokens 32 \
  --greedy
```

That shortcut replaces all 36 instances, but it is a reference for exporting
and running FlashInfer rather than a raw graph Recipe: `--kernel` embeds the DSO
in the bundle, so changing it requires another bundle build. Repeat step 4
against the native bundle before accepting it. Do not describe it as a
load-time-swappable raw graph Recipe.

## Level 2: choose an arbitrary region yourself

Use this path when no family Recipe matches the boundary your kernel needs.
The build and runtime mechanisms stay identical; only selection changes.

### 6. Inspect and circle raw TRT nodes

Capture the raw decode graph, then list its final layers:

```bash
"$TRTMC" graph inspect \
  --engine-role decode \
  --snapshot "$WORK/decode.graph.json" \
  "${BUILD_ARGS[@]}"

"$TRTMC" graph list "$WORK/decode.graph.json" \
  | tee "$WORK/decode.nodes.txt" \
  | tail -n 10
```

For the pinned revision, the tail includes:

```text
node:3598  CONSTANT
node:3599  CAST
node:3600  ELEMENTWISE
node:3601  CAST          -> logits
```

The first three nodes are the region used by the Recipe. Select them manually
and leave the final FP32 logits cast in the graph:

```bash
export BINDING_ID=qwen3.decode.logits_copy.manual@1
NODES=(node:3598 node:3599 node:3600)

"$TRTMC" graph select "$WORK/decode.graph.json" \
  --nodes "${NODES[@]}" \
  --binding-id "$BINDING_ID" \
  --workspace-bytes 0 \
  -o "$WORK/manual.selection.json"
```

Node IDs above are a receipt for the pinned build, not a stable API. Always
copy IDs from your own snapshot. Use `--match GLOB` to filter the displayed
node ID, operation, or name, then follow tensor edges. `--match` only changes
the display; it never selects nodes. For programmatic TensorRT layers, the
`OP` column appends the current `.op` subtype when available, for example
`LayerType.ELEMENTWISE/ElementWiseOperation.SUM`.

Start with the smallest boundary that matches the kernel. The requested nodes
must form one connected, convex region: no path may leave the region and later
re-enter it.

For a dynamic output, add:

```text
--output-shape-like-input INPUT_INDEX
```

The chosen boundary input must have the same dtype and declared shape as the
output. A `-1` means the DSO must handle every runtime shape allowed by the
engine profile; Model Connect does not infer shape relationships.

For fixed scalar or null arguments, repeat strict JSON arguments in call order:

```text
--extra-arg '{"type":"int","value":32}'
--extra-arg '{"type":"float","value":0.5}'
```

Allowed types are `none`, signed 32-bit `int`, finite `float`, and null `ptr`.
Build the slot-ready bundle through the existing advanced path:

```bash
"$TRTMC" build "${BUILD_ARGS[@]}" \
  --graph-patch "$WORK/manual.selection.json" \
  -o "$WORK/qwen3-manual-slot-ready.trtfb"
```

For this same logits boundary, reuse the identity-copy DSO. For any other
boundary, first implement the exact ABI in its selection JSON using the rules
in step 2. Then repeat the load-time binding and verification in steps 3 and
4, reading `abi_sha256` from `manual.selection.json`. A matching operation name
alone never makes an existing DSO compatible.

## Current graph-slot limits

- Only native TensorRT bundles with TVM-FFI enabled are supported. Graph slots
  reject TensorRT-RTX, optimized-runtime, quantized, and tensor-parallel builds.
- One build replaces one connected, convex region with exactly one output in
  one engine role. A recipe instance does not bypass this one-region limit.
- A region cannot contain a network output, shape or host tensor, control-flow
  or collective layer, or an existing plugin layer.
- Boundary tensors must be linear, rank 8 or lower, and use BF16, FP16, FP32,
  or INT32.
- The selection is tied to its graph fingerprint and build metadata. Reuse the
  exact build arguments and recapture after graph-producing code changes.
- Parser-created or otherwise raw TRT layers may expose only their base
  `LayerType` in a snapshot. Treat the `OP` and `NAME` columns as inspection
  aids, not a semantic kernel contract; the selected tensor boundary and your
  DSO implementation remain authoritative.
- Binding occurs only while a new pipeline loads. There is no in-place rebind
  API for a running pipeline.
- v1 rejects a selected engine whose deserialization is deferred until first
  inference instead of completing during pipeline load.
- Slot-ready bundles load through the CLI or C++ binding overload. The current
  C-linkage API has no kernel-binding argument.
