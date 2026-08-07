---
title: "Bring Your Own Kernel with TVM FFI"
description: Replace a family-owned TensorRT graph region with a load-time TVM-FFI kernel and validate the result.
---

import Diagram from '@site/src/components/Diagram';

This tutorial starts by replacing part of a Qwen3-8B TensorRT graph with a
TVM-FFI kernel, then manually replaces a DistilBERT region with a CuTe DSL
kernel. You do not write TensorRT C++.

## Learning objectives

By the end of this lab, you should be able to select a family recipe or explicit
graph region, explain its ABI receipt and load-time binding, and require both
numerical correctness and a measured no-regression gate before promoting a
kernel.

For the API contract, lifecycle, and supported limits, see the
[TVM FFI feature reference](../../features/tvm-ffi.md).

Start with the simplest graph workflow and move to the escape hatch only when
you need it:

| Level | How the region is chosen | Who should use it |
| --- | --- | --- |
| Recommended | The model family publishes a versioned recipe and you choose one exact instance. | Most kernel authors. |
| Advanced | You inspect the raw TensorRT graph and type every node ID yourself; if needed, you also author the TVM-FFI kernel. | Authors whose region has no recipe or supplied DSO. |

A recipe is only a family-owned shortcut for a known manual selection. It
records exact TRT node IDs, workspace, scalar arguments, and output-shape rule
while the family constructs the graph. `build --recipe` automatically captures
the graph, passes those values to the same `select_region()` validator used by
the advanced path, and then runs the ordinary graph-patch build. It adds no
semantic graph, matching language, plugin schema, or runtime behavior.

Both levels use the same backend:

<Diagram
  src="/img/diagrams/tutorials/advanced/byok-workflow.svg"
  alt="Recipe and manual graph selection paths converging on region validation, a slot-ready bundle, load-time kernel binding, and a newly constructed pipeline"
  caption="A Recipe changes how you select the region, not how Model Connect validates, patches, binds, or runs it."
/>

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
  -o "$WORK/qwen3-slot-ready.bundle"
```

The command internally performs the existing capture, Recipe resolution,
`select_region()`, and `--graph-patch` steps. It writes two files:

```text
qwen3-slot-ready.bundle
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
qwen.decode_attention_region@2       decoder.layers.0.decode_attention  node:83,node:84
...
qwen.decode_attention_region@2       decoder.layers.35.decode_attention node:3548,node:3549
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
"$TRTMC" run "$WORK/qwen3-slot-ready.bundle" \
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
"$TRTMC" build "${BUILD_ARGS[@]}" -o "$WORK/qwen3-native.bundle"
```

Compare deterministic output:

```bash
TEST_PROMPT='Explain grouped-query attention, KV caching, and decode latency.'
MAX_NEW_TOKENS=64

"$TRTMC" run "$WORK/qwen3-native.bundle" \
  --prompt "$TEST_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --output "$WORK/native.jsonl"

"$TRTMC" run "$WORK/qwen3-slot-ready.bundle" \
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
"$TRTMC" run "$WORK/qwen3-native.bundle" \
  --prompt "$TEST_PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --greedy \
  --warmup 3 --benchmark 10 > "$WORK/native.perf.log" 2>&1

"$TRTMC" run "$WORK/qwen3-slot-ready.bundle" \
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

Qwen also records one decode-attention Recipe per layer:

```bash
"$TRTMC" build "${BUILD_ARGS[@]}" \
  --recipe qwen.decode_attention_region@2 \
           decoder.layers.0.decode_attention \
  -o "$WORK/qwen3-attention-slot.bundle"
```

For Qwen3-8B layer 0, the printed boundary in
`qwen3-attention-slot.selection.json` defines this ordered contract:

```text
input[0]  scaled query       BF16  [1, 32, -1, 128]
input[1]  key cache          BF16  [1, 8, 40960, 128]
input[2]  value cache        BF16  [1, 8, 40960, 128]
input[3]  key/value lengths  INT32 [1]
output[0] context            BF16  [1, 32, -1, 128]
```

The Qwen builder applies the model's query scale and BF16 rounding before the
selected region. The external kernel therefore consumes the already-scaled
query and uses a softmax scale of `1.0`.

Export any CUDA DSL, FlashInfer, or custom CUDA kernel through TVM-FFI with
that exact ABI and then reuse the slot-ready build and load-time binding steps.
No Model Connect change is required for another DSO with this contract.

The Qwen family contains a small FlashInfer linear-KV POC for this exact
Recipe boundary. Building the DSO is kernel-integrator work; an ordinary model
user can receive the resulting `.so` and start at the binding command below.

:::warning SM 10.3 proof of concept
The supplied FlashInfer exporter currently refuses GPUs whose CUDA compute
capability is not exactly SM 10.3. It is an integration example for that target,
not a portable prebuilt kernel.
:::

FlashInfer 0.6.15 currently needs a small optional device-length patch because
Model Connect keeps a fixed-capacity KV tensor and passes its active length as
an `int32[1]` CUDA tensor:

```bash
python -m pip install \
  "nvidia-cutlass-dsl==4.5.0" \
  "apache-tvm-ffi==0.1.12" \
  "flashinfer-python==0.6.15"

git clone --branch v0.6.15 --depth 1 \
  https://github.com/flashinfer-ai/flashinfer.git \
  "$WORK/flashinfer-v0.6.15"

git -C "$WORK/flashinfer-v0.6.15" apply \
  "$PWD/python/tensorrt_model_connect/families/qwen/kernels/flashinfer_device_kv_length.patch"

PYTHONPATH="$WORK/flashinfer-v0.6.15:$PWD/python" \
  python python/tensorrt_model_connect/families/qwen/kernels/export_flashinfer_decode_attention.py \
  --output "$WORK/qwen3-flashinfer-linear.so"
```

Bind that DSO to the attention Recipe when constructing a pipeline:

```bash
export ATTENTION_BINDING_ID=qwen.decode_attention_region@2
export ATTENTION_ABI_SHA256="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["abi_sha256"])' \
    "$WORK/qwen3-attention-slot.selection.json"
)"

cat > "$WORK/attention-kernel-bindings.json" <<EOF
{
  "schema_version": 1,
  "bindings": [
    {
      "id": "$ATTENTION_BINDING_ID",
      "abi_sha256": "$ATTENTION_ABI_SHA256",
      "library": "./qwen3-flashinfer-linear.so",
      "function": "run"
    }
  ]
}
EOF

"$TRTMC" run "$WORK/qwen3-attention-slot.bundle" \
  --kernel-bindings "$WORK/attention-kernel-bindings.json" \
  --prompt "Explain grouped-query attention in one sentence." \
  --max-new-tokens 32 \
  --greedy
```

The DSO is external to the bundle. To try another implementation of the same
ABI, point a new binding manifest at it and construct a new pipeline; the
slot-ready bundle does not change.

This FlashInfer exporter is an integration POC for the documented boundary,
not a generally qualified built-in kernel. Before shipping a kernel, run the
same deterministic output and native-versus-external performance checks shown
above on every model shape and GPU you support. Successfully loading a DSO
proves only that the binding manifest matched and its named function resolved;
it does not prove that the function implements the ABI, model accuracy, or a
performance improvement.

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
  -o "$WORK/qwen3-manual-slot-ready.bundle"
```

For this same logits boundary, reuse the identity-copy DSO. For any other
boundary, first implement the exact ABI in its selection JSON using the rules
in step 2. Then repeat the load-time binding and verification in steps 3 and
4, reading `abi_sha256` from `manual.selection.json`. A matching operation name
alone never makes an existing DSO compatible.

### 7. Manually author and bridge a CuTe DSL kernel

Step 6 changed only selection and reused a supplied DSO. This second worked
example changes both halves: you manually select a raw TensorRT residual-add
node, then write and export the matching CuTe DSL kernel.

The example uses DistilBERT so graph capture and compilation stay small. Its
fixed boundary is useful for learning the mechanics, but the single scalar add
is **not** an acceleration candidate. The validated measurement below rejects
it.

Set up a separate pinned build:

```bash
export CUTE_REVISION=12040accade4e8a0f71eabdb258fecc2e7e948be
export CUTE_MODEL="${CUTE_MODEL:-distilbert/distilbert-base-uncased}"
export CUTE_WORK="$PWD/artifacts/distilbert-cutedsl"
export CUTE_BINDING_ID=distilbert.layer0.attention_residual_add.cutedsl@1
mkdir -p "$CUTE_WORK"

CUTE_BUILD_ARGS=(
  "$CUTE_MODEL"
  --model-revision "$CUTE_REVISION"
  --precision fp16
)
```

The Hub ID requires normal Hugging Face access. In an offline environment, set
`CUTE_MODEL` to the absolute directory of that exact cached revision before
running the block. Keep the same value for inspect and both builds.

Capture the raw graph and narrow the display to elementwise sums:

```bash
"$TRTMC" graph inspect \
  --engine-role decode \
  --snapshot "$CUTE_WORK/decode.graph.json" \
  "${CUTE_BUILD_ARGS[@]}"

"$TRTMC" graph list "$CUTE_WORK/decode.graph.json" \
  --match '*ElementWiseOperation.SUM*' \
  | tee "$CUTE_WORK/elementwise-sum.nodes.txt"
```

For this exact revision and build, the first attention residual add is:

```text
node:49  LayerType.ELEMENTWISE/ElementWiseOperation.SUM
         tensor:22,tensor:50 -> tensor:51
```

Select that node yourself:

```bash
"$TRTMC" graph select "$CUTE_WORK/decode.graph.json" \
  --nodes node:49 \
  --binding-id "$CUTE_BINDING_ID" \
  --workspace-bytes 0 \
  -o "$CUTE_WORK/residual-add.selection.json"
```

`node:49` is a receipt for this pinned graph, not a stable semantic name. After
any model revision, build-option, or graph-code change, inspect again and
follow the displayed tensor edges. Here the selection receipt plus snapshot
define this ordered contract:

```text
input[0]  hidden                FP16 device [256, 768]
input[1]  attention projection  FP16 device [256, 768]
output[0] residual              FP16 device [256, 768]
workspace 0 bytes
extra args none
```

Install the exporter dependencies in the Python environment on the target GPU:

```bash
python -m pip install \
  'nvidia-cutlass-dsl==4.5.0' \
  'apache-tvm-ffi==0.1.12'
```

The operation-specific part of
`examples/byok/export_cutedsl_residual_add.py` is ordinary CuTe DSL:

```python
ROWS, COLS, THREADS = 256, 768, 256

@cute.kernel
def residual_add_kernel(
    hidden: cute.Tensor,
    attention_projection: cute.Tensor,
    output: cute.Tensor,
):
    thread_x, _, _ = cute.arch.thread_idx()
    block_x, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    index = block_x * block_size + thread_x
    row, column = index // COLS, index % COLS
    output[row, column] = hidden[row, column] + attention_projection[row, column]

@cute.jit
def run(
    hidden: cute.Tensor,
    attention_projection: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
):
    residual_add_kernel(hidden, attention_projection, output).launch(
        grid=((ROWS * COLS) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )
```

For your own region, copy this exporter and update its shape constants,
contract check, and two CuTe functions to match your selection receipt. Keep
the TVM-FFI environment stream and compile/export/link scaffold.

The exporter supplies
`make_fake_stream(use_tvm_ffi_env_stream=True)`, so `stream` is TensorRT's
current CUDA stream delivered through TVM-FFI. It is not another argument in
the selection ABI: the exported call remains `run(input0, input1, output)`.
Do not synchronize inside the kernel wrapper or launch on the default stream.

Compile for the active GPU and export `run` through TVM-FFI:

```bash
test ! -e "$CUTE_WORK/residual-add-cutedsl.so" || {
  echo "Choose a fresh CUTE_WORK; the exporter will not overwrite a DSO"
  exit 1
}

python examples/byok/export_cutedsl_residual_add.py \
  --snapshot "$CUTE_WORK/decode.graph.json" \
  --selection "$CUTE_WORK/residual-add.selection.json" \
  --output "$CUTE_WORK/residual-add-cutedsl.so"
```

The exporter first checks that the selected node is a two-input FP16 sum with
the exact fixed shapes above. It then compiles the CuTe code, links the CuTe
runtime, and exports `__tvm_ffi_run`. The native-versus-BYOK comparison below
is the end-to-end correctness test.

Build the slot-ready and matched native bundles:

```bash
"$TRTMC" build "${CUTE_BUILD_ARGS[@]}" \
  --graph-patch "$CUTE_WORK/residual-add.selection.json" \
  -o "$CUTE_WORK/distilbert-slot-ready.bundle"

"$TRTMC" build "${CUTE_BUILD_ARGS[@]}" \
  -o "$CUTE_WORK/distilbert-native.bundle"
```

Bind the new DSO using the ABI hash computed from the selected boundary:

```bash
CUTE_ABI_SHA256="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["abi_sha256"])' \
    "$CUTE_WORK/residual-add.selection.json"
)"

cat > "$CUTE_WORK/kernel-bindings.json" <<EOF
{
  "schema_version": 1,
  "bindings": [
    {
      "id": "$CUTE_BINDING_ID",
      "abi_sha256": "$CUTE_ABI_SHA256",
      "library": "./residual-add-cutedsl.so",
      "function": "run"
    }
  ]
}
EOF
```

There is intentionally no DSO hash. The ABI hash makes the manifest target the
selected call contract while another compatible DSO can be supplied at the
next pipeline load. It does not inspect the DSO's signature; the kernel author
must still implement the ordered contract exactly.

Compare full embeddings rather than byte-diffing them. Moving the FP16 rounding
point can produce small differences that propagate through later layers:

```bash
CUTE_PROMPT='The quick brown fox jumps over the lazy dog.'

"$TRTMC" embed "$CUTE_WORK/distilbert-native.bundle" \
  --prompt "$CUTE_PROMPT" \
  > "$CUTE_WORK/native.json" 2> "$CUTE_WORK/native.stderr"

"$TRTMC" embed "$CUTE_WORK/distilbert-slot-ready.bundle" \
  --kernel-bindings "$CUTE_WORK/kernel-bindings.json" \
  --prompt "$CUTE_PROMPT" \
  > "$CUTE_WORK/byok.json" 2> "$CUTE_WORK/byok.stderr"

python - "$CUTE_WORK/native.json" "$CUTE_WORK/byok.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    native = json.load(file)["embedding"]
with open(sys.argv[2], encoding="utf-8") as file:
    byok = json.load(file)["embedding"]
if not native or len(native) != len(byok):
    raise SystemExit("FAIL: embedding lengths differ")
max_abs = max(abs(a - b) for a, b in zip(native, byok))
denominator = math.sqrt(
    sum(value * value for value in native) * sum(value * value for value in byok)
)
cosine = sum(a * b for a, b in zip(native, byok)) / denominator
print(f"values={len(native)} max_abs={max_abs:.6g} cosine={cosine:.9f}")
if max_abs > 0.02 or cosine < 0.999:
    raise SystemExit("FAIL: numerical gate")
print("PASS: numerical gate")
PY
```

The validated GB300 result was 196,608 values, maximum absolute difference
0.01563, and cosine similarity 0.999997915.

Finally, apply the same 2% no-regression rule as step 4 with a task-appropriate
benchmark. The retained validation used one warm-up per arm followed by six
alternating fresh-process `trtmc embed` pairs. It measured median engine
execution at 1.644208 ms native and 2.070288 ms BYOK on GB300:
**25.914% slower**. This example therefore fails the performance gate and must
not ship as an acceleration.

For a real optimization, circle a larger or fused high-work region whose
eliminated TensorRT work outweighs the plugin and kernel-launch overhead,
implement that selection's exact ABI, and repeat the correctness and 2%
no-regression gates. The success criterion is not “the DSO loaded”; it is
“correct output without a measured regression.”

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

## Self-check

1. What does a family recipe change compared with manual graph selection?
2. Why is “the DSO loaded” not a sufficient success criterion?
3. Can a running pipeline be rebound to a different kernel DSO in place?

<details>
<summary>Check your answers</summary>

1. It selects a versioned known region and parameters; both paths use the same
   graph capture, region validation, patching, ABI, and load-time binding.
2. The replacement must preserve task/numerical correctness and satisfy a
   task-appropriate performance gate. Load success proves neither.
3. No. Binding happens while a new pipeline is constructed; another compatible
   DSO requires constructing another pipeline.

</details>

{/* Collaborative review anchor. */}
