---
title: "Bring Your Own Kernel"
---

This tutorial replaces Qwen3-8B decode attention with a FlashInfer TVM-FFI
kernel. The same workflow works for any kernel that implements a slot already
published by a model family.

You bring only:

- a trusted TVM-FFI shared library (`.so`);
- a small YAML file.

You do not edit Model Connect and you do not write TensorRT C++. The Qwen
family owns where the slot connects to the TensorRT graph.

Before you start, complete [Installation](../../getting-started/installation.md),
clone this repository, and run every command below from its root directory.
Building the included exporter requires an SM 10.3 GPU; on another supported
GPU, bring an already compiled DSO that implements the same slot instead. The
reference Qwen3-8B build produced about 31 GB per bundle, so reserve at least
70 GB of disk for the native and patched pair.

## 1. See the available slots

Pin the model revision so that the model and slot contract cannot change under
you:

```bash
export MODEL=Qwen/Qwen3-8B
export REVISION=b968826d9c46dd6066d109eabc6255188de91218

trtmc kernel slots "$MODEL" --model-revision "$REVISION"
```

For this tutorial, the output includes:

```text
qwen.decode_attention@1
  Single-token post-RoPE grouped-query attention over native Qwen KV cache exposed as 64-token pages.
  inputs:
    query: bfloat16 [num_query_heads, head_dim]
    key: bfloat16 [1, num_kv_heads, kv_capacity, head_dim]
    value: bfloat16 [1, num_kv_heads, kv_capacity, head_dim]
    key_value_lengths: int32 [1]
    page_offsets: int32 [1]
    page_table: int32 [num_pages]
  outputs:
    context: bfloat16 same_as_input_0
  workspace: 0 bytes
  instances (36):
    decoder.layers.0.decode_attention
    ...
    decoder.layers.35.decode_attention
  model arguments:
    softmax_scale: float32
```

A slot is a model-owned call site. It defines tensor shapes, data types,
workspace, argument order, and how inputs and outputs connect to TensorRT.
YAML selects a slot; it does not try to rediscover model semantics.
For this slot, `key_value_lengths[0]` is the live prefix length,
`page_offsets[0]` is zero, and `page_table` is the identity mapping over
64-token pages.

## 2. Prepare a TVM-FFI kernel

If you already have a compatible DSO, skip this step. This repository also
contains the small FlashInfer CuTe exporter used by the example:

```bash
python -m pip install \
  "flashinfer-python==0.6.15" \
  "nvidia-cutlass-dsl==4.5.0" \
  "apache-tvm-ffi==0.1.12"

mkdir -p artifacts/qwen3-byok

python python/tensorrt_model_connect/families/qwen/examples/byok_flashinfer/export_flashinfer_kernel.py \
  --output artifacts/qwen3-byok/flashinfer-qwen3.so
```

The exporter requires FlashInfer, CUTLASS DSL, TVM FFI, CUDA PyTorch, and an
SM 10.3 GPU. It exports a function named `run` with the ABI owned by
`qwen.decode_attention@1`. The example was tested with FlashInfer 0.6.15,
CUTLASS DSL 4.5.0, and Apache TVM FFI 0.1.12.

For the pinned Qwen3-8B revision, the symbolic dimensions above resolve to
32 query heads, 8 KV heads, head dimension 128, and KV capacity 40960. It uses
64-token pages, so the page table has 640 entries. The included exporter is
statically specialized to those values and an SM 10.3 GPU; do not reuse its DSO
for another model merely because that model publishes a slot with the same name.

:::warning Trusted native code only
A DSO can execute native code when it is loaded. SHA-256 checks identity, not
safety. Use only a library that you trust.
:::

## 3. Write the YAML

Copy your DSO beside the YAML, calculate its digest, and fill the example
template:

```bash
export WORK="$PWD/artifacts/qwen3-byok"
export DSO="$WORK/flashinfer-qwen3.so"

sha256sum "$DSO"

sed "s/@SHA256@/$(sha256sum "$DSO" | awk '{print $1}')/" \
  python/tensorrt_model_connect/families/qwen/examples/byok_flashinfer/qwen3-flashinfer.yaml.in \
  > "$WORK/qwen3-flashinfer.yaml"

cat "$WORK/qwen3-flashinfer.yaml"
```

The complete contract is intentionally small:

```yaml
schema_version: 1
slot: qwen.decode_attention@1
instances:
  all: true
  expect_count: 36
kernel:
  library: ./flashinfer-qwen3.so
  sha256: <the lowercase SHA-256 printed above>
  function: run
```

`expect_count` prevents an accidental partial replacement if a different
model variant has another layer count. To replace only specific layers, use:

```yaml
instances:
  ids:
    - decoder.layers.0.decode_attention
```

Qwen owns the slot's tensor ABI, so the YAML does not repeat shapes, ports,
workspace, TensorRT layer names, or Model Connect's internal function alias.

## 4. Build native and patched bundles

First build the native comparison bundle:

```bash
trtmc build "$MODEL" \
  --model-revision "$REVISION" \
  --precision bf16 \
  --max-cache-length 40960 \
  -o "$WORK/qwen3-native.trtfb"
```

Then add only `--kernel`:

```bash
trtmc build "$MODEL" \
  --model-revision "$REVISION" \
  --precision bf16 \
  --max-cache-length 40960 \
  --kernel "$WORK/qwen3-flashinfer.yaml" \
  -o "$WORK/qwen3-flashinfer.trtfb"
```

Build validates the YAML, DSO digest, slot, and instance count. It then
connects the slot directly and embeds the exact DSO in the output bundle.
There is no capture, lowering-map, selection-lock, or graph-patch workflow.

## 5. Check correctness

Run deterministic generation through both bundles:

```bash
PROMPT="$(<"$PWD/python/tensorrt_model_connect/families/qwen/examples/byok_flashinfer/prompt.txt")"
COMMON_ARGS=(
  --prompt "$PROMPT"
  --max-new-tokens 64
  --greedy
)

trtmc run "$WORK/qwen3-native.trtfb" "${COMMON_ARGS[@]}" \
  > "$WORK/native.txt"

trtmc run "$WORK/qwen3-flashinfer.trtfb" "${COMMON_ARGS[@]}" \
  > "$WORK/flashinfer.txt"

diff -u "$WORK/native.txt" "$WORK/flashinfer.txt"
```

An empty diff passes this smoke. If it fails, use the native bundle and fix
the kernel or YAML.

## 6. Check performance does not regress

Use a longer fixed prompt so attention work is measurable, then benchmark both
bundles on the same idle GPU:

```bash
PERF_UNIT='Grouped-query attention shares key and value heads across several query heads, reducing cache bandwidth while preserving distinct query projections. This fixed paragraph creates a stable decode benchmark with enough context for attention work to be measurable. '
PERF_PROMPT=
for _ in {1..12}; do
  PERF_PROMPT+="$PERF_UNIT"
done

trtmc run "$WORK/qwen3-native.trtfb" \
  --prompt "$PERF_PROMPT" --max-new-tokens 64 --greedy \
  --warmup 3 --benchmark 10 > "$WORK/native.perf.log" 2>&1

trtmc run "$WORK/qwen3-flashinfer.trtfb" \
  --prompt "$PERF_PROMPT" --max-new-tokens 64 --greedy \
  --warmup 3 --benchmark 10 > "$WORK/flashinfer.perf.log" 2>&1
```

Compare decode throughput:

```bash
python - "$WORK/native.perf.log" "$WORK/flashinfer.perf.log" <<'PY'
import re
import sys

def throughput(path):
    text = open(path, encoding="utf-8").read()
    values = [float(x) for x in re.findall(r"tokens_per_sec=([0-9.]+)", text)]
    if not values:
        raise SystemExit(f"{path}: tokens_per_sec was not reported")
    return values[-1]

native = throughput(sys.argv[1])
patched = throughput(sys.argv[2])
print(f"native={native:.2f} patched={patched:.2f} tokens/s")
if patched < native:
    raise SystemExit("FAIL: the external kernel is slower")
print("PASS: patched is not slower than native")
PY
```

The gate is simple: on the same model revision, prompt, settings, and hardware,
the patched throughput must be at least the native throughput. In the reference
SM 10.3 run used to qualify this example, native measured 88.68 tokens/s and
FlashInfer measured 94.62 tokens/s, with byte-identical generated text. Treat
those numbers as a receipt, not a promise for different hardware.

## What requires a Model Connect change?

- Another kernel for `qwen.decode_attention@1`: no change; bring YAML + DSO.
- A different subset of the 36 instances: no change; edit `instances`.
- A new operation boundary or tensor ABI: the owning model family adds one
  new versioned slot.

This boundary keeps every model family encapsulated while letting kernel
authors use the same two-file workflow.
