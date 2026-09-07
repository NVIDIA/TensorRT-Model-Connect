---
title: Advanced Tutorial - Quantization and Runtime Knobs
---

import Diagram from '@site/src/components/Diagram';

## Learning objectives

- distinguish build, load, and Task-request inputs;
- find the family that owns a precision or quantization path;
- use Llama's runtime-sized KV cache without generalizing it to other families;
- select the optional TensorRT-RTX backend explicitly.

<Diagram
  src="/img/diagrams/tutorials/advanced/knob-scopes.svg"
  alt="Build inputs flow into one family-owned bundle, explicit load options reach one backend and family, and request options reach the abstract Task implementation"
  caption="A value belongs to exactly one boundary. There is no central layered configuration registry."
/>

## Precision

Precision is a build input:

```bash
python -m tensorrt_model_connect build MODEL \
  --output model-fp16.bundle \
  --precision fp16
```

The selected family validates the value and decides how weights and TensorRT
layers use it. The shared parser accepting `fp16`, `bf16`, or `fp32` does not
prove every family supports each value.

## Quantization

Quantization is also family-owned:

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --output qwen3-fp8.bundle \
  --precision fp16 \
  --quantization fp8
```

<Diagram
  src="/img/diagrams/tutorials/advanced/quantization-flow.svg"
  alt="One family consumes checkpoint weights and its own quantization inputs before writing a TensorRT engine into the bundle"
  caption="Calibration and graph changes stay in the family that implements and tests them."
/>

Before running the example, install only that family's declared dependencies:

```bash
python -m pip install -r families/qwen/requirements.txt
```

Read `families/qwen/model.py`, `families/qwen/quantization.py`, and a matching
manifest. Do not reuse this command for a different family unless its own
builder and test declare the same path.

## Dynamic KV cache

The current runtime-sized KV path is owned by Llama. Build it explicitly:

```bash
python -m tensorrt_model_connect build nvidia/Llama-3.1-Minitron-4B-Width-Base \
  --output minitron-dynamic-kv.bundle \
  --precision fp16 \
  --max-sequence-length 131072 \
  --dynamic-kv-cache
```

Choose the physical allocation when loading:

```bash
trtmc run minitron-dynamic-kv.bundle \
  --runtime-root /opt/trtmc/lib \
  --kv-cache-size 1GiB \
  --prompt "Explain KV caching briefly."
```

<Diagram
  src="/img/diagrams/tutorials/advanced/dynamic-kv-cache.svg"
  alt="A family-built dynamic KV bundle and explicit memory budget create model-owned cache state for a text-generation Task"
  caption="The family owns cache behavior; the shared loader only carries the explicit byte budget."
/>

The allocation cannot exceed the engine's legal TensorRT shapes. Qwen currently
rejects `--dynamic-kv-cache`; this is not a shared decoder feature.

## Explicit runtime loading

Every execution command requires the directory containing the matching native
core, loader, backend DSO, and family DSO:

```bash
trtmc inspect model.bundle
trtmc run model.bundle --runtime-root /opt/trtmc/lib --prompt "Hello"
```

The runtime does not search the current directory, environment variables,
backend directories, or model-plugin directories.

## TensorRT-RTX

Build for the optional backend only when its Python and native SDK components
are installed:

```bash
python -m tensorrt_model_connect build MODEL \
  --output model-rtx.bundle \
  --backend trt_rtx

trtmc run model-rtx.bundle \
  --runtime-root /opt/trtmc/lib \
  --runtime-cache kernels.cache \
  --cuda-graphs \
  --prompt "Hello"
```

<Diagram
  src="/img/diagrams/tutorials/advanced/rtx-runtime.svg"
  alt="An explicit TensorRT-RTX bundle loads the TensorRT-RTX backend through the abstract Engine interface"
  caption="The family implements the Task interface and the backend implements the Engine interface."
/>

Standard TensorRT bundles reject `--runtime-cache` and `--cuda-graphs`.

## Advanced knob checklist

- Is the value a build, load, or request input?
- Does the selected family manifest prove the exact combination?
- Did you install only the owning family's extra requirements?
- Does `trtmc inspect` name the expected family, task, and backend?
- Did the semantic family test run on the claimed target?

## Self-check

1. Why does a generic `--quantization` option not imply generic support?
2. Which component owns `--kv-cache-size` semantics after the loader passes it?
3. Why is `--runtime-root` mandatory?
4. What interface does a backend implement, and what behavior stays in a family?
