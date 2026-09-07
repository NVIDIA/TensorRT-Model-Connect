---
title: Beginner Tutorial - Inspect Bundles
---

import Diagram from '@site/src/components/Diagram';

## Learning objectives

- inspect a bundle without loading TensorRT engines;
- identify its family, Task interface, backend, and named sections;
- distinguish shared container fields from family-owned section meaning.

<Diagram
  src="/img/diagrams/getting-started/bundle-contents.svg"
  alt="A format-1 bundle contains a small shared header and family-owned named byte sections"
  caption="Core validates only the bounded container. The selected family owns every section's semantics."
/>

## Why inspect first

The format-1 header contains exactly:

```json
{
  "format": 1,
  "family": "gpt2",
  "task": "text_generation",
  "backend": "trt",
  "sections": {
    "runtime.json": {"offset": 0, "length": 42},
    "engine.plan": {"offset": 42, "length": 1234}
  }
}
```

Core validates field types, safe identities, section bounds, and file layout.
It does not interpret model configuration or calculate section hashes.

## Build an example

```bash
python -m tensorrt_model_connect build openai-community/gpt2 \
  --output gpt2.bundle \
  --precision fp16
```

## Use the inspector

```bash
trtmc inspect gpt2.bundle
```

Inspection needs neither `--runtime-root` nor a GPU execution. It reads the
header and emits JSON. The current CLI has no `--list-engines` mode; section
names are already listed in the output.

## Fields to check

| Field | Question |
| --- | --- |
| `format` | Is it the only supported value, `1`? |
| `family` | Which `libtrtmc_model_<family>.so` must be present? |
| `task` | Which abstract interface must the factory return? |
| `backend` | Which `libtrtmc_backend_<backend>.so` must be present? |
| `sections` | Which family-owned payload names and bounded locations exist? |

There is no model ID, precision, TensorRT version, runtime strategy, plugin
registry key, or content digest in the shared header. If a family needs one of
those values, it stores and validates it in its own section.

## Read loading like a runtime engineer

<Diagram
  src="/img/diagrams/tutorials/beginner/native-runtime-dispatch.svg"
  alt="The explicit loader reads family task and backend identities, opens exactly two DSOs, and returns an abstract Task implementation"
  caption="Runtime control transfer does not create a source dependency from core to a family."
/>

Given the example header and `/opt/trtmc/lib`, the loader opens:

```text
/opt/trtmc/lib/libtrtmc_backend_trt.so
/opt/trtmc/lib/libtrtmc_model_gpt2.so
```

The backend implements the abstract Engine API. The family DSO implements the
abstract Task API and depends on that Engine API. The loader does not search
another directory or another family if either file fails.

## Common mismatch

If the bundle says `family=qwen` but only `libtrtmc_model_gpt2.so` is installed,
loading fails. If the family factory returns a different Task identity than the
header, loading also fails. Rebuild the bundle and runtime from the same source
tree; do not add an alias or compatibility shim.

## Debugging checklist

1. Run `trtmc inspect BUNDLE`.
2. Confirm the file is format 1 and every required identity is non-empty.
3. Confirm the explicit runtime root contains the exact family and backend DSOs.
4. Read `families/<family>/model.py` for section names.
5. Read `families/<family>/runtime/` for how those sections are consumed.
6. Run the exact family-owned testcase.

## Learning Log Prompts

- Which fields belong to core and which belong to the family?
- Why does inspecting a bundle not prove an engine can execute?
- What dependency direction exists between a family implementation and the
  abstract Task interface?
- Why does the loader fail instead of searching for an alternative?
