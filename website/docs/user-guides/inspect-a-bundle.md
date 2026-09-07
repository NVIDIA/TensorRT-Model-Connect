---
title: Inspect a Bundle
description: Identify bundle ownership and execution requirements before loading it.
---

```bash
trtmc inspect model.bundle
```

Inspection reads the bounded container and prints JSON without loading a
backend or family DSO. Its shared fields are:

| Field | Meaning |
| --- | --- |
| `format` | Bundle container version. |
| `family` | Family DSO that owns runtime behavior. |
| `task` | Abstract Task interface the family returns. |
| `backend` | Engine API implementation to load. |
| `sections` | Names with byte offsets and lengths. |

Section names and contents belong to the family. One family may store a single
engine while another stores prefill/decode engines, tokenizer assets, or task
metadata. Shared inspection deliberately does not infer model behavior from a
section name.

Before execution, confirm that `--runtime-root` points to compatible core,
runtime, backend, and `libtrtmc_model_<family>.so` files. Then use the command
matching `task`; asking for another interface fails instead of silently
choosing a different implementation.

For a guided artifact lab, use
[Inspect Bundles](../tutorials/beginner/inspect-bundles.md).
