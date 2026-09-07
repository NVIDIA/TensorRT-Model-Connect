---
title: Runtime Identity
---

The current runtime has no strategy registry. A format-1 bundle names three
identities directly:

| Header field | Meaning |
| --- | --- |
| `family` | The only family DSO that may implement the bundle. |
| `task` | The abstract Task interface the family factory must return. |
| `backend` | The only Engine backend DSO the loader may open. |

For example, a Qwen text bundle can name `qwen`, `text_generation`, and `trt`.
The runtime opens `libtrtmc_model_qwen.so` and `libtrtmc_backend_trt.so` from
the caller-supplied runtime root. It does not search aliases, consult a
registry, or try another implementation.

## Family identity versus Task identity

Several families may implement the same abstract Task interface without
depending on each other:

```text
families/qwen/runtime     ----> ITextGeneration
families/llama/runtime    ----> ITextGeneration
families/whisper/runtime  ----> ITranscription
```

The arrows mean the family implementations depend on and realize the
interfaces in `core/runtime/include/trtmc/task.h`. The Task API never depends
on a concrete family.

A family owns all model-specific dispatch behind its returned interface. If it
needs multiple engines or request modes, that decision stays in
`families/<family>/runtime/`; it does not become a shared strategy key.

## Inspect the actual contract

```bash
trtmc inspect model.bundle
```

The output is the source of truth for `format`, `family`, `task`, `backend`, and
section locations. Exact checkpoint support remains in
`families/<family>/tests/manifests/*.json`.
