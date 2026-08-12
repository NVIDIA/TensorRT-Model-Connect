---
title: Inspect a Bundle
description: Identify bundle ownership and execution path before loading or debugging it.
---

```bash
trtmc inspect model.bundle
trtmc inspect model.bundle --list-engines
```

Use regular inspection first. It reports header metadata and section names.
Then classify the artifact:

| Signal | Meaning |
| --- | --- |
| `runtime_strategy` and native plan sections | The native model DSO and backend path own runtime dispatch. |
| `optimized_runtime.json` | A platform-specialized implementation claims the bundle before native dispatch. |
| Tokenizer/processor/config sections | The runtime has packaged model assets it may need without reopening the source checkpoint. |
| TensorRT compatibility metadata | The loader can check whether the current runtime cohort is compatible. |

`--list-engines` recognizes native plan naming. An optimized bundle may use
provider-owned artifact names, so a nonzero “no engine sections” result is not
by itself proof that the optimized bundle is invalid.

If native loading reports that no plugin is registered, compare the exact
strategy with `src/runtime/models/<owner>/MODEL.toml`, confirm the owning model
DSO was built, and pass its directory with `--model-plugin-dir` if it is not in
the default search path.

For a guided artifact-debugging lab, use
[Inspect Bundles](../tutorials/beginner/inspect-bundles.md).

{/* Collaborative review anchor: batch 2. */}
