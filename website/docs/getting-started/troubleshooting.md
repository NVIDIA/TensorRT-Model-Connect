---
title: First-run Troubleshooting
description: Diagnose the Qwen smoke test by environment, build, bundle, load, and request boundary.
---

Do not change several model flags at once. Identify the first boundary that
fails in the [Quick Start](quick-start.md).

| Failure | Check first | Next action |
| --- | --- | --- |
| `nvidia-smi` or GPU access | Host driver/container runtime | Fix GPU visibility before installing or building. |
| `trtmc` not found | Active wheel environment or source build path | Run `trtmc version` or `./build/trtmc version` in the same shell. |
| Hugging Face 401/403/not found | Exact `hf_id`, network, auth, gated access | Verify the checkpoint and cache; do not substitute a same-family model silently. |
| CMake cannot find CUDA/TensorRT | Development environment | Return to the supported container or provide the matching headers/libraries. |
| Build runs out of memory/disk | Full-context model and workspace budget | Free capacity or choose a separately declared smaller configuration. |
| Bundle inspection fails | Incomplete/corrupt output | Rebuild and retain the first builder error. |
| Native plugin not registered | Model DSO/search path | Confirm the owning runtime DSO and pass `--model-plugin-dir` when needed. |
| TensorRT or DSO ABI error | Mixed build/runtime cohort | Run with the same compatible environment used to build/package the artifact. |
| Output differs | Prompt, revision, precision, decoding | Reproduce the exact manifest and deterministic settings before changing thresholds. |

Collect this receipt when asking for help:

```text
Git commit:
Install path: wheel | source
Model ID and revision:
Build command:
Run command:
Bundle checksum:
GPU / driver / CUDA / TensorRT:
First failing command:
Complete error:
```

For failures after the first smoke test, use
[Troubleshooting](../release-support/troubleshooting.md) and the task-specific
User Guide.
