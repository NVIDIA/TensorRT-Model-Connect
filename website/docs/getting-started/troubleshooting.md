---
title: First-run Troubleshooting
description: Diagnose the first smoke test by environment, family resolution, build, bundle, load, and request boundary.
---

Keep the first failure. Later messages are often consequences of the same
problem.

| Symptom | Boundary to inspect first |
| --- | --- |
| `nvidia-smi` fails | Host driver or GPU access. |
| Docker cannot see the GPU | NVIDIA Container Toolkit setup from the source-build page. |
| No family claims the checkpoint | Check `config.json` or `model_index.json` and the intended `families/<family>/support.py`. |
| More than one family claims it | The family support declarations overlap; resolution intentionally stops. |
| Import fails only after selection | Install the selected family's `requirements.txt`, if present. |
| TensorRT build fails | Preserve the first builder error and verify precision, shapes, GPU, CUDA, and TensorRT. |
| `trtmc inspect` fails | The bundle is incomplete or malformed; rebuild it from the first error. |
| `--runtime-root is required` | Pass the exact directory that contains the runtime, backend, and selected family DSOs. |
| Backend or family DSO cannot be loaded | Build or install that DSO in the same runtime root and check its dynamic-library dependencies. |
| Loaded family does not implement the requested Task API | Use the command matching the bundle's `task` field shown by `trtmc inspect`. |
| Output differs from a reference | Run the owning family's comparator and thresholds; plausible output alone is not parity proof. |

Inspection does not load code and therefore does not take `--runtime-root`:

```bash
trtmc inspect model.bundle
```

Every execution command does:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello" \
  --max-new-tokens 16
```

Collect this receipt when asking for help:

```text
Git commit:
Install path: wheel | source
Model ID and revision:
Selected family and task:
Build command:
Inspect output:
Run command, including --runtime-root:
GPU / driver / CUDA / TensorRT:
First failing command:
Complete error:
```

For later failures, continue with the task-specific User Guide and the owning
family's tests.
