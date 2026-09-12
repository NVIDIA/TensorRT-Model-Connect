---
title: First-run Troubleshooting
description: Diagnose the first smoke test by environment, build, bundle, load, and request boundary.
---

Identify the first boundary that fails in the [Quick Start](quick-start.md).

| Failure | Check first | Next action |
| --- | --- | --- |
| `nvidia-smi` or GPU access | Host driver/container runtime | Fix GPU visibility before installing or building. |
| Python build module unavailable | Active wheel/editable environment | Run `python -m tensorrt_model_connect build --help` in the same shell. |
| `trtmc` unavailable | Native build/install output and `PATH` | Run `trtmc version` or the explicit built binary. |
| Hugging Face 401/403/not found | Exact ID/revision, network, authentication | Verify the checkpoint/cache; never substitute a nearby model silently. |
| CMake cannot find CUDA/TensorRT | Development image and explicit SDK paths | Return to the documented container/toolchain cohort. |
| Build OOM or disk failure | Requested checkpoint, shape, precision, and cache capacity | Use the exact family manifest/profile or free capacity; retain the first error. |
| No family or multiple families match | Root checkpoint identity metadata | Use a supported exact checkpoint; do not add prefix/fallback matching. |
| Bundle inspection fails | Partial/corrupt bundle | Rebuild; failed builds must not publish a partial output. |
| Family/backend DSO missing | Selected runtime-root contents | Confirm `libtrtmc_core.so`, `libtrtmc_runtime.so`, selected backend, and exact family DSO are together. |
| Task mismatch | Bundle `task` versus CLI command | Use the Task command named by the family manifest/header. |
| TensorRT/DSO ABI error | Mixed native product builds or incompatible hardware/software | Use one product build in a compatible environment and rebuild the bundle when required. |
| Output differs | Revision, input framing, precision, sampling, oracle | Reproduce the exact family testcase before changing code or thresholds. |

Collect the source revision, model ID/revision, complete build/run commands,
bundle inspection, GPU/driver/CUDA/TensorRT versions, runtime-root listing, and
first complete error when asking for help.
