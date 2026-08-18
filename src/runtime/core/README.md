# Core Runtime

Model-neutral CUDA/runtime infrastructure linked into `trtmc_core`.

Key files:

- `trt_common.*`: process-wide TRT logging and CUDA graph capture/replay.
- `cuda_common.*`: move-only CUDA stream and buffer wrappers.
- `device_tensor.cpp` with `include/trtmc/runtime/device_tensor.h`:
  GPU-resident tensor allocation and copies.
- `distributed_runtime.cpp` with
  `include/trtmc/runtime/distributed_runtime.h`: tensor-parallel environment
  discovery and NCCL rendezvous.
- `pipeline_pool.cpp` with `include/trtmc/runtime/pipeline_pool.h`:
  concurrent pipeline leasing and pool-wide LoRA lifecycle.
- `stb_impl.cpp`: stb implementation unit for public image I/O.

On the native path, TensorRT backend DSO loading plus generic engine/context
execution and tensor binding live under `src/runtime/backend/`. Decode loops,
sampling, masks, and KV-cache policy are model-owned under
`src/runtime/models/<family>/`. Optimized bundles instead use the generic host
under `src/runtime/providers/` to verify and load their embedded implementation
DSO.

<!-- Collaborative review anchor: batch 2. -->
