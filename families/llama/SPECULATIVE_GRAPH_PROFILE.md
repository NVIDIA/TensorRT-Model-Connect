# Speculative decoding graph and runtime profile

Keeping features on device and selecting greedy IDs with TensorRT graph
operations reduces MC OOTB's GPU-idle time from 79.72 to 16.65 ms per request.
GPU activity occupies 273.74 ms of the resulting 290.38 ms request.

## Trace scope and interpretation

September 22, 2026, one GB100/SM100, TensorRT 11.1.0.106, FP16 Llama 3.1 8B +
EAGLE3, batch one, IST=1024, 101 output IDs, depth-four chain, split1024 profiles,
CUDA graphs off. Each row is the mean of three warm Nsight Systems requests.
Request time includes reset, prefill and decoding; loading is excluded.

| Runtime | Traced wall, ms | GPU busy, ms | GPU idle, ms | Kernel calls/request |
|---|---:|---:|---:|---:|
| MC OOTB before device residency | 336.36 | 256.63 | 79.72 | 15,699 |
| MC OOTB with device residency | 290.38 | 273.74 | 16.65 | 16,534 |

GPU busy is the union of kernel, memcpy and memset intervals within each
request. GPU idle is the remainder. These are disjoint durations rather than
chronological intervals. CPU waits can overlap GPU work and cannot be added
to GPU duration to estimate wall time.
Idle time includes host policy, preparation, submission and synchronization
gaps; it does not measure CPU computation alone.

Before/after model-plan sections are byte-identical. The baseline includes the
[finite-check allocation fix](SPECULATIVE_HOST_RUNTIME.md). Device residency
adds feature transport, reusable buffers and independent TensorRT selection
plans; see their [contract](SPECULATIVE_RESIDENT_RUNTIME.md).

The latest [unprofiled AR/EAGLE3 comparison](SPECULATIVE_PERFORMANCE.md) comes
from a separate run and worker. Its medians must not be mixed with these trace
means. Clocks were unlocked in both experiments.

## Transfers and selection

The first OOTB chain request's D2H traffic falls from 116,022,272 bytes across
254 copies to **2,216 bytes across 127 copies**. Features never visit the host;
only selected IDs and finite flags return. H2D traffic falls from 58,470,620
to 19,247,324 bytes. Dense masks and control inputs remain host-generated.
Two 8.389 MB prefill-mask uploads are the largest remaining H2D transfers.

The standalone selector adds 835 kernel calls: five per target call and seven
per draft call. The reduction cost replaces full-logit transfer and CPU scans.
Model kernel durations also vary with clock and duty cycle, so the increase in
total GPU-busy time cannot all be attributed to the selector.

## GPU overlap and remaining work

MC OOTB's summed kernel duration is 276.66 ms and its kernel interval union is
271.21 ms. Summed durations can count overlapping execution more than once;
use interval unions to estimate time occupied by GPU activity. Overlap alone
does not establish useful concurrent work or a recoverable speedup.

1. Profile attention/GEMM lowering, layout conversions and scheduling on the
   critical path. MC's attention is composed from graph primitives; TensorRT
   owns fusion and tactic selection.
2. Fuse selection into model output graphs or reduce selector launches while
   preserving stable ties and per-row non-finite checks.
3. Reduce mask/control uploads and repeated shape/binding submission; then
   evaluate graph replay. The remaining 16.65 ms of GPU-idle time bounds the
   benefit of eliminating idle gaps alone on this traced workload.

## Evidence and reproducibility

The before/after measurements use MC revisions
`7714178cc2afd7d8ecdc474477d1670ba93f1f12` and
`f0b03c45cd46a65c31ea6c593d4e789e48e5d26b`, respectively.

The source data is from `primitives-before.analysis.json` and
`primitives-resident.analysis.json` in
`/home/trentl/Working/specdecode-resident-runtime/artifacts/`. The experiment
README records scripts, plan hashes, raw captures and analysis. Capture used
CUDA/NVTX tracing, no CPU sampling, and a CUDA-profiler-API request range.
No profiling instrumentation is included in production MC.

This is one repetitive, high-acceptance fixture. MC retains checkpoint RoPE
scaling. Agreement between execution modes here does not establish general
model quality or long-context accuracy.
