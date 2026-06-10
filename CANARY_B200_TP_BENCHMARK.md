# Canary B200 Tensor Parallel Benchmark

Date: 2026-06-10 21:01:47 UTC
Repo: `9b39c884` on `bench/canary-tp-b200`
Host: 8x NVIDIA B200, driver `580.159.03`, compute capability `10.0`

## Setup

- Model: `nvidia/canary-1b-v2`
- Audio: `tests/e2e/data/Recording.wav`
- Runtime command: `trtmc transcribe ... --max-new-tokens 50`
- Build command shape: `python -m tensorrt_model_connect build ... --max-cache-length 128 --method trt`
- TP builds used the existing Canary TP path via `--tp-size 2`, `--tp-size 4`, and `--tp-size 8`; no source changes were needed.
- Fresh runtime build: `build-bench/trtmc`, `trtmc 0.1.0`, TRT backend enabled.
- Software: TensorRT `11.0.0.114`, CUDA nvcc `13.0`, Open MPI `4.1.6`, Nsight Systems `2025.3.2.474`, torch `2.7.1+cpu`, transformers `5.11.0`.
- Missing host packages installed for the run: Open MPI, CUDA/TensorRT dev libraries, cuBLAS/cuRAND, and Nsight Systems.

## Build Artifacts

| Config | Bundle | Size | Total build | Decoder/main TRT build | Encoder TRT build |
|---|---:|---:|---:|---:|---:|
| SD | `canary-1b-v2.trtfb` | 3.5G | 213.4s | 46.0s | 141.3s |
| TP2 | `canary-1b-v2-tp2.trtfb` | 3.6G | 255.2s | 95.1s | 130.1s |
| TP4 | `canary-1b-v2-tp4.trtfb` | 3.8G | 287.8s | 154.5s | 110.4s |
| TP8 | `canary-1b-v2-tp8.trtfb` | 4.3G | 466.6s | 329.7s | 108.0s |

Artifacts are under `/tmp/trtmc-canary-bench/`.

## Benchmark Method

Final numbers are from a rerun after profiling: 1 warmup + 3 measured runs per config.

`wall_s` is process wall time and includes process start, bundle read/load, and one transcription. The steadier latency metric is `engine_ms_max_rank`: max rank sum of `[trtmc.engine_timing]` encoder + decoder. For SD this is rank 0 only.

All runs produced:
`<|notimestamp|><|nodiarize|> Hello there, how is the weather today?<|endoftext|>`

## Final Results

| Config | GPUs | Mean wall_s | Mean engine_ms_max_rank | Median | Min | Max | Relative to SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| SD | 1 | 18.23 | 49.3 | 48.9 | 47.1 | 51.8 | 1.00x |
| TP2 | 2 | 18.49 | 788.7 | 1059.7 | 107.4 | 1199.0 | 0.06x |
| TP4 | 4 | 22.06 | 958.8 | 965.4 | 878.1 | 1032.7 | 0.05x |
| TP8 | 8 | 33.32 | 2225.0 | 2654.0 | 1254.1 | 2767.0 | 0.02x |

Result: no TP config beats SD. Even the best TP2 measured run, `107.4ms`, is slower than the worst SD run, `51.8ms`.

## Profiling Notes

Nsight Systems profiles were captured for SD, TP2, TP4, and TP8 under `/tmp/trtmc-canary-bench/nsys/`.

| Profile | Max engine timing in profiled run | NCCL all-reduce NVTX total | NCCL all-reduce kernel instances | NCCL kernel GPU total |
|---|---:|---:|---:|---:|
| SD | 79.0ms | 0.0ms | 0 | 0.0ms |
| TP2 | 1583.5ms | 1602.2ms | 1008 | 33.1ms |
| TP4 | 883.7ms | 2521.2ms | 2016 | 48.3ms |
| TP8 | 688.2ms | 3428.3ms | 4032 | 153.1ms |

TP profiles show the top CUDA kernel bucket is `ncclDevKernel_AllReduce_Sum_f32_RING_LL`. The aggregate all-reduce count scales with rank count: TP2/TP4/TP8 show 1008/2016/4032 all-reduce kernels across ranks. This workload is a short single transcription, so decoder TP introduces many small collectives while the encoder remains replicated per rank.

## Conclusion

For Canary on this B200 x8 host and this short transcription workload, SD is the latency winner. TP2, TP4, and TP8 all regress steady-state engine latency because NCCL/all-reduce overhead dominates the decoder shard speedup; TP8 is the slowest in the final benchmark.
