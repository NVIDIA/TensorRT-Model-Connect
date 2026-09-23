# Separate prefill and decode profiles

Two specialized profiles with 1024-token prefill reduced MC OOTB EAGLE3 chain
latency by **16.1%** versus a fresh single-profile control on September 21.
Keeping 64-token chunks gave only a 0.8% difference, within observed variation.
Most of the gain came from larger prefill chunks.

This is a historical experiment before the host-allocation and device-residency
optimizations. See [current performance](SPECULATIVE_PERFORMANCE.md) and the
[current graph profile](SPECULATIVE_GRAPH_PROFILE.md) for subsequent results.

## Contract and measured results

The builder supports `--execution-profiles split`. Each target/draft engine
has two persistent execution contexts sharing weights, stream and KV storage.
Profile 0 serves prefill and profile 1 serves decode. The default remains
`single`; both forms use the same linear state ABI. Bounds and ordering are
specified in the [compiler/runtime contract](SPECULATIVE_DECODING.md#separate-prefill-and-decode-profiles).

| Implementation | Profiles | Prefill chunk | Chain median, ms | P10–P90, ms |
|---|---|---:|---:|---:|
| MC OOTB | Single | 64 | 622.16 | 609.72–633.50 |
| MC OOTB | Split | 64 | 616.95 | 595.10–618.36 |
| MC OOTB | Split | 1024 | **521.98** | 511.87–530.80 |

| MC chain variant | Prefill including draft, ms | Decode, ms |
|---|---:|---:|
| Single / 64 | 175.89 | 446.25 |
| Split / 64 | 166.12 | 450.81 |
| Split / 1024 | **73.43** | 445.46 |

MC prefill falls by 58.3%; decode changes by less than 1 ms. Phase medians need
not sum to the request median. These are MC host phase timers, including
draft prefill in the prefill phase.

GB100/SM100, FP16/B1/TP1, IST=1024, 101 output IDs, depth-four chain, capacity
2048, CUDA graphs off. Each mode has three warm-ups and ten timed requests.
All OOTB variants use the same runtime and checkpoints; the split plans were
rebuilt. Runtime preparation, host selection and feature transport were unchanged
in this experiment. The runs therefore distinguish larger prefill chunks from
profile specialization, but do not isolate individual compiler tactic changes.

Each OOTB bundle passed AR/chain/tree, request reuse and short-prompt checks.
All full-prompt AR/chain/tree streams matched across 101 IDs. Chain acceptance
remained at 26 rounds and tree at 25. The Python contract and C++ policy
tests passed.

## Reproduction and limits

```bash
python3 -m tensorrt_model_connect llama build-speculative \
  --model-dir /path/to/target --draft-dir /path/to/draft \
  --max-sequence-length 2048 --max-query 64 --draft-depth 4 \
  --execution-profiles split --prefill-query 1024 \
  --output llama-eagle3-split1024.bundle
llama_speculative_benchmark llama-eagle3-split1024.bundle input_ids.json 101 3 10 output.json
```

Use `--prefill-query 64` for the profile-only comparison. These measurements
use host selection; GPU selection is a separate later capability.

TensorRT 11.1.0.106, CUDA 13.3.73, driver 595.58.03. Checkpoints and fixture are
pinned in the [performance report](SPECULATIVE_PERFORMANCE.md). Historical raw
JSON, telemetry, scripts and metadata remain in
`/home/trentl/Working/specdecode-profiles/`.

Clocks were unlocked. This is one repetitive, high-acceptance fixture, not a
general numerical or long-context qualification. Compare controls within this
experiment, not absolute times from different allocations.
