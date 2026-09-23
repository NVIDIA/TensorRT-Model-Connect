# Reducing speculative runtime host overhead

This records the host-allocation fix that preceded the
[device-resident runtime](SPECULATIVE_RESIDENT_RUNTIME.md). In the original
runtime, full logits and features crossed to the host after each model call.
A historical trace measured roughly 270 ms with no GPU activity per request.

## Finite-check allocation fix

`argmax` formerly passed a string literal to
`require(bool, const std::string&)` for every value. A temporary string was
constructed before the condition was checked. The fixture scanned 19.6 million
values per request, and the 17-character error text allocated heap storage in
the measured binary.

The runtime now constructs the message only when a non-finite value is found.
It retains the exception, complete finite check and first-index tie behavior.
No model engine rebuild or state ABI change was needed. Regression cases cover
NaN, both infinities, tied maxima, signed zero, negative values and empty ranges.

The controlled diagnostic reduced finite-scan time from about 200 to 10 ms and
OOTB request latency from 512 to 337 ms. Those diagnostic timings differ from
the subsequent production measurements:

| Implementation | AR median, ms | Chain median, ms | Tree median, ms |
|---|---:|---:|---:|
| MC OOTB, allocation fix | 782.71 | 332.56 | 362.35 |

September 22, GB100/SM100, FP16/B1/TP1, IST=1024, 101 generated IDs, split1024
profiles, CUDA graphs off, three warm-ups and ten timed requests per mode.
Loading is excluded; prefill and reset are included. Clocks were unlocked.
The OOTB build, C++ policy tests, formatting and AR/chain/tree/reset checks
passed. Full-prompt AR, chain and tree outputs matched exactly.

## Subsequent optimization

The allocation fix left CPU selection, feature preparation, transfers and
submission gaps. Approximately 116 MB D2H and 58 MB H2D were transferred per
chain request. Actual GPU copy time was about 10.5 ms; pageable-copy API time
also includes waiting for GPU work and must not be added to it.

The [resident runtime](SPECULATIVE_RESIDENT_RUNTIME.md) now keeps features on
device, provides optional TensorRT GPU selection and reuses staging/storage.
Its [graph profile](SPECULATIVE_GRAPH_PROFILE.md) records the remaining costs.
Buffer lifetimes, cross-stream ordering, first-index ties and per-row finite
checks remain explicit parts of the contract. CUDA graph replay is future work.

Historical raw results, scripts and source/binary hashes are retained in
`/home/trentl/Working/specdecode-host-overhead/`. These timings describe one
high-acceptance fixture, not a general workload suite.
