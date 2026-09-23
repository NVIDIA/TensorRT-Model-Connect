# Llama 3.1 / EAGLE3 request performance

The September 22 remeasurement compares autoregressive execution with EAGLE3
chain decoding on the same GB100. MC uses TensorRT graph primitives, separate
prefill/decode profiles and the [device-resident runtime](SPECULATIVE_RESIDENT_RUNTIME.md).

| Implementation | Autoregressive median, ms | EAGLE3 chain median, ms | Speedup |
|---|---:|---:|---:|
| MC OOTB | 810.52 | **299.90** | **2.703×** |

Each cell contains 30 timed requests across three blocks and nine warm-ups.
All 78 MC requests across both modes produced the same 101 token IDs.
The EAGLE3 chain uses 26 target-verification rounds.

## Measurement scope

FP16 Llama-3.1-8B-Instruct plus EAGLE3, batch one, one GPU, IST=1024, greedy
selection, depth four and width one. The output is 101 IDs: the prefill
prediction plus 100 subsequent tokens. CUDA graphs are disabled. MC uses
`--execution-profiles split --prefill-query 1024 --greedy-selection device_v1`.

Host wall time includes reset, prefill, decoding, transfers, acceptance and
output conversion. Engine loading and warm-up are excluded. AR disables
speculation using the same target plan, whose target-feature outputs remain
part of the graph; this is not a separate AR-only engine build. Draft inference
is skipped. Each request starts with fresh logical state, without prefix reuse.

AR/chain order alternates within each block. One benchmark runs at a
time. Clocks and power limits were not changed. This worker differs from
earlier measurements; comparisons below use only this run's data.

| Implementation/mode | P10–P90, ms | Min–max, ms |
|---|---:|---:|
| MC OOTB AR | 787.89–818.70 | 786.84–821.33 |
| MC OOTB EAGLE3 | 291.94–302.92 | 290.00–304.20 |

MC's three block speedups were 2.697×, 2.702× and 2.704×. The fixture repeats
a deterministic paragraph and has high acceptance. These are not diverse-prompt
or long-context throughput results; see the
[validation scope](SPECULATIVE_DECODING.md#recorded-native-validation-september-22).

## Reproduction

Build the bundle as documented in [speculative decoding](SPECULATIVE_DECODING.md),
adding the split-profile and GPU-selection flags above. With
`TRTMC_BUILD_TESTS=ON`, build and run:

```bash
cmake --build build --target llama_speculative_benchmark
build/families/llama/llama_speculative_benchmark \
  llama-eagle3.bundle input_ids.json 101 3 10 mc.json
```

That harness rotates AR, chain and tree. The remeasurement used an external
copy restricted to AR/chain by changing only the mode list and loop size. It
ran three blocks for MC.

Measured MC revision: `f0b03c45cd46a65c31ea6c593d4e789e48e5d26b`.
TensorRT 11.1.0.106, CUDA 13.3, driver 595.58.03, GB100/SM100.
Target revision: `0e9e39f249a16976918f6564b8830bc894c89659` of
`meta-llama/Llama-3.1-8B-Instruct`; draft revision:
`ada412b672e293d682423de84a095447bf38a637` of
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`.
Fixture SHA256: `1e265729ff51f6d4874c70a32f1abde42b2feddc0b24623400c1c998937d1c4c`.

Local evidence is retained in `/home/trentl/Working/specdecode-eagle3-remeasure/`:
scripts, source/binary hashes, library resolution, per-request JSON and telemetry.
These measurements precede the review cleanup; they are not a new performance
qualification of the cleanup commit. The [graph profile](SPECULATIVE_GRAPH_PROFILE.md)
uses separate traced requests to explain GPU execution and host gaps.
