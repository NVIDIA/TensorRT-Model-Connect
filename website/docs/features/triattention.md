---
title: TriAttention
---

import Diagram from '@site/src/components/Diagram';

TriAttention is an experimental native decoder KV-cache policy. It scores
older cache rows, keeps a protected and recent subset, and compacts the cache
when the logical budget is exceeded. TensorRT continues to execute the same
decoder engine after compaction; TriAttention changes runtime cache state, not
the public generation API.

:::warning Builder acceptance is not runtime support

The generic build CLI accepts TriAttention options for a family that advertises
the native `decoder_kv` capability. That capability currently describes a
builder/engine shape, not a model-owned TriAttention runtime contract. A build
can therefore contain TriAttention config and stats that its selected runtime
never consumes.

The current model-owned implementation is the Qwen native
`qwen_decoder_kv_cache` runtime. In particular, Qwen-VL advertises
`decoder_kv`, so the builder accepts and embeds these options, but its C++
runtime constructs the ordinary `QwenVlKvCache` and does not read or compact
TriAttention state. Do not use a successful Qwen-VL build as evidence that the
feature is active.

Use this runbook only with a runtime whose model-owned implementation reads the
TriAttention config/stats section and reports TriAttention cache behavior.
Then use calibration stats produced for the exact model and compare the
resulting bundle with a dense bundle on the intended long-context workload.

:::

## Build a TriAttention bundle

For the currently implemented Qwen native path, TriAttention activation
requires a matching calibration-stats file:

```bash
trtmc build /path/to/qwen-model \
  --triattention-stats /path/to/triattention-stats.pt \
  --triattention-kv-budget 6144 \
  --triattention-divide-length 1024 \
  --triattention-recent-window 128 \
  -o /path/to/model-triattention.trtfb
```

The stats must match the Qwen model's attention dimensions and rotary-position
contract. The builder converts them into the
`triattention_stats.json` bundle section and writes the selected policy into
`config.json`. Those sections prove bundle assembly only; confirm that the
selected model runtime consumes them before claiming activation.

The main build controls are:

| Option | Meaning |
| --- | --- |
| `--triattention-stats FILE` | Embed matching calibration statistics and enable TriAttention. |
| `--triattention-kv-budget N` | Logical number of cache rows retained by the policy. It must not exceed the engine's cache capacity. |
| `--triattention-divide-length N` | Trigger compaction when the cache reaches the budget plus this interval. It must be positive. |
| `--triattention-recent-window N` | Always retain this many recent rows. It must be nonnegative. |
| `--triattention-score-aggregation mean|max` | Combine offset scores with the selected aggregation. |
| `--triattention-no-count-prompt-tokens` | Exclude prompt tokens from logical budget accounting. |
| `--triattention-no-protect-prefill` | Allow prompt rows to be removed during compaction. |
| `--triattention-disable-mlr` | Disable the magnitude-based additive score term. |
| `--triattention-disable-trig` | Disable the trigonometric score term. |

Enabling TriAttention also enables the dynamic-KV engine path. A
tensor-parallel decoder build currently rejects both dynamic KV and
TriAttention.

## Physical capacity versus logical budget

The engine cache capacity, runtime allocation, and TriAttention budget are
different limits:

- The engine is built with an upper bound for legal TensorRT shapes.
- `--kv-cache-size` can lower the actual runtime allocation for a compatible
  dynamic-KV bundle.
- `triattention.kv_budget` controls how many logical cache rows the compaction
  policy tries to retain.

The runtime budget cannot increase the engine's build-time shape limit.
Allocating too little physical cache for the prompt and compaction interval can
also fail or damage the intended evaluation setup. Keep enough physical
capacity to admit the prompt before interpreting quality results.

## Runtime configuration

Core policy values are registered in the `triattention` config namespace.
Runtime `--set` values can tune a compatible bundle without rebuilding:

```bash
trtmc run /path/to/model-triattention.trtfb \
  --prompt "Explain why reproducibility matters." \
  --max-new-tokens 128 \
  --kv-cache-size 12GiB \
  --set triattention.kv_budget=4096 \
  --set triattention.divide_length=512 \
  --set triattention.recent_window=128
```

The schema also exposes session-only diagnostics such as
`triattention.debug`, `triattention.profile`, and dump controls. These are
debugging surfaces, not stable output or performance contracts. Inspect the
live config catalog and CLI help before using them.

## Runtime design

<Diagram
  src="/img/diagrams/features/triattention-runtime-sequence.svg"
  alt="TriAttention runtime sequence from GPU candidate scoring through host normalization and top-k selection, GPU compaction, and conditional dynamic cache rebinding"
  caption="On the default GPU path, kernels score and copy rows while Qwen-owned host logic normalizes, aggregates, and selects keep indices; a later decode step rebinds cache tensors only when its row bucket changes."
  sequence
/>

The current Qwen model-owned runtime owns position tracking, cache selection,
GPU gather, and rebinding. The generic config registry owns validation and
layered value resolution. Another family's ordinary dynamic-KV cache does not
gain those behaviors merely because its builder advertises `decoder_kv`.

## Validation

At minimum, retain:

1. exact model and stats provenance;
2. dense and TriAttention bundle identities;
3. identical prompts, sampling settings, token budgets, and hardware;
4. output or answer parity on the intended workload;
5. throughput or latency artifacts from synchronized runs; and
6. cache configuration, runtime evidence that the TriAttention state was
   created, compaction counts, and any runtime overrides.

Host-only unit tests cover stats export, schema parity, config validation, and
bundle assembly. They do not prove long-context model quality or a speedup.
The dated implementation investigation is retained in the
[TriAttention Native C++ Worklog](../context/triattention-native-cpp-worklog.md);
its temporary paths and benchmark numbers are historical evidence, not a
current runbook.
