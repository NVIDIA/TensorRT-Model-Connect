# Runtime-Allocated KV Cache + Split Engine Prototype

Date: 2026-07-22 (America/Los_Angeles)

Branch: `dynamic-sequence-lengths` tracking `origin/main`

Models:

- `Qwen/Qwen3-0.6B`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Review Summary

This prototype implements the requested model-only build flow and moves the
user-visible KV decision to runtime:

```bash
# Build: model only. Output name, dynamic KV, engine cap, and split layout are automatic.
trtmc build Qwen/Qwen3-0.6B
trtmc build TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Runtime: automatic 90% of post-engine free memory.
trtmc run qwen3-0.6b.trtfb --prompt "..." --max-new-tokens 32

# Runtime: percentage or explicit bytes.
trtmc run qwen3-0.6b.trtfb --kv-cache-memory 80% --prompt "..." --max-new-tokens 32
trtmc run qwen3-0.6b.trtfb --kv-cache-memory 8GiB --prompt "..." --max-new-tokens 32

# Runtime: optionally impose a logical prompt + output limit.
trtmc run qwen3-0.6b.trtfb --max-sequence-length 4K \
  --prompt "..." --max-new-tokens 32
```

The same split bundle is reused for every runtime choice. No engine rebuild is
needed when the KV budget changes.

This is a successful single-pipeline prototype, not yet a vLLM-style paged,
multi-request KV allocator. TensorRT still has a finite hidden engine
capability, while the physical KV allocation inside that capability is chosen
at runtime. Dynamic bundles are intentionally rejected by the current
multi-lane `PipelinePool`: pool-level budget partitioning is a next-phase
requirement, and silently allocating the requested budget once per lane would
be incorrect.

## User Experience

### Build

The user supplies only a model:

```bash
trtmc build Qwen/Qwen3-0.6B
```

The CLI:

1. derives `qwen3-0.6b.trtfb`;
2. resolves the Qwen family;
3. sees the family-owned `dynamic_kv_split_decoder` capability;
4. selects the native dynamic-KV builder instead of an optimized provider that
   cannot satisfy this contract;
5. derives the engine cap from model metadata and the prototype-certified cap;
6. builds separate prefill and decode plans.

The build-time KV/profile flags are no longer shown in ordinary help. They
remain parsable only for compatibility and developer bisects.

### Runtime

Supported controls are:

```text
--kv-cache-memory auto
--kv-cache-memory 80%
--kv-cache-memory 8GiB
--max-sequence-length auto
--max-sequence-length 32K
```

`--kv-cache-size` remains an alias for explicit-byte compatibility.

When no memory option is provided, the policy is `auto`, which budgets 90% of
free device memory observed after both TensorRT plans and their execution
contexts have loaded. The resolver also keeps a model-specific safety reserve
(`64 MiB + one KV row` in this prototype), and explicit byte requests are
rejected if they exceed safely usable post-load memory.

Percentage and logical-max controls are accepted only by bundles declaring the
Qwen/Llama dynamic-KV runtime contract. Passing them to a static or unsupported
bundle fails with a clear error instead of being ignored. Existing explicit
build-profile flags remain a hidden compatibility path and continue to produce
the legacy static behavior.

### Allocation Point

```text
load decode engine + profile contexts
            |
load split prefill engine + context
            |
synchronize CUDA
            |
cudaMemGetInfo()
            |
resolve auto / percentage / bytes / max-sequence-length
            |
allocate runtime-sized per-layer K/V state
            |
bind the same K/V pointers into prefill and decode contexts
```

The TensorRT module is told at construction time that `cache_k_*` and
`cache_v_*` are caller-owned inputs. It retains dtype/shape/profile metadata
but does not allocate profile-MAX buffers for those inputs. The Qwen/Llama
runtime state then allocates the selected capacity once and binds it to both
engines.

## Why Split Acceleration Is Preserved

Dynamic KV and split prefill/decode are independent:

```text
runtime-owned contiguous KV allocation
        |
        +--> prefill engine
        |      dynamic cache input shape
        |      one batched prompt execution
        |      only logits copied to host
        |      present K/V kept on device and copied D2D into shared KV
        |
        +--> decode engine
               fixed Sq=1
               smallest profile covering active KV rows
               same shared KV pointers rebound after prefill
```

The old loss of split acceleration was caused by software gates:

- the generic builder rejected dynamic KV when selecting split plans;
- the Qwen/Llama builders rejected a dynamic-KV prefill role;
- the runtime lacked dynamic input metadata, covering-profile selection, and
  construction-time deferred input allocation.

The prototype removes those gates only for families that explicitly declare
`dynamic_kv_split_decoder`.

The TensorRT module now supports selective host materialization. Split prefill
executes the complete graph but downloads only logits; it does not copy every
layer's profile-sized `present_k/v` output to host before the D2D KV write.

## Hidden Engine Capability

TensorRT dynamic shapes still require a finite profile MAX. The user no longer
chooses that value.

| Model | Model context limit | Automatic engine cap |
|---|---:|---:|
| Qwen3-0.6B | 40,960 | 4,096 |
| TinyLlama-1.1B | 2,048 | 2,048 |

The prototype cap is `min(model context limit, 4096)`.

The first Qwen build exposed a build-policy issue: FP32 prefill at a 4K
envelope needed about 3.8 GiB for a TensorRT tactic, while the family builder
allowed only 1 GiB. Dynamic-KV Qwen/Llama builders now use an internal 8 GiB
build-only workspace ceiling. This is a TensorRT build allowance, not an 8 GiB
runtime KV reservation.

## Real Build Results

Both commands below were run without output, precision, KV, sequence, profile,
or layout flags:

```bash
trtmc build Qwen/Qwen3-0.6B
trtmc build TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Environment: TensorRT 11.2.0.113, NVIDIA GB300, default FP32 precision.

| Model | Prefill plan | Decode plan | Build time | Bundle bytes |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 2,882.2 MB | 2,913.2 MB | 463.5 s | 6,092,828,479 |
| TinyLlama-1.1B | 4,204.9 MB | 4,218.9 MB | 250.8 s | 8,835,354,215 |

Bundle inspection confirms both `engine_plan` and `prefill_engine_plan`.

## Real Runtime Results

All rows below reuse the bundle built once above.

### Qwen3-0.6B

KV row size: 229,376 bytes.

| Runtime policy | Post-load free | Policy budget | Resolved rows | Physical KV | Split prefill |
|---|---:|---:|---:|---:|---|
| default `auto` | 262.55 GiB | 236.29 GiB (90%) | 4,096, engine-capped | 896.00 MiB | 8 tokens, one call |
| `28MiB` + max seq 128 | 262.54 GiB | 28.00 MiB | 128 | 28.00 MiB | 8 tokens, one call |
| `0.01%` | 262.55 GiB | 26.88 MiB | 122 | 26.69 MiB | 8 tokens, one call |

The 122-row percentage result is the physical allocation capacity, not a
build-time profile bucket. Decode selects the smallest profile covering the
rows active for the current request; the earlier 8-token run therefore used
the 32-row context rather than claiming a 122-row execution shape.

### TinyLlama-1.1B

KV row size: 45,056 bytes.

| Runtime policy | Post-load free | Policy budget | Resolved rows | Physical KV | Split prefill |
|---|---:|---:|---:|---:|---|
| default `auto` | 265.22 GiB | 238.69 GiB (90%) | 2,048, engine-capped | 88.00 MiB | 9 tokens, one call |
| `5.5MiB` + max seq 128 | 265.22 GiB | 5.50 MiB | 128 | 5.50 MiB | 9 tokens, one call |
| `0.002%` | 265.21 GiB | 5.43 MiB | 126 | 5.41 MiB | 9 tokens, one call |

Every positive run logged:

```text
[trtmc] Batched prefill (prefill engine): N tokens in one call
```

### Long-Prompt Profile-Boundary Proof

Both bundles were then run with the same runtime allocation capped at 128 rows
and a prompt long enough to cross the 32- and 64-row decode profiles:

| Model | Tokenized prompt | Prefill launches | First decode trace | Result |
|---|---:|---:|---|---|
| Qwen3-0.6B | 100 | 1 | `decoder_idx=2, rows_before=128` | passed |
| TinyLlama-1.1B | 102 | 1 | `decoder_idx=2, rows_before=128` | passed |

The engine timing labels identify the separate
`prefill_engine_plan:prefill` and `engine_plan:decode` plans. Each trace
contains four decode steps, all on the 128-row covering profile, proving that
the runtime rebound the caller-owned KV state after one batched split-prefill
call rather than falling back to token-by-token prefill.

Trace receipts:

- `/tmp/trtmc-dynamic-memory-prototype-20260722/runtime-evidence/qwen-100-token-profile-trace.jsonl`
  - SHA256: `8647e777b593aab5f11082085a769dfa0a7bec65449f40dbe51761eae4d7c145`
- `/tmp/trtmc-dynamic-memory-prototype-20260722/runtime-evidence/tinyllama-102-token-profile-trace.jsonl`
  - SHA256: `516fc57f9383b777cf982b986e7d93464278c12904f209275a5e370930d737b3`

### Runtime Guards

The runtime rejects a request before prefill when prompt plus requested output
does not fit:

```text
Error: QwenTextGenerationPipeline: prompt length 8 plus max_new_tokens 4
exceeds runtime max sequence length 10
```

It also distinguishes logical runtime choice from engine capability:

```text
Error: Requested max sequence length 32768
exceeds bundle engine capability 4096
```

## Implementation Areas

- model-only build policy and derived output;
- Qwen/Llama family capability declaration;
- dynamic cache shapes in split prefill and multi-profile decode builders;
- runtime CLI parsing and `LoadOptions` propagation;
- pure memory-budget resolver;
- construction-time deferred TensorRT inputs;
- post-engine `cudaMemGetInfo` sizing;
- shared KV pointer/shape binding across split contexts;
- selective logits-only D2H for split prefill;
- overflow-safe prompt + output admission;
- separate logical sequence and physical compacting-cache limits;
- relative-bundle effective-config handling.

## Validation

- Full C++ runtime plus the Qwen/Llama plugins compiled successfully.
- Focused C++ tests: 7/7 passed:
  - CLI parsing;
  - config/effective-config path;
  - pure KV budget/admission;
  - real TensorRT deferred-input allocation, selective-output execution, and
    device-output retention on GPU;
  - runtime C ABI propagation and unsupported-policy/pool rejection;
  - optimized-runtime-host rejection;
  - compacting-cache logical-limit behavior.
- Build CLI/routing regression set: 241 passed.
- Dynamic split builder/profile tests: 5 passed on TensorRT/GPU.
- Two real model builds: passed.
- Six positive real model runtime cases: passed.
- Two long-prompt, 128-row profile runtime cases: passed.
- Two real runtime rejection cases: passed.
- `git diff --check`: passed.

Artifacts:

- `/tmp/trtmc-dynamic-memory-prototype-20260722/qwen/qwen3-0.6b.trtfb`
  - bytes: `6092828479`
  - SHA256: `120557f08b7f7f4a5e4811b0f7956ef76a61494c9387e072260ad1754744cf58`
- `/tmp/trtmc-dynamic-memory-prototype-20260722/tinyllama/tinyllama-1.1b-chat-v1.0.trtfb`
  - bytes: `8835354215`
  - SHA256: `fbc633b314f422f730ffd2bb030706943a6cbde353d0e3868d4e32eff27cadbf`

## Known Prototype Limits

1. **Finite engine cap remains.** Runtime allocation is dynamic only inside the
   hidden TensorRT profile envelope. Supporting Qwen's full 40K context needs
   chunked prefill and a different mask/profile design.
2. **This is one contiguous pool per pipeline.** There is no paged/block
   allocator, multi-request scheduler, eviction, or prefix-cache sharing yet.
   Dynamic bundles are rejected by `PipelinePool` until one pool-level budget
   can be partitioned across lanes.
3. **Prefill staging is still profile-MAX.** Deferred allocation removes hidden
   `cache_k/cache_v` input buffers, but TensorRT prefill outputs, attention
   masks, and execution-context workspace still scale with the build envelope.
   Selective output prevents `present_k/v` D2H traffic, but the device outputs
   are still copied D2D into the shared KV allocation; direct output binding is
   a further optimization.
4. **Split duplicates plan weights.** The measured bundles contain separate
   multi-gigabyte prefill and decode plans.
5. **All decode profiles are initially loaded before budgeting.** Small runtime
   allocations later discard unneeded contexts, so the memory measurement is
   conservative but startup can be optimized.
6. **The caller controls policy, not a raw CUDA pointer.** The KV allocation is
   external to TensorRT and owned by the model runtime state. A future public
   allocator API can accept a caller-provided buffer/allocator.
7. **Prototype family scope is Qwen and Llama.** Other model plugins do not yet
   implement percentage/max-sequence semantics and explicitly reject those
   controls.
8. **Default precision remains FP32.** The no-flag UX works, but bundle size and
   load cost show that an automatic LLM precision policy should be a separate
   product decision.
9. **Binary components require lockstep deployment.** Public structs and the
   TensorRT module interface were tail-extended for this prototype, but there
   is no size/version handshake yet. The core, TensorRT backend, Qwen/Llama
   model plugins, and clients must be rebuilt and deployed from the same source
   revision; mixed old/new DSOs are not supported.
10. **TriAttention has two capacities.** `--max-sequence-length` is a logical
    request limit for compacting KV, while the runtime memory budget controls
    its smaller physical row allocation. Dense KV uses the same value for both.

## Recommended Next Phase

If this user contract is approved, the next implementation should separate
three independent limits:

```text
engine context capability
prefill chunk maximum
runtime KV pool capacity
```

Then add:

1. chunked prefill so the engine cap can grow without quadratic profile-MAX
   staging;
2. a public allocator/buffer descriptor with pointer, capacity, device, stream,
   lifetime, and callback contracts;
3. a paged KV pool and request admission layer for concurrency;
4. per-device reserve and multi-lane/TP coordination;
5. automatic precision and certified-cap policy stored in family metadata.
