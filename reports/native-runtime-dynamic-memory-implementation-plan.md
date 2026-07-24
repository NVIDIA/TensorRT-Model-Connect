# Native Runtime Dynamic Sequence Length and KV Memory Implementation Plan

Date: 2026-07-24
Status: Implementation candidate; promotion remains gated by source-bound qualification and nightlies
Scope: Model Connect native runtime only

## 中文概要

这份方案把“模型能力上限”和“本次运行实际分配多少显存”彻底拆开。
用户 build 时只提供模型：

```bash
trtmc build Qwen/Qwen3-0.6B
trtmc build TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

产物仍然是每个模型一个 bundle。用户不需要、也不应该在 build 时选择
KV cache、context length 或 TensorRT profile。bundle 内部记录模型语义上限
`M`、prefill 分块上限 `C`、每 token KV 字节数 `B` 和内部 history buckets；
这些是可执行 contract，不是用户调参。

runtime 才决定本次进程的物理 KV capacity `R`：

- 默认 `auto`：固定 engine 加载后，取安全可用显存的 90%，再受模型上限约束；
- 百分比：例如 `--kv-cache-memory 80%`；
- 具体值：例如 `--kv-cache-memory 8GiB`；
- 可另加 `--max-sequence-length 32K`，它是请求 admission 上限 `U`，不是
  build 参数，也不会改变 bundle。

目标实现保留独立的 prefill/decode engine，因此不会丢失 split-engine tactics。
两个 engine 绑定同一个 runtime-owned contiguous KV allocation；TensorRT
只读 history，当前 chunk 的 K/V 使用精确 `Sq` staging，engine 完成后只把
新行 D2D commit 到 cache。不会复制完整 history，不会把 KV 搬回 host，也
不会创建 `O(L^2)` dense mask 或 score tensor。prefill 使用 chunking，所以
Qwen bundle 必须能够表达并执行模型的 40,960 context，而不是被某个
2K/4K profile 伪装成模型能力上限。

显存分配时序如下：

1. 读取 bundle contract，并验证 exact model/revision/config；
2. 在任何 engine deserialize 或 KV allocation 之前，验证当前真实 target；
   prototype 至少要求 `sm103 + TensorRT 11.2.0.113`，release 还必须匹配
   Section 10 固定下来的 CUDA/cuDNN/Frontend/NVRTC/driver tuple；
3. 反序列化 split engines，使用 `kUSER_MANAGED` planning contexts 查询
   actual-shape context memory；此时不挂载 context device memory，也不分配 KV；
4. 读取 post-load free memory，解析 `auto/%/bytes/U`，求出 `R`；
5. 只分配一次 `R * B` KV slab、精确 staging，以及所有保留 contexts 共用的
   一个 device-memory block；
6. pipeline 生命周期结束时统一释放。后续请求复用同一 allocation。

本阶段严格只开放两个 qualified tuple：Qwen3-0.6B 用于 32K/40K
long-context 证明，TinyLlama 用于跨 family 证明。其他 Qwen/Llama
checkpoint 不会自动继承资格。EdgeLLM adapter、runtime DSO 和 tests 均不在
修改范围内。1M context、多请求 continuous batching、paged KV、prefix cache
和 offload 属于后续 device-level pool 阶段；本阶段不拿 contiguous prototype
冒充这些能力。

review 时不要只看“能运行”。Section 10 的 producer 必须亲自执行 fresh build
和真实 benchmark worker，记录 build 前后完整 source-state、bundle/request/
binary SHA，并独立反序列化 engine 核对 plan bytes、resident weights、
weight-copy count 和 streaming 状态。缺字段、复用旧 bundle、source 漂移或
手工拼接旧 JSON 都会 fail closed。

## 1. Decision

Implement runtime-sized KV memory first for exactly two native-runtime models:

| Model | Native family | Model context limit | Purpose |
|---|---|---:|---|
| `Qwen/Qwen3-0.6B` | `qwen` | 40,960 | Prove that a no-flag bundle can cross 32K and reach the model's full context without a hidden 4K engine cap. |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `llama` | 2,048 | Prove that the implementation is a shared native-runtime contract rather than a Qwen-only special case. |

Qwen is the long-context qualification model. TinyLlama is the second-family
control model; it is not evidence for 32K or 1M support.

Dynamic-memory routing is enabled only for a qualified tuple:

```text
canonical model ID
+ resolved model revision
+ graph-relevant config fingerprint
+ target TensorRT/platform qualification
```

The Qwen/Llama family capability means the implementation is available; it
does not qualify every checkpoint in those families. A different Qwen or
Llama checkpoint retains its previous routing and behavior. Once one of the
two exact qualified model identities is recognized, however, a target,
revision, or config mismatch fails explicitly rather than silently producing
a bundle with a different runtime contract. There is no wildcard family
enablement in this milestone.

The first deliverable is a product-quality, single-pipeline beta using a
runtime-sized contiguous KV allocation, read-only segmented fused attention,
and a runtime commit of current K/V rows only. A device-level paged KV pool is
the next delivery phase for concurrency, prefix reuse, and 1M-class serving.

## 2. Hard scope boundary

This work must not modify, wrap, or depend on the EdgeLLM adapter.

The following trees are explicitly out of scope:

```text
python/tensorrt_model_connect/families/qwen/edge_llm_adapter/**
src/runtime/models/qwen/edge_llm_adapter/**
tests/e2e/models/qwen/edge_llm_adapter/**
```

For the two qualified tuples, the model-only build flow must select the native
Qwen or Llama implementation.
An EdgeLLM-qualified bundle continues to use its existing contract and
behavior unchanged by this work. Runtime dynamic-memory options passed to an
incompatible bundle must fail before provider dispatch; they must not be
silently ignored or translated into EdgeLLM build parameters.

Also out of scope for the initial two-model beta:

- VLM image-token accounting;
- tensor parallel or pipeline parallel;
- multi-request continuous batching;
- prefix-cache sharing;
- host/NVMe KV offload;
- KV quantization;
- changing the repository's default precision policy;
- claiming 128K or 1M qualification.

## 3. Required user experience

### 3.1 Build

The user supplies only the model:

```bash
trtmc build Qwen/Qwen3-0.6B
trtmc build TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The build command must not require or advertise:

```text
--max-cache-length
--max-sequence-length
--max-input-length
--kv-cache-size
--kv-cache-memory
--decoder-engine-layout
```

For the two selected model-only paths, the builder must:

1. resolve the native family;
2. resolve the canonical ID to the internally qualified, pinned model revision;
3. verify the graph-config fingerprint and target qualification;
4. read the model's semantic context limit from model metadata;
5. choose an internal prefill chunk policy;
6. build separately optimized prefill and decode execution roles;
7. write the runtime-memory contract into the bundle;
8. derive the default output filename.

The revision is an internal qualification choice, not a new user flag. A local
snapshot or embedding-API revision only enters this dynamic path when it
matches the same qualification record. Once the canonical model ID or its HF
snapshot identity is recognized, an explicit revision mismatch is an error; it
must not silently fall back to the legacy build. Unknown model IDs and merely
similar names remain unqualified and may follow the ordinary builder path.

For a recognized model, target qualification is also fail-closed against the
native runtime-KV plugin's independently detected complete live tuple:
`sm`, exact TensorRT build, CUDA runtime, cuDNN backend, cuDNN Frontend
revision, NVRTC, and driver. Failure to load/probe that evidence or a mismatch
in any field rejects the qualified build. Unknown models return before loading
the plugin, so this gate does not add a CUDA/plugin side effect to the generic
builder path.

The public build result must not contain a product-visible 4,096-token
capability. Developer-only overrides may exist for builder bisects, but they
must not alter the supported user contract or appear in ordinary CLI help.

### 3.2 Runtime

The same bundle must support all memory policies without rebuild:

```bash
# Default: automatic runtime policy.
trtmc run qwen3-0.6b.trtfb --prompt "..." --max-new-tokens 32

# Percentage of safely usable post-load GPU memory.
trtmc run qwen3-0.6b.trtfb \
  --kv-cache-memory 80% \
  --prompt "..." \
  --max-new-tokens 32

# Explicit KV budget ceiling.
trtmc run qwen3-0.6b.trtfb \
  --kv-cache-memory 8GiB \
  --prompt "..." \
  --max-new-tokens 32

# Optional runtime admission policy; never a build request.
trtmc run qwen3-0.6b.trtfb \
  --max-sequence-length 32K \
  --prompt "..." \
  --max-new-tokens 32
```

Default `auto` uses 90% of safely usable GPU memory remaining after fixed
runtime allocations. The model's semantic limit remains the upper bound, so a
small model does not allocate otherwise unused hundreds of GiB just because
the GPU is large.

### 3.3 Inspectability

`trtmc inspect` is static and must report only bundle facts:

```text
model_context_limit
prefill_chunk_limit
runtime_kv_contract_version
kv_layout
kv_dtype
kv_bytes_per_token
active_kv_profile_limits
qualified_model_revision
qualified_config_fingerprint
qualified_runtime_stack
```

It must not initialize CUDA, load an engine, or claim a runtime capacity.

The model-load log and runtime introspection API report values that exist only
after device selection and allocation:

```text
policy
post_load_free_bytes
safety_reserve_bytes
kv_budget_bytes
runtime_kv_capacity_tokens
effective_request_limit
context_device_memory_bytes
kv_allocation_id
capacity_decision_free_bytes
settled_free_bytes
```

Neither surface may use one ambiguous `max_cache_length` value to represent
all of these concepts.

## 4. Terminology and invariants

Use the following symbols in code, logs, metadata, and tests:

| Symbol | Name | Owner | Meaning |
|---|---|---|---|
| `M` | `model_context_limit` | Model metadata | Maximum prompt plus generated-token length supported by model semantics. |
| `C` | `prefill_chunk_limit` | Internal build policy | Maximum number of new prompt tokens processed by one prefill invocation. |
| `R` | `runtime_kv_capacity_tokens` | Runtime memory planner | Physical token rows available in the runtime KV allocation. |
| `H` | `active_history_tokens` | Request scheduler | Tokens already present before the current invocation. |
| `A` | `active_kv_tokens` | TensorRT invocation | Valid cache extent after the current invocation: `A = H + Sq`. |
| `T` | `bound_history_extent` | TensorRT invocation | Read-only history tensor extent bound for this enqueue. Use `T=1` as the cold `H=0` sentinel; otherwise use `T=P=min(ceil_bucket(H), R)`. |
| `U` | `request_context_limit` | User/runtime admission | Optional user cap; `U <= M`. It may exceed `R`, in which case runtime resources remain the effective bound. |
| `B` | `kv_bytes_per_token` | Bundle/runtime validation | Bytes required for one token across all local K and V layers. |
| `P` | history bucket/profile limit | Internal performance policy | `H/T` range optimized by a prefill or decode execution context. |

The v1 bundle field is named `active_kv_profile_limits` for continuity with
the prototype, but its selected segmented-attention semantics are history
`T` buckets. It must never be interpreted as a second model context limit.
Here `ceil_bucket(H)` means the smallest declared history bucket greater than
or equal to `H`. Therefore `H=P` may still bind `T=P`; the next bucket is
selected only when history advances to `P+1`.

Required invariants:

```text
0 < C <= M
0 < R <= M
0 <= H and 1 <= T <= R
(H == 0 and T == 1) or (H > 0 and T >= max(H, 2))
0 < Sq <= C
A = H + Sq <= R
0 < effective_request_limit = min(M, R, U if specified)
prompt_tokens + max_new_tokens <= effective_request_limit
```

For a dense, local KV cache:

```text
B = 2 * local_layer_count * local_kv_head_count
      * head_dim * cache_dtype_bytes
```

The builder records `B`, while the runtime recomputes it from the actual engine
bindings and rejects a mismatched bundle. This prevents stale metadata from
causing an undersized allocation.

## 5. Target architecture

### 5.1 Build-time contract

The bundle records model capability and the execution ABI, not a user memory
choice:

```text
{
  "runtime_memory": {
    "contract_version": 1,
    "qualified_model_id": "Qwen/Qwen3-0.6B",
    "qualified_model_revision": "<resolved-commit>",
    "qualified_config_sha256": "<graph-config-fingerprint>",
    "qualified_target": "sm103",
    "qualified_runtime_stack": {
      "tensorrt": "11.2.0.113",
      "cuda_runtime": "13.3",
      "cudnn_backend": "9.20.0",
      "cudnn_frontend_revision":
        "7b9b711c22b6823e87150213ecd8449260db8610",
      "nvrtc": "13.3",
      "driver": "580.105.08"
    },
    "native_kv_plugin_abi": 2,
    "model_context_limit": 40960,
    "prefill_chunk_limit": 1024,
    "kv_layout": "contiguous_runtime_v1",
    "kv_dtype": "bfloat16",
    "kv_bytes_per_token": 114688,
    "active_kv_profile_limits":
      [128, 256, 512, 1024, 2048, 8192, 32768, 40960],
    "runtime_owned": true
  }
}
```

The exact dtype and byte count come from the actual build. The JSON above is an
illustration, not a hard-coded Qwen policy.

Initial internal qualification candidates:

| Model | `M` | Initial `C` | `B` | Full one-sequence KV at `M` | Active-KV profile candidates |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | 40,960 | 1,024 | 114,688 bytes/token | 4,697,620,480 bytes (4.375 GiB) | 128, 256, 512, 1,024, 2,048, 8,192, 32,768, 40,960 |
| TinyLlama-1.1B | 2,048 | 512 | 22,528 bytes/token | 46,137,344 bytes (44 MiB) | 128, 256, 512, 2,048 |

The exact byte values are recomputed from the resolved model config and engine
bindings; the table records the two pinned qualification candidates rather
than a family-wide constant. The chunk and bucket values are family-owned
performance choices, not public knobs. Qualification may remove a bucket if
it increases engine size/build time without a measurable benefit, but it may
not reduce `M`.

The revised Qwen candidate uses `C=1,024`. A clean `C=2,048` qualification
run exposed a deterministic BF16 near-tie at the 2,048-token boundary, while
the independently built `C/2=1,024` path retained the numerical gates and
selected the HF top token on the same prefix. The 256-token history bucket is
also present for both models so a medium prompt does not jump directly from
`T=128` to `T=512`. Neither change is user-visible or reduces `M`; both must
still pass the complete fresh-build correctness, plan-size, and performance
matrix before the candidate is promoted.

The TensorRT profile envelope uses:

```text
Sq MAX = C
bound KV extent T MAX = M
```

`T MAX = M` is a legality bound for build/tactic selection, not an instruction
to allocate `M` rows at engine load. The graph contains no `M`-sized constant,
dense mask, RoPE table, or internally allocated cache tensor. At runtime the
external buffer has `R` rows and each invocation enforces the Section 4 cold
sentinel (`H=0,T=1`; otherwise `H>0,T>=max(H,2)`), `T<=R`, and
`A=H+Sq<=R`;
the selected fused path binds
`T=1` for cold history or `T=P=min(ceil_bucket(H), R)`, and user-managed
context memory is queried for
that actual bucket shape. `T` changes only tensor metadata and the cuDNN
logical view; it never allocates a second `P`- or `M`-row KV buffer.

If a single broad optimization profile produces unacceptable build time or
tactics, split the internal `T` range across buckets whose union reaches `M`.
The last bucket must still have `MAX = M`; removing that bucket is a
capability regression, not a performance optimization.

### 5.2 Split execution roles

Prefill and decode stay separately optimized:

```text
prefill engine
    Sq in [1, C]
    history length H and bound history extent T are dynamic
    physical capacity R remains runtime-owned
    continuation-prefill buckets P
    fused lower-right causal attention

decode engine
    Sq = 1
    bound-extent buckets P
    fused lower-right causal attention

both engines bind the same runtime-owned KV allocation
```

The first two-model beta deliberately retains physically split prefill and
decode plans. This preserves the existing role-specific tactic selection and
makes dynamic KV ownership independent of engine specialization. Both plans
receive external views into one runtime allocation, so there is no prefill to
decode KV copy.

The runtime loads both plans before measuring post-load free memory. Therefore
their duplicated resident weights are accounted for before the default 90%
KV budget is chosen. Record that weight cost in the memory receipt and compare
it with a one-engine/multi-profile experiment; a later packaging optimization
may use one serialized engine only if it passes the same split-performance
gates. It is not required for dynamic memory and is not exposed as a user
choice.

The user never selects this packaging.

### 5.3 Chunked prefill

Long prompts are processed as bounded chunks:

```text
position = 0
while position < prompt_length:
    chunk_length = min(C, prompt_length - position)
    history_length = position
    total_length = position + chunk_length
    bound_length = (
        1 if history_length == 0
        else min(ceil_bucket(history_length), R)
    )  # history T
    bind the R-capacity cache buffer with history shape
        [bound_length, kv_width]
    set history_length = position
    set position_ids = [position, ..., position + chunk_length - 1]
    run prefill profile with segmented history/current attention
    commit only the current K/V rows at runtime offset position after
        successful enqueue
    position += chunk_length
run decode profile with Sq = 1
```

For a prompt of length `L`, the required prefill launch count is:

```text
ceil(L / C)
```

Only the last chunk's logits need to be materialized for ordinary generation.
No K/V tensor may be copied to host.

The first chunk has `history_length = 0` and active shape
`[chunk_length, ...]`; it does not use a zero-length cache binding. The
external allocation is larger when `R > chunk_length`, but the plugin must
not read any history row when `H=0`.

### 5.4 Attention graph

For these two causal, batch-one models, use the common
`NativeContiguousAttention` plugin version/ABI 2 backed by qualified cuDNN
Frontend SDPA graphs. The plugin:

- consumes read-only history K/V, current Q/K/V, and the valid history length
  `H`;
- uses non-causal fused SDPA for the history segment and lower-right causal
  fused SDPA for the current segment;
- merges the two segment results with cuDNN's standard per-query
  log-sum-exp (LSE) statistics;
- removes `attention_mask` from the normal engine I/O contract;
- never materializes `[Sq, Skv]`, `[M, M]`, or per-head score tensors;
- applies the model's explicit `1 / sqrt(head_dim)` query scaling;
- fails the build or qualification for an unsupported fused configuration
  instead of enabling a decomposed `QK^T -> softmax -> V` fallback.

Lower-right semantics are required for the current chunk: query row `i` may
attend to current-segment key rows through `i`. The separately computed
history segment contains exactly the `H` preceding tokens, so the stable merge
is equivalent to allowing the full logical prefix through global position
`H + i`.

For padded execution, the history bindings expose `[T, Hkv*D]`, while `H`
selects the valid prefix and the runtime enforces the Section 4 cold-sentinel
invariant and `T<=R`. The selected
implementation uses `T=1` for cold history and `T=P` for ordinary and captured
non-cold execution. The final, non-standard capacity value `R` is itself a
valid terminal `P` when `R` falls between build buckets. The plugin must not
read rows `[H, T)`.

The graph must also remove the current
`Concatenation(cache, new_rows)` operation. Although that concatenation is
dynamic, it creates or copies an `O(H)` full-history tensor on every layer and
every decode step. Dynamic allocation alone does not make that acceptable.

The supported graph boundary must not mutate a TensorRT input. The selected
design keeps cache history read-only during the engine invocation and commits
only the current rows after the split engine returns:

```text
runtime cache history K/V [T,Hkv*D] (read-only) --------+
new Q ---------------------------------------------------+
current K/V [Sq,Hkv*D] -- transpose view ----------------+--> segmented
history length H ----------------------------------------+    NativeContiguousAttention
                                                             version/ABI 2
                                                               |
                                                               +--> context

current K/V are also exact-shape engine outputs [Sq,Hkv*D]
                                                               |
engine completion --> copy only current rows to cache at H ----+
```

`NativeContiguousAttention` version/ABI 2 treats history and current K/V as
two logical segments without concatenating them. On cuDNN 9.20 it runs:

1. non-causal SDPA over the valid history prefix;
2. lower-right causal SDPA over the current chunk;
3. one stable merge using each segment's per-query LSE output.

If the two segment results are `(O1, lse1)` and `(O2, lse2)`, the merge uses
`l=max(lse1,lse2)`,
`w1=exp(lse1-l)`, `w2=exp(lse2-l)`, and
`O=(w1*O1+w2*O2)/(w1+w2)`. This is mathematically identical to one softmax
over `[history,current]`, while neither segment materializes a score matrix or
copies history. Cold prefill uses an empty valid-history length and reduces to
the current segment.

The K/V rows produced inside the decoder are also marked as engine outputs
before their transpose view feeds the attention plugin. The backend binds
these outputs to one exact-`Sq` staging allocation shared by the serialized
prefill/decode roles. After successful engine completion, the native runtime
copies only those `Sq` rows to their final offsets in the persistent cache.
This traffic is `Sq * B`, independent of `H`; it replaces the forbidden
full-history present-cache copy.

Phase 0 must prove all of these details on the pinned TensorRT build:

1. marked-and-consumed current K/V outputs use the caller's exact-shape
   staging addresses after serialization, with no profile-MAX output
   allocation;
2. cold prefill, continuation prefill, and decode match one-segment reference
   attention for both target GQA shapes;
3. stable segment merging remains finite and accurate for extreme logit
   offsets;
4. the post-engine commit touches only current rows, preserves red zones, and
   completes before the next prefill/decode invocation;
5. inspector and transfer traces contain no full-history copy.

Qualified builds require the cuDNN 9.20 standard SDPA LSE capability and fail
closed if it is unavailable. The simple reference CUDA kernel remains a
test/spike oracle only and is not a qualified runtime fallback.

cuDNN plan creation is part of target qualification, but a specific heuristic
index is not a portable runtime ABI. The former optional-output graph requested
separate score-max and score-sum-exp tensors. For `Sq=1` and history profiles
through 1,024, its first Heur A candidate (`eng3_k24=7` on the current stack)
failed NVRTC finalization before cuDNN selected a slower fallback. Forcing the
matching Python CUDA 13.0 NVRTC and builtins did not repair that candidate, so
the system-CUDA-13.3 versus Python-CUDA-13.0 mismatch was not the root cause.

The implementation now uses cuDNN's standard `set_generate_stats(true)` LSE
output. On the qualified GB300 candidate stack, cold-cache execution selects
`eng10_k24=7` for both target GQA shapes and executes the actual TensorRT
plugin chain at Qwen `T=512/1024/1025` and `T=40,960, H=40,959, Sq=1`, plus
TinyLlama `T=512, H=511, Sq=1`, without
`CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED`. Independent old-versus-LSE
probes produced bit-identical BF16 history and merged outputs at Qwen
`T=512/2,048/40,960` and TinyLlama `T=512/2,048`. The selected engine name is
evidence for this stack only, not a serialized ABI requirement.

The release implementation must therefore:

1. discover and link the NVRTC library from the same qualified CUDA
   distribution as the packaged TensorRT/cuDNN runtime, and include that
   directory in both build-tree and installed-package RPATHs; this is a
   packaging invariant, not the fix for the observed `Sq=1` failure;
2. record CUDA, cuDNN, cuDNN Frontend, NVRTC, driver, selected execution-plan
   identity, and cold/warm JIT-cache state in qualification receipts. The
   product worker emits one structured `[trtmc.runtime_stack]` row only after
   the independently detected live tuple has matched the bundle, so receipt
   tooling never substitutes builder-process or bundle-declared values for
   runtime evidence;
3. validate cold-cache and warm-cache loads in separate processes, including
   concurrent loads on different GPUs, without relying on a shared cache to
   make the first request succeed;
4. require every plan actually selected by the supported stack to pass the
   same numerical and 95% split-decode performance gates;
5. fail model qualification before release if the stack cannot produce a
   stable passing plan; do not fail inside `enqueueV3` merely because a
   heuristic index changed.

Cross-process locks, bounded retries, a different NVRTC search path, or a
pre-populated JIT cache are not correctness fixes. They must not be used to
turn an intermittent build into qualification evidence. Investigate the
failing cuDNN engine and qualify the actually selected fallback on cold and
warm processes. If that path cannot pass deterministically, replace decode
attention with a separately qualified fused kernel/plugin; never accept a
dense or full-history-copy fallback.

The original all-in-one PluginV3 alias proposal is retained only as a negative
regression test: TensorRT 11.2 accepts its build-time `getAliasedInput()`
declarations but omits them from serialized engine I/O metadata. Trusting
bundle names, reading pre-existing output memory, or mutating a const input
would not repair that contract and is forbidden.

TensorRT's built-in `IKVCacheUpdateLayer` is a second negative qualification
test for this milestone. Its alias contract is official and survives
serialization, but TensorRT 11.2 rejects a dynamic cache sequence dimension
with `The KV cache tensor must have a static sequence length dimension`.
Building it at `T=M` would restore the hidden allocation coupling, so it is not
used by the qualified graph.

Arbitrary masks, speculative trees, sliding-window models, and bidirectional
blocks are outside this first capability and must use a different declared
runtime contract.

### 5.5 Position encoding

Do not embed a full `M`-row RoPE cache into the execution graph solely because
the model supports a large context.

For Qwen3 and TinyLlama:

1. keep the inverse-frequency vector as a small constant;
2. derive angles from runtime `position_id`;
3. compute or gather cos/sin only for the current chunk;
4. validate the last legal model position in E2E.

This keeps position-encoding activation proportional to `C`, not `M`.

### 5.6 KV ownership and writes

The model-native runtime owns the default KV allocation. TensorRT receives
external bindings and must not allocate a duplicate profile-MAX input buffer.

The contiguous-v1 cache is physical token-major order inside each layer K or V
span. Engine execution reads history and writes only an exact-size staging
span; the runtime commit then appends current rows:

```text
cache_k_i input       -> layer K span, bound history view [T, Hkv*D]
cache_v_i input       -> layer V span, bound history view [T, Hkv*D]
current_k_i output    -> shared staging span [Sq, Hkv*D]
current_v_i output    -> shared staging span [Sq, Hkv*D]
history_length        -> H
post-engine commit    -> cache_[k/v]_i[H:H+Sq] = current_[k/v]_i
```

The cache pointer has physical capacity `R`, even though the invocation history
view is only `T`; the backend validates the Section 4 cold-sentinel invariant,
`T<=R`, `A=H+Sq<=R`, and byte capacity before enqueue. Prefill and decode use
the same base allocation. New
K/V rows are committed once to their final offsets, so there is:

- no K/V D2H;
- no full-history D2D;
- no cache/new-row concatenation;
- no temporary profile-MAX cache or present buffer;
- no cache copy when switching from prefill to decode.

The native attention plugin owns no persistent cache, performs no cache write,
and does not choose the budget. The Model Connect runtime remains the sole
owner of allocation, current-row staging, commit, admission, reset, and
lifetime.

For the public beta, the user controls the size policy, while Model Connect
owns the pointer and lifetime. The internal allocator interface accepts a
caller-supplied allocation for tests and future embedding APIs. In either
case, split acceleration is preserved because prefill and decode bind the same
pointer; cache ownership is independent of engine-role specialization.

### 5.7 Runtime allocation sequence

The runtime load sequence is:

```text
deserialize engine and load weights
        |
create USER_MANAGED execution contexts without device-memory blocks
        |
query free memory after fixed engine/plugin allocations
        |
derive a tentative R from policy, M, and U
        |
enumerate every Sq/T shape reachable by the native chunk/decode scheduler
        |
query every actual-shape context requirement and retain the true maximum
        |
solve R again after reserving that non-KV memory
        |
repeat downward until the memory plan is stable
        |
allocate one shared context block plus external output buffers
        |
synchronize, re-query, and allocate exactly R token rows of KV
        |
bind the same KV allocation to prefill and decode contexts
```

This is a bounded, decreasing solve rather than a one-time
`cudaMemGetInfo()` snapshot. Let:

```text
F = free bytes after engine weights and fixed plugin state
O(r) = max context device-memory block for capacity r
       + external device output buffers
       + graph-private device allocations reserved up front
S = safety reserve bytes
```

For a percentage `a`, start with:

```text
r0 = min(M, U if specified, floor(a * max(0, F - S) / B))
```

For each candidate `rn`, query TensorRT across the complete finite set of
prefill/decode shapes reachable by the native scheduler and compute:

```text
rn+1 = min(
    rn,
    M,
    U if specified,
    floor(a * max(0, F - S - O(rn)) / B)
)
```

Stop when the token capacity and enabled internal bucket set are unchanged.
If tactic/profile discontinuities prevent convergence within a small fixed
iteration count, evaluate the remaining lower bucket boundaries and choose the
largest plan whose measured allocation fits. Never increase `R` during this
solve.

The context envelope must not assume that
`updateDeviceMemorySizeForShapes()` is monotonic between TensorRT profile
endpoints. For prefill, probe every `Sq` from one through the current chunk
extent for the cold `T=1` shape and for each distinct history bucket reachable
after native chunk scheduling. For decode, probe the cold sentinel and every
enabled bucket, including a final bucket clipped to `R`. Deduplicate identical
`Sq/T` pairs, but do not replace the sweep with only MIN/OPT/MAX or largest
endpoint queries. The allocated shared block is the maximum returned by that
sweep; a later invocation still re-queries its exact shape and fails closed if
the backend violates the proven envelope.

Default `a` is `0.90`. The percentage applies to safely usable memory after
non-KV runtime overhead, not to total device memory and not to the
pre-engine-load free-memory reading.
Treat the accepted binary64 value of `a` as an exact integer ratio and perform
the multiply-and-floor with integer arithmetic; a rounded floating-point
product must never allocate even one byte above that ratio. Serialize
`policy_fraction` with enough significant digits to round-trip the original
binary64 value exactly.

Explicit bytes:

- first cap the requested budget by `M * B` and, when present, `U * B`; this
  avoids allocating rows the single-pipeline beta can never address;
- convert that semantically useful byte ceiling into `R`;
- query `O(R)` before allocating the context block or KV;
- require `R * B + O(R) <= F - S`;
- round down by at most one token row;
- fail clearly if fewer than one token row fits;
- do not silently shrink `R` because of memory pressure.

If `M` or `U` makes the useful allocation smaller than the supplied byte
ceiling, succeed and report both values. A later multi-request paged pool may
use a byte budget larger than one sequence's `M * B`; the single-pipeline beta
must not reserve unusable memory merely to make the byte count exact.

Final allocation:

```text
R = min(
    M,
    floor(resolved_kv_bytes / B),
    U if U was specified
)
allocated_kv_bytes = R * B
```

After allocating the context block and external device outputs, synchronize
and re-query free memory before the KV allocation. `auto`/percentage may
perform one final downward
recalculation if another process consumed memory; an explicit-byte policy
fails with resolved allocation and available values instead of silently
changing capacity.

If that final downward recalculation crosses a profile or `R<C` boundary and
reduces the required context/staging envelope, release and reallocate those
blocks to the final envelope, synchronize, and keep the already-decreased
`R`. Do not retain a hidden `C*B` staging allocation while reporting
`min(C,R)*B`, and do not use newly freed bytes to increase `R` again.

Receipt schema v3 gives the two synchronized snapshots distinct meanings:

- `capacity_decision_*` is sampled after the tentative context/output
  reservation and is the only second snapshot used to derive the final `R`
  and `kv_budget_bytes`;
- `settled_*` reuses the synchronized `after runtime KV allocation` boundary
  after any smaller context/staging replacement and the final KV slab are all
  resident. It reports actual settled residency and never feeds another solve;
- schema-v2 `final_*` remains only as an explicitly deprecated alias of
  `capacity_decision_*`, preserving evidence compatibility without presenting
  it as settled state.

If settled sampling fails, product inference may continue with `settled_*`
set to `null` and a typed unavailable reason, but qualification fails closed.

The first beta allocates this contiguous buffer once at model load for stable
addresses and predictable latency. The buffer is reused across sequential
requests and cleared logically by resetting cache length; it is released on
pipeline destruction.

### 5.8 TensorRT context and output memory

Opt in to user-managed actual-shape allocation only when all are true:

- the bundle declares `runtime_memory.contract_version = 1`;
- the selected backend is standard TensorRT;
- the runtime satisfies the pinned TensorRT 11.2 capability check;
- the engine was built with runtime activation resize enabled.

Legacy/static bundles and TensorRT-RTX retain their existing allocation path.
This is not a global `trt_module_impl` behavior change.

For an opted-in context, use this exact order:

1. enable `kRUNTIME_ACTIVATION_RESIZE_10_10` when building;
2. create the context with `kUSER_MANAGED`;
3. call `setOptimizationProfileAsync(profile, execution_stream)`;
4. set all input shapes;
5. allocate or reuse ordinary dynamic execution-input buffers at their
   concrete planned bytes and bind their addresses;
6. set addresses for every supported shape-inference I/O tensor; API v1 fails
   closed on shape-inference inputs because it has no value-aware planning
   descriptor and must not infer from an uninitialized placeholder;
7. require `inferShapes()` to report no missing shape inputs;
8. allocate or reuse ordinary dynamic output buffers at the concrete inferred
   bytes, and replace each host output staging allocation with its exact
   logical byte size;
9. call `updateDeviceMemorySizeForShapes()`;
10. allocate or reuse one correctly aligned context device-memory block;
11. call `setDeviceMemoryV2(pointer, bytes)`.

An ordinary dynamic device allocation begins at zero bytes. Growth
synchronizes in-flight execution, binds the replacement address
transactionally, releases the superseded allocation, and invalidates any CUDA
graph captured with the prior address. A smaller shape may reuse device
high-water capacity, but output metadata and host staging must still shrink to
the exact inferred shape. Any output that remains non-concrete after
`inferShapes()` fails closed until an `IOutputAllocator` policy is implemented.
Forward paths reject shape, dtype, or byte counts beyond the planned
materialization; they never truncate with a `min()` copy.

Because the native text-generation pipeline serializes prefill and decode,
their contexts may share one context block sized to the largest currently
enabled runtime bucket. Contexts must never use that block concurrently.

Dynamic outputs use one of three policies:

- exact-shape external current-K/V staging outputs, sized from the concrete
  `Sq` after `inferShapes()` and shared across serialized execution roles;
- exact-shape allocation for outputs whose shape is known after
  `inferShapes()`;
- `IOutputAllocator` only for genuinely data-dependent outputs whose concrete
  shape remains unknown until enqueue.

The backend must stop allocating all dynamic inputs and outputs at profile MAX.
An output allocator must not own KV memory, and it must not change an address
during CUDA graph replay.

The runtime-memory accounting ledger reports ordinary dynamic input and output
device capacities separately. Only allocations materialized after the
post-load snapshot enter the solver's ordinary-I/O overhead, summed once per
actual module/context rather than deduplicated by engine identity. Static I/O
remains in the post-load baseline. Caller-owned current-K/V staging remains
separate from both ordinary-I/O fields.

### 5.9 CUDA graphs

CUDA graphs are an optimization layer, not a correctness or context cap.

- allocate the KV buffer before graph capture so its base address is stable;
- bind a fixed history `T=P` within each selected non-cold decode bucket and
  pass changing `history_length=H`; use `T=1` only for `H=0`, while host
  admission validates the Section 4 cold-sentinel invariant, `T<=R`, and
  `A=H+Sq<=R`;
- key each graph by engine role, profile, `Sq/batch/T`, context-block
  generation, KV base addresses, and all external output addresses;
- after any shape/profile/address/generation change, run one uncaptured
  `enqueueV3` warm-up before capturing a new entry;
- use ordinary `enqueueV3` for an uncaptured shape;
- never reject a model-valid request merely because no CUDA graph was captured;
- include any preallocated graph-private device memory in `O(R)`;
- charge capture-time allocations that cannot be known in advance against the
  safety reserve, and skip capture if that reserve is insufficient.

Prefill graph capture is optional in the first beta. Correct chunked prefill
without capture is sufficient. CUDA graph capture must never trigger a second
KV allocation or silently reduce `R`. Padded history `T=P`
segmented-attention
semantics must pass Phase 0 before they become the qualified binding rule;
uncaptured `enqueueV3` remains available when no graph entry exists.

### 5.10 Steady-state decode scheduling

Runtime reconfiguration safety is tied to the TensorRT execution context, not
to every command queued on its CUDA stream. The backend therefore records
whether its most recent TensorRT enqueue is still in flight:

- a successful TensorRT enqueue marks the context in flight;
- a successful public `sync()` clears that state;
- a subsequent shape/address/context reconfiguration synchronizes only if the
  context is still in flight, then clears it;
- an asynchronous caller that reconfigures without first calling `sync()`
  still takes the required synchronization and remains safe.

The native Qwen/Llama pipeline synchronizes each engine execution before it
commits current rows. The commit itself is stream ordered but does not use or
mutate the TensorRT execution context. Consequently, the next decode
invocation may configure that already-idle context without draining the
commit first; the next H2D copies and enqueue remain ordered after the commit
on the same stream. This removes hundreds of redundant
`cudaStreamSynchronize()` calls per token without weakening the async API
contract.

Both the persistent KV slab and exact-`Sq` staging use:

```text
layer 0 K | layer 0 V | layer 1 K | layer 1 V | ...
```

The post-engine commit therefore uses one checked
`cudaMemcpy2DAsync()`:

```text
dst        = kv_base + H * row_bytes
dst_pitch  = R * row_bytes
src        = staging_base
src_pitch  = min(C, R) * row_bytes
width      = Sq * row_bytes
height     = 2 * layer_count
```

Admission has already proved `Sq <= min(C,R)` and `H+Sq <= R`. A copy error
poisons the request state before logical position advances. The transfer
ledger reports the exact logical bytes and one physical D2D event per
invocation. Injected-copy tests include `L=3,H>0,R!=C` and verify every K/V
span's pitches, prefix/write/tail extents, red zones, failure poisoning, and lifetime.

The copy remains asynchronous between decode steps. Immediately before a
successful public request result or qualification receipt is returned, one
request-completion barrier checks the last pending commit. A delayed CUDA
error poisons runtime KV state and is propagated; it cannot be swallowed by
the separate best-effort, `noexcept` memory-observability sample. Request
reset also finalizes any pending commit left by an earlier exceptional exit.
This adds one fence per request, not one fence per token.

CUDA graph capture remains disabled for the initial two-model beta. The
uncaptured path must meet the 95% split-decode gate, keeps
`graph_private_device_bytes == 0`, and avoids charging an unknown first-capture
allocation after runtime capacity has already been resolved. If it does not
meet the gate, the beta stops until the fused decode path is fixed; CUDA graph
capture must not be used to hide an unstable plan-selection or JIT problem.
CUDA graphs remain a later optimization under Section 5.9, not a correctness
dependency or hidden context cap.

## 6. Current prototype disposition

The existing [dynamic KV split prototype](dynamic-kv-split-prototype.md)
provides useful evidence but is not the implementation to publish.

### Reuse conceptually

- model-only no-flag build routing;
- runtime `auto`, percentage, bytes, and sequence-limit parsing;
- post-engine `cudaMemGetInfo()` budgeting;
- deferred external TensorRT input binding;
- split prefill/decode traceability;
- logits-only host materialization;
- overflow-safe admission checks;
- Qwen and Llama implementation capability detection.

Do not reuse the prototype's family-wide routing decision. Replace it with the
qualified model/revision/config/platform gate defined in Section 1.

### Replace before beta

| Prototype behavior | Required replacement |
|---|---|
| Hidden `min(M, 4096)` engine cap | Use `M` as the semantic limit; use `C` only as the per-invocation prefill maximum. |
| Full prompt in one prefill launch | Chunked prefill with exactly `ceil(L/C)` launches. |
| Explicit FP32 dense mask | Segmented fused cuDNN attention with lower-right causal handling for the current chunk. |
| `Concatenation(cache, new_rows)` | Segmented history/current attention plus current-row-only runtime commit; never materialize or copy full history. |
| Profile-MAX I/O allocation | External bindings and actual-shape output/context allocation. |
| D2D copy from full present K/V into cache | Copy only exact-`Sq` current K/V staging rows into the runtime-owned cache. |
| Profile-MAX RoPE table | Runtime-position-driven RoPE for the active chunk. |
| TensorRT contexts allocate hidden profile-MAX memory before budgeting | Create planning contexts with `kUSER_MANAGED`, query actual-shape requirements without attaching device memory, allocate one shared context block only after resolving `R`, and retain only the decode bucket set needed by `R`. Temporary planning contexts may consume host-side TensorRT state but must report zero independently allocated context/KV device bytes. |
| One ambiguous max-cache field | Versioned runtime-memory metadata with `M`, `C`, `B`, layout, and buckets. |
| Dynamic `PipelinePool` rejection | Retain fail-fast for beta; replace with a shared device-level paged pool in the concurrency phase. |

The prototype currently touches more than forty files. Do not publish it as
one monolithic PR. Land the new implementation through the staged PR sequence
in Section 12.

### Current qualification status

The former `Sq=1` blocker is resolved in the component implementation by
replacing separate max/sum-exp optional outputs with the standard cuDNN LSE
statistics path. Cold private-cache plugin execution reaches Qwen
`T=40,960, H=40,959, Sq=1` without a compilation failure.

The clean `0639f7ab` candidate built all six required artifacts: no-flag
dynamic, exact-head static split, and source-bound `C/2` bundles for both
models. Its fixed manifest passed 161 CTests, 22 dynamic-memory CTests, 290
selected dynamic-memory pytest nodes, and both real TensorRT graph tests.
Those results are diagnostic rather than promotable because the formal
correctness producer compared a device-wide CUDA delta with a current-process
NVML delta while unrelated GPU processes were changing. The signed mismatch
changed direction across otherwise identical runs, so increasing the
tolerance would be incorrect. The formal performance producer also exposed
two product defects: `cuda_jit_cache` was not emitted at the top-level schema
required by its consumer, and the Qwen dynamic prefill plan retained a second
copy of the tied 151,936-by-1,024 vocabulary matrix, causing an approximately
11.4% MEM-13 packaging regression.

The current review candidate fixes the producer schema, records independently
attributable CUDA/NVML scopes at every synchronized boundary, preserves all
runner evidence on failure, and reuses the Qwen embedding tensor for the LM
head only after shape, dtype, and bit-exact transpose validation. It also
fail-closes the core/backend/model-plugin ABI boundary, binds the fixed test
manifest to exact mapped build artifacts, and adds a GQA-aware direct
`Sq=1, P<=512` decode kernel while retaining the standard-LSE cuDNN path for
larger history profiles and all prefill work. The integrated dirty review
source passes 162/162 CTests, all 24 dynamic-memory CTests, all 549 selected
dynamic-memory pytest nodes, and both real TensorRT graph tests. CUDA
memcheck/racecheck report no errors or hazards for the direct decode path; its
same-GPU component microbenchmark improves from a 40.3789 microsecond median
to 11.6749 microseconds. The independent CUDA-13.0 NVRTC negative replay also
passes every component gate while remaining explicitly non-promotable because
the source is dirty.

These component and diagnostic results do not replace the source-bound model
matrix. This source state has not yet completed the clean exact-HEAD
40,960/2,048 correctness, pressure, soak, surface, isolation, and end-to-end
performance receipts. Therefore the current answer remains “implementation
candidate,” not “qualified full-context support.”

One final clean HEAD must regenerate both dynamic bundles, both exact-head
static baselines, both `C/2` variants, every
correctness/memory/soak/surface/performance receipt, and the v2
process-isolation aggregates. Older or dirty receipts remain diagnostic
evidence only; they do not qualify a different source state.

## 7. File-level implementation plan

### 7.1 Shared build policy and bundle metadata

Primary files:

```text
python/tensorrt_model_connect/build_cli.py
python/tensorrt_model_connect/engine_builder.py
python/tensorrt_model_connect/bundle_writer.py
python/tensorrt_model_connect/families/qwen/MODEL.toml
python/tensorrt_model_connect/families/llama/MODEL.toml
include/trtmc/bundle.h
src/bundle/bundle_format.h
src/bundle/bundle_format.cpp
src/cli/main.cpp
```

Add one shared Python contract helper, rather than duplicating policy in both
families:

```text
python/tensorrt_model_connect/dynamic_memory_contract.py
python/tensorrt_model_connect/families/qwen/MODEL.toml
python/tensorrt_model_connect/families/llama/MODEL.toml
```

Responsibilities:

- read and validate `M` from the pinned model revision, including any
  family-qualified RoPE scaling semantics;
- select family-owned `C` and history-extent buckets (serialized in contract
  v1 as `active_kv_profile_limits`);
- calculate and serialize `B`;
- emit contract version and layout;
- match canonical model ID, resolved revision, graph-config fingerprint, and
  target platform against a model-owned qualification profile;
- select the native implementation only after that exact match;
- remove the 4,096 prototype cap from user builds;
- keep runtime memory policy out of build arguments.

The qualification resolver has separate `not_applicable` and `invalid`
results. An unqualified Qwen/Llama tuple is `not_applicable` and follows the
same legacy/provider routing it used before this work. A tuple that matches a
qualification record but has an ambiguous/contradictory context declaration
is `invalid` and fails with the revision, fingerprint, and relevant config
fields. Never substitute a small certification default. Tests must prove that
the family capability bit alone cannot enable dynamic-memory routing.

### 7.2 Qwen and Llama graph builders

Primary files:

```text
python/tensorrt_model_connect/families/qwen/standard_decoder_builder.py
python/tensorrt_model_connect/families/qwen/dual_profile_decoder_builder.py
python/tensorrt_model_connect/families/qwen/graph_ops.py

python/tensorrt_model_connect/families/llama/standard_decoder_builder.py
python/tensorrt_model_connect/families/llama/dual_profile_decoder_builder.py
python/tensorrt_model_connect/families/llama/graph_ops.py
```

Responsibilities:

- dynamic `Sq` bounded by `C`;
- dynamic bound cache extent `T` with profile maximum `M`;
- runtime history length `H`, with backend admission enforcing
  the Section 4 cold-sentinel invariant, `T<=R`, and `A=H+Sq<=R`;
- one runtime `history_length` scalar carrying exactly `H`; commit offsets
  stay in the runtime state and are not graph inputs;
- separate prefill/decode profiles;
- segmented fused attention with lower-right causal current-segment semantics;
- no normal causal `attention_mask` input;
- no cache/new-row concatenation;
- expose exact-`Sq` current K/V rows as external outputs and feed their
  transpose views, plus read-only cache history, to the common segmented
  attention plugin;
- dynamic-position RoPE;
- engine inspector names for attention/profile qualification.

Avoid a broad Qwen/Llama graph refactor in the same change. Introduce only the
small shared helpers needed to keep the new runtime-memory contract identical.

### 7.3 Common native attention plugin

Create:

```text
src/plugins/runtime_kv/
  native_contiguous_attention_plugin.h
  native_contiguous_attention_plugin.cu
  native_contiguous_attention_creator.cpp
  cudnn_attention.h
  cudnn_attention.cpp
  runtime_kv_plugin_api.h
  runtime_kv_plugin_api.cpp
  native_kv_append_plugin.h
  native_kv_append_plugin.cu
  native_kv_append_creator.cpp
  CMakeLists.txt
```

Also update the top-level build and plugin registration/packaging paths:

```text
CMakeLists.txt
pyproject.toml
tools/ci/package.py
python/tensorrt_model_connect/engine_builder.py
```

Responsibilities:

- implement `IPluginV3OneBuildV2` and `IPluginV3OneRuntime`;
- accept read-only history K/V with dynamic bound extent `T`, current K/V and
  Q with dynamic `Sq`, and history length `H`;
- return only the context tensor and declare no PluginV3 input/output alias;
- perform segmented GQA through two qualified cuDNN Frontend SDPA graphs and a
  stable LSE merge;
- assume host-side validation of the Section 4 cold-sentinel invariant,
  `T<=R`, and `A=H+Sq<=R` has already passed and
  retain defensive runtime-shape validation;
- return zero persistent allocation and use TensorRT workspace if scratch is
  needed;
- serialize an explicit plugin ABI/version;
- build a common `libtrtmc_trt_plugins` target that the Python builder loads
  before network construction and the TensorRT backend loads/registers before
  engine deserialization;
- expose layer metadata to the engine inspector.
- bind cuDNN and NVRTC from one qualified CUDA runtime stack and preserve that
  pairing in build/install RPATHs;
- expose selected-plan and JIT-cache provenance to the qualification tools
  without making either one a user setting.

The plugin is common TensorRT infrastructure. Do not place it in either model
DSO and do not place it under an EdgeLLM directory. `NativeKvAppendV1`, if
kept, is a negative qualification fixture only and must never be selected by a
qualified model graph.

Engine inspection after serialization, exact-shape output-address proof, and
transfer evidence are release gates, not optional profiling.

### 7.4 TensorRT backend

Public/internal interfaces:

```text
include/trtmc/runtime/trt_backend.h
include/trtmc/runtime/trt_module.h
src/runtime/backend/runtime_memory_backend.h
src/runtime/backend/backend_loader.cpp
src/runtime/backend/trt_backend.cpp
src/runtime/backend/trt_module_impl.h
src/runtime/backend/trt_module_impl.cpp
src/runtime/backend/rtx_backend.cpp
src/runtime/core/cuda_common.h
src/runtime/core/trt_common.h
src/runtime/core/trt_common.cpp
```

`rtx_backend.cpp` is in this list only for explicit compatibility guarding and
tests; it does not adopt the new allocation strategy in this milestone.

Responsibilities:

- create user-managed contexts;
- query actual-shape device-memory requirements;
- accept a caller-owned shared context device-memory block;
- defer selected input and output allocation;
- replace the current pointer-plus-shape external binding with a versioned
  descriptor containing `pointer`, `shape`, `capacity_bytes`, `dtype`,
  `format/strides`, `alignment`, `device`, and `owner/lifetime`;
- bind read-only cache history inputs to the persistent allocation and
  exact-shape current K/V outputs to the shared staging allocation;
- require `kLINEAR` cache I/O for v1, or calculate required bytes from actual
  strides rather than assuming contiguous rows;
- reject any PluginV3 cache alias metadata in the qualified engine and verify
  that every declared current-K/V output has the expected mode, dtype, rank,
  and dynamic `Sq`;
- perform every cold-sentinel, `T<=R`, `A=H+Sq<=R`, byte-capacity, alignment,
  and write-range check on the host before `enqueueV3`;
- expose D2H/D2D byte counters for receipts;
- retain execution/profile state without allocating profile-MAX copies;
- invalidate/warm CUDA graph entries when shape, profile, context-block
  generation, or cache base address changes.

Any public ABI extension must carry an explicit size/version handshake. The
core, backend, and model DSO must fail closed on an incompatible contract
instead of relying only on tail-appended C++ structs.

### 7.5 Runtime memory planner

Create one shared text-generation dynamic-memory implementation:

```text
src/runtime/domains/text/dynamic_memory/kv_cache_budget.h
src/runtime/domains/text/dynamic_memory/runtime_memory_plan.h
src/runtime/domains/text/dynamic_memory/runtime_memory_plan.cpp
src/runtime/domains/text/dynamic_memory/runtime_kv_allocation.h
src/runtime/domains/text/dynamic_memory/runtime_kv_allocation.cpp
src/runtime/domains/text/dynamic_memory/runtime_kv_state.h
src/runtime/domains/text/dynamic_memory/runtime_kv_state.cpp
src/runtime/domains/text/dynamic_memory/runtime_kv_setup.h
src/runtime/domains/text/dynamic_memory/runtime_kv_setup.cpp
src/runtime/domains/text/dynamic_memory/runtime_memory_qualification.h
src/runtime/domains/text/dynamic_memory/runtime_memory_qualification.cpp
```

Responsibilities:

- checked parsing-independent budget arithmetic in the header-only budget
  helper;
- exact bytes-per-token validation;
- safety reserve;
- auto/percentage/bytes resolution;
- runtime admission;
- structured memory receipt;
- exact live-target validation before deserialization: query the current CUDA
  device compute capability and the four-component version from the actually
  loaded `libnvinfer`, then require `sm103` and `11.2.0.113`;
- allocator injection for tests and future embedding APIs;
- deterministic error objects/messages.

The default allocator uses CUDA device memory. Internally expose an allocator
contract with pointer, byte size, device, alignment, stream, and lifetime so a
future C++ embedding caller can provide its own KV buffer without redesigning
the model pipelines.

### 7.6 Qwen and Llama native runtimes

Primary files:

```text
src/runtime/models/qwen/kv_cache.h
src/runtime/models/qwen/kv_cache.cpp
src/runtime/models/qwen/inference_state.h
src/runtime/models/qwen/pipeline.h
src/runtime/models/qwen/pipeline.cpp
src/runtime/models/qwen/plugin.cpp
src/runtime/models/qwen/MODEL.toml

src/runtime/models/llama/kv_cache.h
src/runtime/models/llama/kv_cache.cpp
src/runtime/models/llama/inference_state.h
src/runtime/models/llama/pipeline.h
src/runtime/models/llama/pipeline.cpp
src/runtime/models/llama/plugin.cpp
```

Responsibilities:

- consume the shared runtime-memory contract;
- allocate `R * B` bytes after fixed runtime load;
- bind the same KV state to prefill and decode;
- execute chunked prefill;
- select decode buckets;
- bind cache history and exact-`Sq` current-K/V staging, then commit only
  current rows after a successful engine invocation;
- keep checked commit offsets and `H/A` in runtime state while passing only
  `history_length=H` into the graph;
- validate prompt plus output before prefill;
- reset and reuse state safely across requests;
- emit structured trace and memory receipts.

Qwen and Llama should share the memory planner and behavioral contract. Their
model-owned code remains responsible for tensor names, tokenizer/chat
behavior, model-specific RoPE, and engine metadata validation.

### 7.7 CLI and pipeline routing

Primary files:

```text
include/trtmc/pipeline.h
src/cli/args.h
src/cli/args.cpp
src/cli/main.cpp
src/runtime/config/cli_support.cpp
src/runtime/registry/pipeline_factory.cpp
```

Responsibilities:

- typed `auto`, percentage, and bytes policy;
- optional runtime max sequence;
- conflict/overflow validation;
- keep the original `LoadOptions` and `PipelineContext` object layouts frozen;
- propagate the new policy through a size/versioned `LoadOptionsV2` and an
  optional `IRuntimeMemoryPipelinePluginV1` handshake;
- fail closed for old-core/new-plugin, new-core/old-plugin, and wrong-version
  model-DSO combinations instead of reading tail fields;
- contract-version checks before provider dispatch;
- static/legacy bundle compatibility;
- exact qualified-tuple routing without family-wide enablement.

Do not add EdgeLLM-specific conditions here. Check the bundle's declared
runtime-memory contract generically.

### 7.8 C ABI and Python embedding surface

Primary files:

```text
include/trtmc/pipeline.h
src/cabi/api/trtmc_c.cpp
python/tensorrt_model_connect/pipeline.py
```

Responsibilities:

- expose the same `auto`, percentage, bytes, and optional sequence-cap policy
  as the CLI;
- leave `TrtmcPipelineOptions` and `trtmc_create_pipeline_ex()` unchanged for
  binary compatibility;
- add `TrtmcPipelineOptionsV2` with `struct_size` and `api_version` as its first
  fields, plus `trtmc_pipeline_options_v2_init()` and
  `trtmc_create_pipeline_v2()`;
- preserve source and binary compatibility for existing callers;
- freeze and test the legacy `LoadOptions`, `PipelineContext`, and
  `TrtmcPipelineOptions` size/alignment/layout with an independently compiled
  old consumer;
- test old-plugin/new-core and new-plugin/old-core combinations: legacy
  static bundles retain their old path, while dynamic bundles fail closed
  when the optional V1 handshake is missing or incompatible;
- reject conflicting policy fields consistently across C++, C ABI, Python, and
  CLI;
- add constructor-level policy fields to the current subprocess-backed Python
  `Pipeline` wrapper and translate them to the same CLI flags;
- return the resolved capacity and structured memory receipt through
  introspection;
- leave room for a later caller-supplied allocator/buffer API without exposing
  raw TensorRT bindings as the public contract.

The first implementation may keep external allocator injection as an internal
test seam, but the public surfaces must not disagree about who chooses runtime
memory.

### 7.9 Tests and receipts

Extend the current prototype tests instead of discarding their useful
contracts:

```text
tests/builder/test_model_only_dynamic_kv_cli.py
tests/builder/test_dynamic_kv_split_prototype.py
tests/builder/test_bundle_writer.py
tests/cpp/test_kv_cache_budget.cpp
tests/cpp/test_cli_args.cpp
tests/cpp/test_bundle_format.cpp
tests/cpp/test_trt_module.cpp
tests/e2e/test_bundle_inspect.py
tests/e2e/models/qwen/test_qwen_builder_engine.py
tests/e2e/models/llama/test_llama_standard_decoder.py
```

Add focused targets:

```text
tests/cpp/test_runtime_memory_plan.cpp
tests/cpp/test_runtime_kv_allocation.cpp
tests/cpp/test_runtime_kv_setup.cpp
tests/cpp/test_runtime_memory_transfer_ledger.cpp
tests/cpp/test_native_contiguous_attention_plugin.cpp
tests/cpp/test_native_kv_append_plugin.cpp
tests/cpp/test_native_segmented_attention_long_context.cpp
tests/cpp/test_runtime_kv_plugin_backend_deserialization.cpp
tests/cpp/test_rtx_runtime_memory_compatibility.cpp
tests/cpp/test_trt_dynamic_kv_cache_update.cpp
tests/cpp/models/qwen/test_qwen_runtime_admission.cpp
tests/builder/test_runtime_memory_contract.py
tests/builder/test_runtime_kv_plugins.py
tests/builder/test_dynamic_memory_qualification.py
tests/e2e/test_native_dynamic_memory_graph.py
tests/qualification/native_dynamic_memory_qualify.cpp
tests/qualification/native_dynamic_memory_surfaces.cpp
tools/build_native_dynamic_memory_chunk_variant.py
tools/capture_dynamic_memory_test_manifest.py
tools/qualify_native_dynamic_memory.py
tools/capture_native_dynamic_memory_perf.py
tools/capture_native_dynamic_memory_process_isolation.py
tools/qualify_native_dynamic_memory_perf.py
tools/qualify_native_dynamic_memory_policies.py
tools/qualify_native_dynamic_memory_soak.py
tools/qualify_native_dynamic_memory_surfaces.py
tests/tools/test_build_native_dynamic_memory_chunk_variant.py
tests/tools/test_capture_dynamic_memory_test_manifest.py
tests/tools/test_capture_native_dynamic_memory_perf.py
tests/tools/test_capture_native_dynamic_memory_process_isolation.py
tests/tools/test_qualify_native_dynamic_memory.py
tests/tools/test_qualify_native_dynamic_memory_perf.py
tests/tools/test_qualify_native_dynamic_memory_policies.py
tests/tools/test_qualify_native_dynamic_memory_soak.py
tests/tools/test_qualify_native_dynamic_memory_surfaces.py
tests/tools/test_package_cuda_runtime_metadata.py
```

Keep long-context, performance, and memory receipts under the normal artifact
root with source/bundle hashes; do not commit generated engine plans or
hardware logs to the source tree.

`test_native_contiguous_attention_plugin.cpp` is the selected production path.
`test_native_kv_append_plugin.cpp` remains only as an executable negative
alias-contract fixture.

Add a clean-package test that builds in one fresh process and deserializes in
another, proving the installed Python builder can locate the plugin DSO and
the runtime registers the same creator/ABI before engine deserialization.
Assign every focused CTest the `dynamic_memory` label and every focused Python
test the `dynamic_memory` marker. Qualification stores the exact `ctest -N`
and `pytest --collect-only` manifests, so the evidence can be reconstructed
without relying on a stale hard-coded test count.

The manifest producer must explicitly build `trtmc_cpp_tests`,
`trtmc_dynamic_memory_qualify`, `trtmc_dynamic_memory_surfaces`, and
`trtmc_benchmark_worker` before collecting either CTest manifest. A successful
generic `cmake --build` alone is not proof that those non-default or stale
qualification binaries match the frozen source snapshot.

The v2 manifest is a fixed, ordered nine-command contract. Its validator
reopens every stdout/stderr log and both pytest JUnit files, recomputes the
collected manifests, and rejects an omitted, reordered, or altered command.
It also binds full device/inode/time/size/SHA identities for the CLI, worker,
core, active TensorRT backend, runtime-KV plugin, both model DSOs, and both
qualification binaries. The active versioned TensorRT backend name must be a
symlink to the generic backend inode; an independently copied alias is not an
exact-head artifact.

The performance gate must never consume hand-authored timing JSON directly.
`capture_native_dynamic_memory_perf.py build` executes the actual build argv,
requires an absent output path, and records identical pre/post source-state
digests. For a dynamic build it accepts only `trtmc build`, binds the one
adjacent or packaged runtime-KV plugin without adding a product build flag,
observes that exact DSO in the `trtmc`/builder process tree, and records its
canonical absolute path, size, and SHA-256. Both the benchmark producer and
the final performance validator reopen that DSO and recompute its identity.
The `benchmark` action executes the real C++ worker, independently deserializes
every engine section through TensorRT, and adds SHA-bound plan, resident-weight,
copy-count, and streaming evidence. For a dynamic bundle it also requires
those independent measurements to equal the pipeline's runtime receipt. The
static baseline uses the same independent engine inspection because it
deliberately has no dynamic-memory introspection interface.

Qualification launches the CLI and worker through their already-open file
descriptors and passes the already-open plugin descriptor into the child.
It records the process mappings while they are live, pins every mapped TRTMC
or ELF DSO, and scans defined ELF dynamic symbols without a directory or
basename allowlist. Path-swap, deleted-map, duplicate-name, renamed-plugin,
wrong-model, and stale-backend evidence all fail closed.

## 8. Implementation phases

### Phase 0: focused TensorRT spikes

No user-visible behavior changes.

Required experiments:

1. Negative executable proofs that PluginV3 cache alias metadata is absent
   after TensorRT 11.2 serialization and that `IKVCacheUpdateLayer` rejects a
   dynamic cache sequence dimension. Neither path may be selected.
2. Segmented history/current attention parity for:
   - cold prefill (`Sq == Skv`);
   - continuation prefill (`Sq < Skv`);
   - decode (`Sq == 1`).
3. Padded decode execution with `H < T=P <= R` and `A=H+1 <= R`, proving
   that `H` controls history padding while current-segment lower-right masking
   remains correct.
4. Exact-shape marked-and-consumed current K/V outputs, proving serialized
   engines bind caller staging, do not allocate profile-MAX outputs, and copy
   no full history.
5. Stable LSE segment merge under adversarial score offsets, plus
   current-row commit red zones and transfer accounting.
6. cuDNN 9.20 standard SDPA LSE support for both models' head/GQA shapes and
   selected dtype on GB300.
7. Opt-in `kUSER_MANAGED` sizing at short and long `Sq/T` shapes, while a
   legacy static bundle and TensorRT-RTX retain their old behavior.
8. The bounded decreasing memory-plan solve under both smooth and
   profile-discontinuous context-memory requirements.
9. Physically split prefill/decode performance and resident-weight accounting;
   optionally measure a one-engine/multi-profile packaging experiment without
   making it a beta dependency.
10. Cold- and warm-cache cuDNN/NVRTC plan creation in isolated processes,
    using the exact packaged CUDA stack, plus a concurrent different-GPU load
    test that proves no cross-process lock or pre-populated cache is required.
    The test must preserve the forced-NVRTC diagnostic proving that CUDA 13.0
    alone does not repair the preferred `Sq=1` engine.

Exit criteria:

- all ten experiments have reproducible source tests;
- segmented read-only attention plus current-row commit is selected with exact
  inspector/profiler evidence;
- no dense-mask or decomposed-attention fallback is required;
- no cache/new-row concatenation or full-history D2D copy remains;
- the selected engine packaging is recorded with memory/performance evidence.

### Phase 1: runtime ownership and versioned contract

Implement:

- bundle metadata split into `M`, `C`, `B`, layout, and buckets;
- exact model/revision/config/platform qualification records;
- typed runtime memory policy;
- shared budget/admission code;
- common segmented-attention plugin build, serialization, registration, and
  capability validation;
- backend deferred external bindings;
- user-managed context memory;
- activation-resize builder feature;
- structured receipts;
- old-bundle and TensorRT-RTX compatibility;
- all target routing still disabled.

Exit criteria:

- CPU/property and synthetic TensorRT tests pass;
- backend-owned cache-input bytes are zero;
- old/static/RTX paths are byte-for-byte or behaviorally unchanged under their
  existing tests;
- plugin discovery works from a clean installed package;
- no model has user-visible dynamic-memory routing yet.

### Phase 2: Qwen vertical slice and hidden-cap removal

Implement:

- qualified no-flag routing for `Qwen/Qwen3-0.6B` only;
- segmented fused attention;
- chunked prefill;
- dynamic-position RoPE;
- read-only segmented attention and current-row-only KV commit;
- runtime bucket selection;
- removal of the 4,096 cap.

Exit criteria:

- Qwen passes 32K and 40,960 total-length qualification;
- a dedicated 40,960-token prefill executes and compares the last legal model
  position;
- context memory follows the measured actual-`Sq/T` envelope and does not
  allocate for profile `M`;
- no `[M, M]` mask/score allocation exists;
- no full-history concatenate/copy exists;
- prefill/decode share one KV allocation;
- split performance gates pass.

This phase is Qwen-only evidence, not the two-model beta.

### Phase 3: TinyLlama generalization and two-model beta

Implement:

- qualified no-flag routing for
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`;
- shared-contract Llama graph/runtime wiring;
- C ABI V2 and Python policy parity;
- cross-family compatibility and documentation.

Exit criteria:

- TinyLlama passes its 2,048 model boundary and a dedicated 2,048-token
  last-position prefill comparison;
- unqualified Qwen/Llama tuples retain previous routing;
- all user-flow, memory, split, error, compatibility, and packaging gates in
  Section 9 pass for both bundle SHAs;
- the exact-head resident-weight and performance gates pass.

This is the first releasable single-pipeline dynamic-memory beta.

### Phase 4: device-level paged KV and concurrency

This is a follow-up after the two-model beta, not part of the first review
diff.

Add:

- fixed-size KV pages;
- a per-device block pool;
- per-request block tables and active lengths;
- paged prefill/decode attention plugin or qualified kernel integration;
- page allocation, release, cancellation, and exception cleanup;
- shared pool partitioning across `PipelinePool` lanes;
- fragmentation and soak receipts.

The native TensorRT linear KV update layer is insufficient for this phase.
The attention kernel must consume the page table directly.

Exit criteria:

- one 90% budget is allocated per device, not once per lane;
- multiple requests cannot overwrite or reserve each other's pages;
- split prefill/decode use the same pool;
- page usage follows active tokens within one-page rounding;
- no full-cache repack is required.

### Phase 5: 1M-class extension

Only begin after a real 1M-capable model and target hardware are selected.

Potential additions:

- FP8 KV where model/hardware qualification permits it;
- host or peer-GPU offload;
- KV/context parallel;
- sequence-sharded attention;
- CUDA VMM-backed elastic page superblocks;
- sparse/window attention only when required by model semantics.

No 1M support claim is allowed from the Qwen3-0.6B and TinyLlama results alone.

## 9. Validation matrix

Each model is built exactly once without memory, sequence, profile, precision,
output, or layout flags. Every runtime row reuses the same bundle SHA.

### 9.1 User-flow and compatibility tests

| ID | Test | Required result |
|---|---|---|
| UX-01 | `trtmc build <model>` | Deterministic output bundle; no user build-time context decision. |
| UX-02 | Static `trtmc inspect` | Shows contract, `M/C/B/layout/buckets`, revision, and fingerprint without initializing CUDA; no runtime-budget claim or 4,096 Qwen product cap. |
| UX-03 | Run `auto`, percentage, bytes, and runtime max sequence | Same bundle SHA and mtime; no rebuild; load-time receipt reports the resolved device values. |
| UX-04 | Same greedy request under different sufficient budgets | Identical generated token IDs and qualified logits. |
| UX-05 | Set equivalent policy through CLI, C++, C ABI, and Python | Same resolved `R`, admission result, and receipt fields. |
| COMPAT-01 | Load legacy static bundle with no new options | Existing behavior remains unchanged. |
| COMPAT-02 | Pass dynamic policy to unsupported/legacy bundle | Clear unsupported-contract error; no silent ignore. |
| COMPAT-03 | Load the corresponding TensorRT-RTX path | Existing static allocation/routing remains unchanged; no accidental `USER_MANAGED` opt-in. |
| QUAL-01 | Build an unqualified Qwen and Llama checkpoint | Previous routing remains unchanged; family capability alone does not select dynamic memory. |
| QUAL-02 | Change target revision/config fingerprint | Qualification miss is explicit and cannot inherit another checkpoint's evidence. |
| QUAL-03 | Cold-cache, warm-cache, and concurrent different-GPU process loads | A v2 aggregate receipt directly proves isolated cache/process behavior, overlapping different-GPU engine loads, and deterministic child token IDs. It must also bind passed full-matrix HF correctness and SPLIT-08/09 companion receipts to the exact same bundle SHA, clean source state, model revision, live runtime stack, and mapped runtime-library identity. |
| SCOPE-01 | Diff scope check | No EdgeLLM adapter or EdgeLLM test file changed. |

QUAL-03 is an exact-tuple aggregate, not a claim that each of the four
isolation children recomputes the HF reference or runs an exact-head static
baseline. The report must preserve that limitation explicitly: the child
processes directly prove cold/warm private-cache behavior, cross-GPU
concurrency, engine-load overlap, and repeatable generated tokens; the
source-bound companion receipts prove full-matrix HF parity and SPLIT-08/09
for the identical tuple.

### 9.2 Long-context correctness

Generate deterministic token ID inputs. Do not estimate token length from
human-readable text.

Qwen3-0.6B:

- `C - 1`, `C`, `C + 1`;
- `2C + 17`;
- for every non-terminal history bucket `P`, a `P`-token prompt followed by
  two decode invocations, proving `H=P,T=P` and then
  `H=P+1,T=ceil_bucket(P+1)` select the next decode profile without re-prefill;
- 32,760 prompt tokens plus 8 decode tokens, totaling 32,768;
- 40,952 prompt tokens plus 8 decode tokens, totaling `M = 40,960`;
- a dedicated prefill-only 40,960-token input that executes position 40,959
  and compares its final-position logits;
- total length `M + 1`, rejected before prefill.

TinyLlama:

- every decode bucket boundary `P - 1`, `P`, `P + 1`;
- for every non-terminal history bucket `P`, a `P`-token prompt plus two
  decode invocations with the same exact profile-crossing proof as Qwen;
- 2,040 prompt tokens plus 8 decode tokens, totaling `M = 2,048`;
- a dedicated prefill-only 2,048-token input that executes position 2,047
  and compares its final-position logits;
- total length 2,049, rejected before prefill.

For long prompts:

- compare the final prompt position and first eight decode steps to the HF
  reference;
- require stable top-1 agreement and existing family logit thresholds;
- run a developer test at `C` and `C/2` and require equivalent results; the
  `C/2` bundle must come from the dedicated source-bound producer receipt
  described below;
- assert Qwen prefill launches equal `ceil(prompt_tokens / C)`;
- never substitute a needle-in-a-haystack answer for numeric parity.

The developer-only `C/2` comparison has no hand-authored bundle escape hatch.
`build_native_dynamic_memory_chunk_variant.py` derives the one legal `C/2`
contract from the exact qualified tuple, performs a fresh build, and records
the producer, command timing, bundle path/size/SHA, contract, unchanged
pre/post source state, and the actual loaded and mapped runtime-KV plugin's
canonical path/size/SHA. `qualify_native_dynamic_memory.py` reopens the plugin
and recomputes that identity; it also requires
`--chunk-variant-bundle` and `--chunk-variant-build-receipt` together and
fails closed unless that receipt identifies the supplied bundle and the same
source SHA/HEAD as the base qualification. Supplying either argument without
the other, reusing a different-source receipt, or changing the variant bundle
after the build is an error. Canonical correctness promotion requires both
the replayed `C/2` engine-graph evidence and the reopened producer receipt;
omitting the pair can produce diagnostic evidence but can never set
`passed=true`.

The canonical base bundle has the same fail-closed provenance requirement.
`qualify_native_dynamic_memory.py` consumes `--build-manifest` together with
`--base-build-receipt`; omission leaves the run diagnostic-only. It replays
the exact `trtmc.dynamic-memory-test-manifest/v2` validator and the fresh
native-dynamic build-receipt validator instead of trusting copied hashes.
The resulting
`trtmc.native-dynamic-memory-base-artifact-binding/v1` evidence binds the
current clean `git_head` and `source_state_sha256` to the supplied base bundle,
qualifier runner, benchmark worker, core DSO, active versioned TensorRT
backend alias and inode, selected Qwen or Llama model DSO, and runtime-KV
plugin. Promotion reopens this evidence after the qualification matrix;
missing, stale, cross-model, dirty-source, replaced-inode, or mismatched
manifest/receipt evidence forces both `passed=false` and
`promotion_eligible=false` while preserving otherwise valid diagnostic
results. Before importing the Python TensorRT plugin loader, the qualifier
binds `TRTMC_TRT_PLUGIN_LIBRARY` to that exact manifest identity and rejects a
different explicit environment selection or competing preloaded ABI DSO.
Immediately after runtime-stack query and engine inspection it resamples
`/proc/self/maps`, requires one non-deleted mapping with the same canonical
path/device/inode, reopens the selected file identity, and persists
`trtmc.native-dynamic-memory-runtime-kv-plugin-binding/v1`.
`runtime_kv_plugin_binding_passed` is independently replayed by both canonical
correctness promotion and the process-isolation aggregate.

### 9.3 Memory tests

Every receipt emits the following schema fields; unavailable observations are
explicitly `null` rather than fabricated:

```text
serialized_plan_bytes
resident_weight_bytes
resident_weight_copy_count
engine_weight_bytes
capacity_decision_free_bytes
capacity_decision_total_bytes
capacity_decision_device_used_bytes
settled_free_bytes
settled_total_bytes
settled_device_used_bytes
context_device_memory_bytes
ordinary_device_input_bytes
ordinary_device_output_bytes
external_device_output_bytes
host_staging_bytes
graph_private_device_bytes
kv_reserved_bytes
kv_committed_bytes
kv_metadata_bytes
peak_device_bytes
```

`context_device_memory_bytes` is the single value returned by
`updateDeviceMemorySizeForShapes()` and includes TensorRT internal activation,
runner scratch, and plugin workspace. Do not claim exact sub-breakdowns that
TensorRT does not expose. External outputs, host staging, graph-private
allocations, and application-owned KV remain separately measurable.

The capacity-decision snapshot binds the automatic fraction formula and the
resolved `R`; the settled snapshot binds actual memory after the final
context, output, and KV allocations. The latter may show more free memory
after a downward envelope replacement, but it cannot increase `R`. The
deprecated `final_*` fields equal `capacity_decision_*` byte-for-byte and must
not be interpreted as settled residency.

`ordinary_device_input_bytes` and `ordinary_device_output_bytes` are the
per-module/context high-water device capacities materialized for ordinary
dynamic TensorRT I/O after the post-load snapshot. They exclude static I/O,
deferred KV/present bindings, and context device memory.
`external_device_output_bytes` includes only the runtime-owned current-K/V
staging capacity, `min(C,R) * B`, and remains exactly measurable even when
engine introspection is unavailable. The concrete output bindings and trace
report only `Sq * B` active/write bytes for an invocation. The load-time
receipt may not yet have a request-completion high-water sample;
qualification requires a non-null `peak_device_bytes` after at least one
successful request, with its sample count, boundary, scope, and source
recorded.

Required gates:

| ID | Test | Required result |
|---|---|---|
| MEM-01 | Allocation timeline | KV policy resolves only after engine weight memory and queried/reserved non-KV overhead are known. |
| MEM-02 | Default auto | Uses 90% of safe post-load free memory, capped by `M`. |
| MEM-03 | Percentage/bytes | Allocation stays within policy; rounding loses at most one row. |
| MEM-04 | Large budget plus small `U` | Allocate only `min(R, U)` token rows. |
| MEM-05 | Small runtime capacity in full-`M` bundle | Allocation equals runtime `R`, with no profile-MAX cache allocation. |
| MEM-06 | Prompt history grows from `C` to 32K/40K | For each execution role, context-block growth stays below its measured smallest-`A` baseline plus at most two BF16 `[Hq,C,A]`-equivalent surfaces. Every measured point is also below one eighth of a full BF16 `[Hq,M,M]` score tensor. A full run covers prefill, decode, `A=M`, and at least three distinct active lengths. |
| MEM-07 | Controlled external GPU reservation | `auto` derives a smaller `R`; the same bundle still runs if the request fits. |
| MEM-08 | 100 sequential requests | No monotonic device-memory growth. |
| MEM-09 | 20 load/unload cycles | Device memory returns to baseline tolerance. |
| MEM-10 | Injected allocator with red zones | Correct size/alignment/lifetime; no out-of-bounds writes. |
| MEM-11 | Two `R` values with one bundle | KV allocation delta equals `(R2 - R1) * B` within allocator rounding. |
| MEM-12 | Active `A` grows while physical `R` is fixed | Both paths use cold `T=1` or `T=P=min(ceil_bucket(H), R)`; allocation ID, base pointer, and reserved KV bytes stay fixed across history-bucket transitions. |
| MEM-13 | Engine packaging A/B | Shared-engine path has one resident weight copy. Separate-plan fallback has at most two and does not exceed the exact-head static split baseline's plan/resident-weight bytes by more than 5%; weight streaming must be disabled. |
| MEM-14 | Context memory receipt | Reported `context_device_memory_bytes` equals the selected maximum `updateDeviceMemorySizeForShapes()` result; no invented activation/workspace split. |

MEM-06 is evaluated only from actual-shape
`updateDeviceMemorySizeForShapes()` observations. For each role independently,
let `baseline_A` be the smallest measured active length and
`baseline_bytes` the largest context requirement observed at that
`baseline_A`. Every point must satisfy:

```text
context_device_memory_bytes
  <= baseline_bytes
     + 2 * C * (A - baseline_A) * Hq * sizeof(bfloat16)

context_device_memory_bytes
  < sizeof(bfloat16) * Hq * M * M / 8
```

The first inequality leaves room for one primary and one auxiliary fused
surface while retaining `O(C*A)` growth. The second is applied to every
measured point, not only the full-context point. Default full-matrix
qualification additionally requires both prefill and decode observations,
an observation reaching `A=M`, and at least three distinct `A` values. A
case-filtered developer run may report partial diagnostics, but it cannot
stand in for this full-coverage gate.

MEM-07's controlled external-pressure proof is assigned to Qwen, whose
full-context KV allocation is large enough for a GB300 reservation to move
`R` below `M`. TinyLlama still runs MEM-08/09/11 and all explicit
auto/percentage/bytes/U policies, but its complete 2,048-row BF16 KV slab is
only 46,137,344 bytes. A large-memory device can retain more unreservable CUDA
tail headroom than that slab, so requiring controlled pressure to reduce
TinyLlama below `M` would test allocator tail granularity rather than this
runtime planner.

MEM-13 compares bundles built from the same source snapshot, pinned model
revision, precision, target, TensorRT/CUDA/cuDNN toolchain, and documented
no-flag product build flow. The receipt records the final topology and obtains
resident weight bytes from TensorRT engine statistics, not bundle-size
proxies. When a qualified checkpoint declares tied embeddings, a plan may
reuse the existing embedding tensor for its transposed LM-head operand only
after the mapped `w_out` and `embedding.T` have identical shape, dtype, and
values. A mismatch fails the qualified build; it may not silently alias
independent weights. The real-plan accounting gate, not source inspection,
must prove that this removes the duplicate resident vocabulary matrix.

For contiguous-v1:

```text
kv_reserved_bytes == R * B
backend_owned_cache_input_bytes == 0
backend_owned_cache_output_bytes == 0
```

The final field means that TensorRT owns no persistent or full-history cache
output. It does not exclude the separately reported, runtime-owned bounded
current-row staging allocation.

NVML peak is supporting evidence, not the sole memory source. For every
non-admission correctness or policy case, the developer-only qualification
runner executes exactly two load lifetimes in one process and installs the
internal runtime-device observer for both:

1. one explicitly labelled cold-start lifetime; and
2. one measured lifetime after the cold-start pipeline has been destroyed.

The two lifetimes use the same typed runtime policy and the same request.
Prompt length, prefill/decode launch counts, final KV position, selected token
IDs, step top-1 IDs, and the complete float32 logits must be bitwise identical.
The runner writes distinct cold and measured complete-float32 artifacts. The
Python producer independently reads both payloads, verifies their headers and
paths, compares every payload byte, derives the top-1 and selected-token IDs,
and records both SHA-256 digests. Runner-reported equality booleans are
diagnostic corroboration, not the promotion gate. The measured logits are
compared with HF only after that cold-versus-measured gate passes.
Admission-rejection cases run neither lifetime, emit an empty invocation/token
ledger plus `attention_started=false`, and must still reject before attention.

Each synchronized boundary records:

```text
D = signed cudaMemGetInfo device-used change
P = signed NVML current-process change
X = signed NVML non-current device-used change
U = D - P - X
```

The runner records the complete visible compute-process ledger, current and
all-process bytes, NVML v2 device total/reserved/free/used values, and a second
CUDA free/total sample after the NVML calls. Each lifetime requires exactly
one pre-engine baseline, exactly one post-KV-allocation boundary, and exactly
one successful-request-completion boundary. Sampler PID, logical/physical GPU,
PCI bus ID, and GPU UUID are typed and bound to every phase sample; the current
PID must appear exactly once in every complete NVML process ledger. The
before-load and after-request endpoints reconcile with the first and last
synchronized phase samples within the same 64 MiB bound. The receipt's
pre-load and post-load samples, capacity-decision sample, and settled
post-KV-allocation sample bind their exact synchronized phase rows. The
device-wide peak
reconstructed from those synchronized rows must equal that lifetime's
structured receipt byte-for-byte. At every cold and measured boundary, both
the unexplained residual `U` and the CUDA/NVML sampling bracket must remain
within `max(64 MiB, 2%)`. The NVML device delta not represented by the visible
process ledger meets that same bound except for the explicitly cross-bound
cold driver/JIT allocation and its later release described below. A cold-start
observation is not made non-gating merely by labelling it warm-up.

The cold-start unload boundary must reconcile independently. One-time
process-global TensorRT/CUDA initialization retained into the measured
baseline is capped at `max(512 MiB, 5% of resident_weight_bytes)`, recorded
explicitly, and included in the measured lifetime's pre-engine baseline. This
floor is evidence-based rather than an allocation request: three isolated
TinyLlama processes retained exactly 400,556,032 process bytes and isolated
Qwen retained 513,802,240 bytes after its first load/unload. A cold-only
driver/JIT allocation that NVML device accounting does not charge to the
process ledger is allowed only when the visible other-compute-process ledger
is empty, its signed delta persists from both cold peak boundaries through
cold unload, and it is at most 2 GiB. The measured lifetime may retain at most
64 MiB of process memory after unload; a negative unlisted delta is accepted
only as an explicitly bounded release of that previously proven cold driver
allocation. The cold unload and measured pre-load samples must otherwise
reconcile under the same `max(64 MiB, 2%)` attribution rule. Missing
attribution fields, duplicated boundaries, unstable brackets, output drift,
excessive retention, or a larger unexplained residual fail closed. Full-GPU
qualification is required; MIG is rejected until CUDA-instance-to-NVML-instance
attribution is implemented.

The Python producer performs TensorRT engine inspection before it launches a
runner, so canonical qualification isolates those two processes on different
physical GPUs. The producer is started with a single incoming
`CUDA_VISIBLE_DEVICES=<producer>`; the required internal
`--runner-cuda-visible-device <runner>` option accepts exactly one physical
index or full GPU UUID and overrides that variable only in the C++ runner
child. The producer's incoming selector and the runner selector are recorded
separately. Every runner trace must still bind logical device zero to the
expected physical index, PCI bus ID, full UUID, and child PID, and its complete
NVML ledger must contain no producer process. This separation removes
qualification-induced GPU occupancy without weakening the exclusive-compute
or memory-reconciliation gates.

The producer writes each runner command, deterministic token input, stdout,
stderr, return code, raw logits, and raw trace before validation. Both
per-case validation failures and post-case envelope/source/outcome failures
write a final failed report with the precise stage and post-failure source
state. A failed run may not disappear with its temporary directory or leave a
stale `status=running` report.

For MEM-07 request-completion headroom, the calibration guard remains
conservative: it uses the maximum of the synchronized device-wide free-memory
delta and the current-process NVML delta. The constrained request's hard
ownership gate, however, compares current-process NVML growth against the
calibrated current-process growth plus the stated tolerance. A simultaneous
device-wide delta that is not present in the current process is recorded
explicitly as external pressure and is not attributed to the pipeline. The
receipt must include both deltas, their signed difference, and the guard and
hard-gate bases; omitted attribution fields fail closed.

The reservation is first placed immediately before planning, but convergence
is decided only after the runtime has allocated its selected context and
output memory. A two-sided controller then allocates aligned tail chunks when
free memory is high or releases one complete aligned tail when it is low. It
has at most 64 correction attempts, rejects a repeated/no-progress state, and
must land in the exact half-open 2 MiB window selected for the target
capacity. The contiguous KV guard stays live through the runtime's actual
capacity-decision CUDA snapshot and is released only by the post-snapshot
observer. The receipt's `capacity_decision_free_bytes` (and deprecated
`final_free_bytes` alias) must equal that snapshot byte-for-byte, and an
independent validator recomputes `R` with the runtime binary64-fraction
formula. The later settled sample records the final context/output/KV
residency after guard release and cannot change `R`. A preplanning sample, a
controller sample, or a receipt that merely approximates the
capacity-decision snapshot is not sufficient.

### 9.4 Split execution tests

| ID | Required proof |
|---|---|
| SPLIT-01 | Bundle/engine exposes distinct prefill and decode roles/profiles. |
| SPLIT-02 | A prompt `<= C` performs one prefill invocation, not token-by-token prefill. |
| SPLIT-03 | A long prompt performs exactly `ceil(L/C)` prefill invocations. |
| SPLIT-04 | Prefill and decode report the same KV allocation ID/base pointer. |
| SPLIT-05 | K/V D2H bytes are zero. |
| SPLIT-06 | Append write traffic is proportional only to new rows; no cache concatenation/full-history D2D appears in the trace. |
| SPLIT-07 | `P-1/P/P+1` selects a valid bucket without re-prefill. |
| SPLIT-08 | Decode throughput is at least 95% of the exact-head static split baseline. |
| SPLIT-09 | Short/medium prompt TTFT regression is no more than 10%. |

Trace records must include plan/profile ID, chunk range, launch count, KV
allocation ID, `H/A/T/R`, CUDA-graph entry or uncaptured status, and transfer
bytes.

### 9.5 Negative and error tests

Reject before attention execution:

- `0%`, a percentage greater than `100%`, negative values, unknown units, and
  integer overflow;
- duplicate or conflicting memory policies;
- a budget smaller than one token row;
- an explicit-byte policy whose semantically useful resolved allocation is
  larger than safe available memory;
- `U > M`, reported as exceeding model capability;
- a model-valid requested length that does not fit the runtime budget,
  reporting capacity tokens, required bytes, and budget bytes;
- `prompt + max_new_tokens > effective_request_limit`, reporting all values;
- a contract claiming `M` that its engine profiles cannot cover;
- dynamic-memory options on an old/static bundle;
- multi-lane dynamic use before device-pool partitioning is implemented.

Never report an internal 4,096 profile cap for a Qwen request. If the Qwen
engine cannot execute a model-valid 40K request, the bundle is unqualified.

## 10. CI and qualification

| Layer | Coverage | Gate |
|---|---|---|
| PR CPU | CLI parsing, budget properties, overflow, admission, metadata, routing, legacy contract, public-help surface | Required |
| PR synthetic TensorRT GPU | Dynamic shapes, user-managed context, external input/output binding, red zones, fused attention, no dense mask | Required |
| PR exact-head model proof | Two no-flag builds, short HF parity, bucket boundaries, two runtime budgets, SHA receipts | Required |
| Nightly GB300 | Qwen 32K/40K, TinyLlama 2K, memory slope, pressure, performance, soak | Required for promotion |
| Release qualification | Every advertised SM/TensorRT/CUDA/cuDNN/Frontend/NVRTC/driver tuple | Required for release claim |

Primary implementation platform:

```text
NVIDIA GB300
TensorRT 11.2.0.113
CUDA runtime 13.3
NVRTC 13.3
NVIDIA driver 580.105.08
cuDNN backend 9.20.0
cuDNN Frontend 1.21.0
  commit 7b9b711c22b6823e87150213ecd8449260db8610
```

### 10.1 Source-bound local qualification workflow

The review receipt root is ignored build output, not committed source:

```text
artifacts/dynamic-memory-qualification/<source-snapshot-id>/
```

Freeze source first, then use the performance producer for every dynamic
bundle and exact-head static split baseline. The dynamic user command nested
inside the producer remains exactly:

```bash
trtmc build Qwen/Qwen3-0.6B
trtmc build TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The producer's own options select receipt paths and trusted metadata; they
are not model build options. Its build action:

1. refuses to overwrite an existing bundle;
2. snapshots `HEAD`, staged/unstaged binary patches, and every non-ignored
   untracked source digest;
3. executes the exact argv itself;
4. requires the source snapshot to be unchanged afterward;
5. records command, log, bundle, source, and tool hashes.

The historical optional-output failure also has an executable, fail-closed
diagnostic. It deliberately retains the qualified product stack
(CUDA runtime 13.3, cuDNN 9.20, Frontend 1.21, and the primary GB300 driver)
while pinning only the Python CUDA 13.0 NVRTC and matching builtins pair:

```bash
python tools/qualify_native_dynamic_memory_nvrtc_regression.py \
  --probe build-dynkv/trtmc_nvrtc_optional_output_regression \
  --nvrtc \
    /opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13 \
  --nvrtc-builtins \
    /opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc-builtins.so.13.0 \
  --output \
    artifacts/dynamic-memory-qualification/<source-snapshot-id>/nvrtc-13.0/qualification-receipt.json
```

This is a root-cause negative replay, not a supported CUDA 13.0 production
tuple. The producer uses two fresh processes and two private, initially empty
CUDA caches. It independently verifies that the exact Qwen `Sq=1, T=512`
legacy graph requests `Max/O/Sum_exp`, selects `eng3_k24=7`, fails with the
expected NVRTC compilation error, and never selects a fallback. The second
process proves that the standard `O/Stats` LSE graph contains none of those
legacy outputs, builds, executes, synchronizes, and returns finite results.
Before releasing either process, the producer reopens `/proc/<pid>/maps`,
rejects competing or deleted CUDA/cuDNN/NVRTC mappings, and records the
device/inode/path/SHA-256 identity of every mapped runtime component. It also
binds both parsed cuDNN graph artifacts, the probe binary, driver evidence,
logs, and pre/post source snapshots.

`passed=true` means that this isolated diagnostic contract passed.
`promotion_eligible=true` additionally requires one unchanged clean exact
HEAD. A passing dirty-source diagnostic remains useful for review but cannot
promote a release, does not authorize the legacy fallback, and does not
advertise CUDA 13.0 as a supported runtime.

For each static/dynamic short/medium case, the benchmark action:

1. verifies the request points at the bundle in the fresh-build receipt;
2. hashes the semantic request plus the matched effective sequence limit;
3. executes `trtmc_benchmark_worker`;
4. independently deserializes both split engine sections;
5. measures serialized plan bytes, total resident weights, engine-copy count,
   and weight-streaming state from TensorRT;
6. cross-checks the independent values against the dynamic pipeline receipt;
7. parses and hashes the product worker's validated live runtime-stack row,
   every selected LSE attention plan, and the before/after CUDA JIT-cache
   state; missing, malformed, conflicting, non-LSE, or
   `COMPILATION_FAILED` evidence fails closed;
8. writes one enriched worker result for
   `qualify_native_dynamic_memory_perf.py`.

The final performance gate requires, for both prompt sizes and both models:

```text
dynamic decode throughput >= 95% of exact-head static
dynamic prefill proxy <= 110% of exact-head static
dynamic bundle/plan/resident weights <= 105% of exact-head static
resident split-engine weight copies <= 2
weight streaming disabled
all four cases share source/model/target/toolchain/environment provenance
```

Workload equivalence is a hard gate, not an inference from equal output-token
counts. Each capture must use fixed-length greedy autoregressive generation:
`generation_mode=ar`, `temperature=0`, `top_k=1`, `top_p=1`, `min_p=0`, one
sample, an unreachable `INT32_MAX` EOS, no chat template, no boxed-answer
early stop, generated-token capture enabled, and exactly `max_new_tokens`
outputs in every measured iteration. The prompt/generation/measurement
structure is hashed, and the static/dynamic pair for a prompt size must have
the same structural identity. All measured iterations within each case must
produce one repeatable token stream.

The tokenizer contract is independently bound to the actual bundle payload:
the `tokenizer.json` bytes/SHA, add-special-tokens setting, and declared
special prefix/suffix token IDs must be present and identical across all four
performance captures. A missing field, tokenizer mismatch, structurally
different request, early EOS, short output, or within-case token
nondeterminism fails the performance qualification.

Static-versus-dynamic generated token-ID equality is intentionally diagnostic,
not a performance hard gate. Numerically acceptable logits can choose
different greedy tokens after a near tie; the separate full correctness
receipt remains responsible for HF logit/token qualification. The performance
report therefore records exact token-stream hashes, equality, and common
prefix length without silently converting divergence into a timing failure.
Likewise, the captured cuDNN graph-build/cache rows prove source-bound graph
identities for the worker and runtime stack, but they are not per-invocation
records. Until the worker emits that trace, performance evidence does not
claim each measured decode step's `H`, `A`, `profile_id`, or selected cuDNN
plan identity; that limitation remains explicit in
`diagnostics.runtime_attention_plan_scope`.

Old worker JSON, missing accounting, a changed request or bundle SHA, source
drift, an overwritten/reused artifact, or unavailable resident-weight
measurement fails closed.

After the full correctness and performance reports pass, produce the v2
process-isolation aggregate with both companion receipts as explicit inputs:

```bash
python tools/capture_native_dynamic_memory_process_isolation.py \
  --repo-root . \
  --bundle <native-dynamic.trtfb> \
  --build-receipt <native-dynamic-build-receipt.json> \
  --request <fixed-greedy-isolation-request.json> \
  --correctness-report <correctness/qualification-report.json> \
  --performance-report <performance/performance-gate.json> \
  --worker build-dynkv/trtmc_benchmark_worker \
  --plugin-library build-dynkv/libtrtmc_trt_plugins.so \
  --comparison-sequence-limit <M> \
  --gpu-a <physical-gpu-a> \
  --gpu-b <physical-gpu-b> \
  --output-dir \
    artifacts/dynamic-memory-qualification/<source-snapshot-id>/process-isolation
```

The aggregate regenerates the performance gate from its four raw captures and
two bundle files, validates the complete canonical correctness matrix, and
requires the companion reports, logits/logs, captures, and mapped runtime
libraries to remain hash-stable throughout the run. Each of the four child
captures must match the same model revision, bundle SHA, clean exact-HEAD
source digest, live runtime stack, and mapped NVRTC/NVRTC-builtins identity.
The report retains the direct-versus-aggregate claim boundary stated under
QUAL-03.

Each hardware receipt contains:

- a clean source commit SHA, or, for an explicitly dirty review snapshot, the
  commit SHA plus complete staged/unstaged patch and non-ignored untracked
  source digest manifest; the pre-build and post-proof snapshot digests must
  match;
- bundle SHA and model revision;
- GPU, SM, TensorRT, CUDA, and driver versions;
- cuDNN backend version and cuDNN Frontend revision;
- resolved NVRTC and NVRTC-builtins paths, versions, and SHA-256 digests;
- selected execution-plan identity for every exercised `Sq/T` geometry and
  whether its JIT cache was cold or warm;
- exact build and runtime commands;
- `M/C/B/P/R/U` plus the exercised `H/A/T` ranges;
- structured memory receipt and peak trace;
- HF comparison artifact;
- split execution trace;
- performance samples and baseline SHA.

Skipped tests, old-head bundles, or thresholds relaxed to obtain a pass do not
count as qualification. Promotion requires three consecutive successful
nightlies on the primary platform.

## 11. Error and observability contract

Example successful load:

```json
[trtmc.memory] {"policy":"auto","policy_fraction":0.90,
"model_context_limit":40960,"prefill_chunk_limit":1024,
"post_load_free_bytes":292367106048,"safety_reserve_bytes":67108864,
"kv_bytes_per_token":114688,"kv_budget_bytes":262798206566,
"runtime_kv_capacity_tokens":40960,"kv_reserved_bytes":4697620480,
"effective_request_limit":40960}
```

Example resource error:

```text
Requested total sequence length 32768 is supported by the model (40960)
but exceeds this runtime KV capacity (28672 tokens).
Required KV memory: ...
Configured KV budget: ...
Increase --kv-cache-memory or reduce --max-sequence-length.
```

Example semantic error:

```text
Requested max sequence length 65536 exceeds the model context limit 40960.
```

Do not expose a builder profile number as a model capability error.

## 12. Pull request sequence

Do not publish the current prototype as one large PR.

### PR 1: TensorRT memory plumbing

Scope:

- user-managed context allocation;
- actual-shape memory query;
- external dynamic input/output bindings;
- common segmented `NativeContiguousAttention` plugin version/ABI 2;
- exact-shape current-K/V staging and current-row-only commit;
- shared context device-memory block;
- ABI/version handshake;
- synthetic TensorRT tests.

No public behavior change and no model-specific routing. Phase 0 experiment
receipts are attached to this PR; it does not merge until the graph/plugin and
engine-packaging choices are resolved.

### PR 2: Runtime-memory contract and Qwen vertical slice

Scope:

- bundle metadata;
- qualified Qwen revision/config/platform profile;
- CLI/runtime policy;
- shared planner;
- Qwen segmented fused attention, chunked prefill, and current-row KV commits;
- Qwen 32K/40K evidence.

Capability remains Qwen-native and version-gated.

### PR 3: Llama generalization and beta UX

Scope:

- TinyLlama graph/runtime wiring;
- qualified TinyLlama revision/config/platform profile;
- shared-contract cleanup;
- cross-family tests;
- C ABI V2 and Python policy parity;
- no-flag build as the documented UX;
- compatibility, docs, and promotion receipts.

This PR enables the two-model beta only after the Qwen Phase 2 gates and every
Phase 3 cross-family/beta gate pass.

### PR 4: Paged KV and multi-request pool

Separate follow-up:

- page/block manager;
- paged attention kernel/plugin;
- `PipelinePool` integration;
- concurrency, fragmentation, cancellation, and soak tests.

Each PR must be independently reviewable, preserve existing static bundles,
and contain no EdgeLLM adapter changes.

## 13. Risk register

| Risk | Mitigation / stop rule |
|---|---|
| cuDNN segmented SDPA does not fuse or expose stable LSE for a target dtype/head shape | Fail Phase 0; do not accept a decomposed quadratic fallback. |
| A selected `Sq=1` plan fails NVRTC finalization or selection changes across cold/warm processes | Keep one coherent packaged CUDA stack, but do not claim that RPATH fixes the former optional-output failure: forced CUDA 13.0 NVRTC did not. Record the selected plan/cache state and require every selected standard-LSE plan to pass deterministic parity/performance, or replace decode with a qualified fused kernel. |
| TensorRT 11.2 omits PluginV3 alias metadata | Keep the proposal as a negative test; never mutate a const input or trust bundle names as a substitute. |
| `IKVCacheUpdateLayer` requires static cache `T` | Keep the failure executable; use read-only segmented attention and current-row runtime commit. |
| Marked-and-consumed current K/V causes profile-MAX output allocation | Fail Phase 0; do not route either target until exact-shape external staging is proven. |
| Two-segment attention regresses decode or TTFT | Optimize graph reuse/merge or stop the beta; never reintroduce full-history concatenation. |
| Shared engine profiles compromise decode tactics | Use the measured separate-plan fallback; record duplicate weight memory. |
| Actual-shape context memory changes as history grows | Size one shared context block for enabled runtime buckets; never use profile-global MAX allocation. |
| Physical capacity `R`, active total `A`, history length `H`, and bound history extent `T` are confused | Centralize the Section 4 cold-sentinel, `T<=R`, and `A=H+Sq<=R` byte-capacity checks in the backend and cover with red-zone tests. |
| Free memory changes between measurement and allocation | Re-query once; auto may recompute, explicit bytes fail with requested/available detail. |
| CUDA graph capture turns buckets into a correctness cap | Always retain uncaptured `enqueueV3` fallback. |
| New API fields break mixed DSOs | Require explicit contract and struct-size/version checks. |
| Qwen passes but Llama diverges | Shared planner plus independent TinyLlama exact-head parity is a release gate. |
| PipelinePool multiplies the requested budget per lane | Fail fast until the device-level paged pool lands. |
| Build time or engine size grows excessively | Track plan bytes/build seconds per profile and remove only unhelpful performance buckets, never reduce `M`. |

## 14. Review budget

The original 3K-5.4K estimate was not accurate once executable TensorRT
negative proofs, ABI/DSO compatibility, exact long-context qualification, soak
and provenance tooling were made release gates. Do not use that estimate to
describe the review snapshot.

Current candidate inventory against `github/main`, after staging the audited
source and before generated receipts:

| State | Files | Added | Deleted | Churn |
|---|---:|---:|---:|---:|
| Tracked diff | 176 | 70,868 | 733 | 71,601 |
| Non-ignored untracked source | 0 | 0 | 0 | 0 |
| Total review surface | 176 | 70,868 | 733 | 71,601 |

The final numbers include all tracked, staged, unstaged, and non-ignored
untracked source. They exclude generated engine plans and hardware receipts
under the ignored artifact root.

This is not acceptable as one publishable PR. Section 12 is therefore a hard
land-order, not a suggestion. Before opening PR 1, move each phase to its own
short-lived branch and recompute its isolated `git diff --stat`; no PR should
contain later-phase model enablement or qualification assets merely because
they coexist in this prototype workspace.

## 15. Final exit criteria

The two-model native-runtime milestone is complete only when all statements are
true:

- users build both models with only `trtmc build <model>`;
- a qualified bundle fails before engine deserialization unless the live
  target matches its complete advertised
  SM/TensorRT/CUDA/cuDNN/Frontend/NVRTC/driver tuple; the prototype's minimum
  guard is `sm103 + TensorRT 11.2.0.113`, and bundle-declared target text is
  not accepted as evidence of the current runtime;
- one bundle per model supports `auto`, percentage, bytes, and runtime sequence
  policy without rebuild;
- Qwen's bundle advertises and executes its 40,960 model limit, including a
  qualified 32K request and a last-position 40,960-token prefill;
- TinyLlama executes a last-position 2,048-token prefill and rejects exactly
  beyond its model boundary;
- non-KV context memory is queried from actual `Sq/T` shapes, with measured
  growth reported; no profile-`M` preallocation or dense `O(L^2)` path exists;
- no dense causal mask or materialized full attention score exists;
- every invocation binds history extent `T` with the Section 4 cold-sentinel
  invariant, `T<=R`, and `A=H+Sq<=R` validated;
- a read-only segmented attention plugin plus post-engine commit touches only
  new KV rows;
- TensorRT does not allocate a duplicate profile-MAX KV input/output buffer;
- KV allocation equals the runtime-selected capacity and is shared by prefill
  and decode;
- no K/V data is copied to host or copied in proportion to full history;
- split execution meets correctness and performance gates;
- cold-cache, warm-cache, and concurrent different-GPU process loads directly
  prove isolated cache behavior, overlapping engine loads, and deterministic
  child token IDs without a shared lock or seeded JIT cache; the v2 aggregate
  additionally binds full HF parity and SPLIT-08/09 companion receipts to the
  identical source/bundle/runtime tuple without claiming those computations
  ran inside every child;
- performance evidence is produced from fresh source-bound builds and real
  worker executions, with independent TensorRT engine accounting rather than
  hand-enriched JSON;
- errors distinguish model capability from current runtime resources;
- existing static bundles remain compatible;
- `PipelinePool` fails safely until shared-pool support exists;
- no EdgeLLM adapter, EdgeLLM runtime DSO, or EdgeLLM test file is modified;
- exact-head commands, bundle hashes, comparison JSON, memory receipts, and
  performance traces are attached to the review.

Only after these gates pass should the project begin the paged-KV concurrency
phase or make claims beyond 40K.

## 16. Review handoff and promotion boundary

The implementation is intentionally reviewable before it is promoted as a
general feature:

- product routing is limited to the two exact qualified model tuples;
- the normal build help exposes no context/KV/profile builder control;
- runtime policy is available through CLI, C ABI V2, and Python;
- common backend/plugin/planner code is shared, while Qwen/Llama retain only
  model-owned tensor and generation behavior;
- production registers only `NativeContiguousAttention` ABI 2;
  `NativeKvAppend` is an uninstalled test fixture;
- the V2 backend/model-plugin handshake intentionally rejects legacy
  out-of-tree DSOs; those DSOs must be rebuilt against the matching SDK before
  this prototype can be promoted;
- all EdgeLLM adapter/runtime/test trees remain outside the diff.

One frozen release-candidate source state must produce all of these green
baselines. A dirty diagnostic, even when every numerical and timing threshold
passes, cannot satisfy this requirement. The receipt stores the command,
collected test manifest, count, and output for each command:

```bash
python tools/capture_dynamic_memory_test_manifest.py \
  --build-dir build-dynkv \
  --python /opt/venv/bin/python \
  --output-dir artifacts/dynamic-memory-qualification/<source-snapshot-id>/tests
```

That producer owns and executes this fixed command set:

```bash
cmake --build build-dynkv --clean-first -j
cmake --build build-dynkv -j --target \
  trtmc_cpp_tests \
  trtmc_dynamic_memory_qualify \
  trtmc_dynamic_memory_surfaces \
  trtmc_benchmark_worker
ctest --test-dir build-dynkv -N
ctest --test-dir build-dynkv --output-on-failure
ctest --test-dir build-dynkv -N -L dynamic_memory
ctest --test-dir build-dynkv -L dynamic_memory --output-on-failure

python -m pytest --collect-only -q -m dynamic_memory \
  tests/builder tests/tools tests/e2e/test_native_dynamic_memory_graph.py
python -m pytest -q -m dynamic_memory \
  tests/builder tests/tools tests/e2e/test_native_dynamic_memory_graph.py \
  --junitxml=<receipt-dir>/pytest_dynamic_memory.junit.xml
python -m pytest -q tests/e2e/test_native_dynamic_memory_graph.py \
  --junitxml=<receipt-dir>/pytest_graph_e2e.junit.xml
```

This remains a fixed nine-command set. The second build command constructs
`trtmc_nvrtc_optional_output_regression` transitively through
`trtmc_dynamic_memory_qualify`; the diagnostic is not a tenth command. The v2
test manifest reopens the probe together with every other build artifact and
records its canonical path, build-relative path, size, mode, device, inode,
mtime, and SHA-256. Every pytest command is also bound to the benchmark worker
and `libtrtmc_trt_plugins.so` from that same build directory. Therefore the
CUDA-13.0 negative replay in Section 10.1 must consume the probe recorded by
this exact manifest and must produce its own source-stable
`qualification-receipt.json`. Its `passed` component result is not promotion
evidence unless both receipts independently satisfy their clean exact-HEAD
gates.

The producer matches every collected `dynamic_memory` node ID to its JUnit
outcome and fails on a missing, failed, errored, or skipped selected test. A
module-level collection skip outside that selected manifest is retained as a
diagnostic but cannot substitute for, or reduce, the selected test count.

The same snapshot must then run the live no-flag TinyLlama build/run through
the complete target guard and fresh source-bound no-flag build, correctness,
memory, soak, surface, and performance producers for both models.

TensorRT-RTX headers/runtime are not present in the primary GB300 development
container, so the RTX runtime path cannot be claimed as hardware-qualified
from this snapshot. Its source/CI compatibility guard remains required and
the dynamic interface must continue to fail before accidental RTX adoption.
Release promotion additionally requires three consecutive primary-platform
nightlies and a clean installed-package two-process proof. Those external
promotion gates do not broaden this milestone's two-model scope.

## 17. Primary TensorRT references

The implementation must be checked against the headers and samples in the
repository's pinned TensorRT 11.2.0.113 environment. These current NVIDIA
references define the APIs and constraints used by this plan:

- [cuDNN Frontend SDPA inference sample](https://github.com/NVIDIA/cudnn-frontend/blob/7b9b711c22b6823e87150213ecd8449260db8610/samples/cpp/sdpa/fp16_fwd.cpp):
  the pinned standard `set_generate_stats(true)` LSE contract used to merge
  history and current segments without materializing scores;
- [TensorRT KV cache](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-kv-cache.html):
  `IKVCacheUpdateLayer` shape, linear mode, application-owned memory, and
  input/output alias requirement;
- [IExecutionContext C++ API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_execution_context.html):
  `inferShapes()`, `updateDeviceMemorySizeForShapes()`,
  `setDeviceMemoryV2()`, tensor addresses, and output allocators;
- [ICudaEngine C++ API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_cuda_engine.html):
  `kUSER_MANAGED` execution-context creation;
- [IPluginV3 API and migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html):
  `IPluginV3OneBuildV2`, `getAliasedInput()`, serialization, registration, and
  no-allocation enqueue guidance;
- [TensorRT fused attention](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-fused-attention.html):
  supported fused-attention configurations and qualification constraints.
