# Llama speculative decoding design

Scope: FP16 Llama 3.1 8B Instruct with its EAGLE3 draft, one request on one GPU,
greedy chain/tree verification and fixed-capacity linear KV state. Attention
uses TensorRT graph primitives; paging and sampled speculation are not implemented.

## Separation of responsibilities

The compiler/runtime contract includes model I/O, state effects and execution
constraints. KV is persistent mutable state; target/draft features are ordinary
model I/O with explicit alignment and lifetime.

[Engine](runtime/speculative/engine.h) owns bindings, state storage and accepted-row
copies. [EAGLE3](runtime/speculative/eagle3.h) owns conditioning, proposals and
feedback; the pipeline owns verification and request progress. Another method
can reuse the engine contract if its state and I/O semantics fit. The contract
does not require token-by-token drafting or claim to cover recurrent-state models.

## Engine ABI v1

[EngineContract](speculative/contract.py) is serialized per target/draft engine in
`speculative.json`. For query rows `Q`, selected logits rows `M`, capacity `C`,
KV heads `Hkv` and head dimension `D`, bindings are dense, contiguous device
tensors. Tensor names, row order and alias requirements are part of the ABI.

| Binding | Type/shape | Semantics |
|---|---|---|
| `token_id` | input INT32 `[Q]` | Target-vocabulary IDs, including for draft embedding lookup. |
| `position_id` | input INT32 `[Q]` | Logical RoPE positions; siblings share a position even in different physical slots. |
| `cache_write_indices` | input INT32 `[1]` | Start `s`; query row `r` writes physical slot `s+r`. |
| `key_value_lengths` | input INT32 `[1]` | Initialized span `s+Q`, including tentative rows; not accepted length. |
| `attention_mask` | input INT32 `[Q,C]` | 1=visible, 0=hidden. Committed prefix plus inclusive ancestors for each query. |
| `logits_indices` | input INT32 `[M]` | Query rows selected for final norm and LM head, in requested order. |
| `cache_k_i`, `cache_v_i` | input FP16 `[1,Hkv,C,D]` | Runtime-owned layer state; cached K has already undergone RoPE. |
| `present_k_i`, `present_v_i` | output FP16 `[1,Hkv,C,D]` | Alias corresponding cache inputs; only slots `[s,s+Q)` change. |
| `logits` | output FP32 `[M,V]` | Raw logits predicting the token after each selected query row under its visible history. |
| `features` (target) | output FP16 `[Q,12288]` | Concatenated inputs to layers 2, 16, 28, before their norms, in query-row order. |
| `target_features` (draft) | input FP16 `[Q,12288]` | Verified preceding-token target features; zero during recurrent drafting. |
| `draft_features` (draft) | input FP16 `[Q,4096]` | Previous draft residual; zero when conditioning on target features. |
| `features` (draft) | output FP16 `[Q,4096]` | Unnormalized draft residual, in query-row order. |

The target vocabulary has 128256 entries; the draft has 32000. Draft output
indices must pass through the checkpoint's `d2t` mapping before becoming target
token IDs. Both draft conditioning inputs have `Q` rows. Each invocation requires
`1 <= M <= Q`, `s >= 0` and `s+Q <= C`, within its declared profile bounds.

### State effects and ownership

- The graph must declare KV mutation and required input/output aliases.
  The manifest describes those effects; it cannot establish them. Binding
  unrelated tensors to the same address is insufficient: the runtime checks
  TensorRT's engine alias metadata.
- Writes materialize tentative rows without committing them. Logical token
  positions, initialized physical span and accepted length are distinct.
  Attention combines declared visibility with the initialized span; inactive
  cache rows are sanitized before matmuls to prevent stale NaN contamination.
- Accepting a tree path can select noncontiguous rows. Gather the entire path
  to scratch before copying it into the committed prefix, so overlapping
  sources and destinations cannot corrupt state.
- Rejected rows and reset state become inaccessible through lengths/masks.
  Their bytes need not be cleared; subsequent writes may overwrite them.

## EAGLE3 alignment and transitions

1. Target prefill of `N` prompt tokens emits a root token whose target KV is
   still pending. Draft prefill pairs `prompt[1:] + root` with the unshifted
   target features at draft positions `0..N-1`.
2. Draft proposals use recurrent residual features. Target verification consumes
   `root + candidates`; a logits row predicts its child, not its own token.
   Tree visibility follows ancestry, and RoPE positions follow depth rather
   than flattened candidate row indices.
3. Acceptance retains the root and matching candidate path. Emit accepted
   candidates and a target bonus token; the bonus becomes the next pending root.
4. Before continuing, draft feedback overwrites tentative recurrent rows using
   verified target features from the retained path. Feature `F_i` pairs with
   token `x_(i+1)` at draft position `i`, ending with the bonus token. Target
   and draft committed lengths advance together despite this conditioning offset.

The implemented tree expands the best branch and exposes a second-choice sibling
at each depth; it is not a full beam-search policy. A future target-prefix cache
hit must also restore or reconstruct compatible draft/feature state.

## Execution and value lifetimes

`execution_profiles` records phase and query/logit MIN/OPT/MAX bounds in engine
profile order: 0=prefill, 1=decode. Missing/empty metadata means one shared profile.
Select by phase, not row count: a one-row prefill still uses the prefill context.
Both contexts share KV allocations and stream ordering; changing phase preserves
state. Target and draft have separate KV storage.

`Engine::run` returns completed borrowed device feature views, valid only until
the producer's next invocation, reset or destruction. Stage draft inputs before
reusing producer outputs. Accumulate prompt features in independently owned
storage across prefill chunks and complete target-stream copies before draft
prefill. The runtime owns KV/staging/compaction buffers; TensorRT owns workspace.

### Optional greedy selector

`greedy_selection=device_v1` requires `target_selection.plan` and/or
`draft_selection.plan` for the engine declaring it. Missing metadata defaults
to host selection. These stateless graphs consume existing model outputs;
enabling them does not change the attention/KV ABI.

| Binding | Type/shape | Semantics |
|---|---|---|
| `logits` | input FP32 `[M,V]` | Borrowed model logits in selected-query order. |
| `selection` (target) | output INT32 `[M,2]` | Best vocabulary index, all-finite flag. |
| `selection` (draft) | output INT32 `[M,3]` | Best and second-best distinct indices, all-finite flag. |

Ranks sort by descending value, breaking ties by lowest index, including signed
zero. NaN or either infinity invalidates the row; IDs on invalid rows are
unspecified. Reject invalid rows when consumed by the policy, not merely because
an unvisited branch contains them. Indices remain in each model's vocabulary.

The selector waits on the model-completion event before reading device logits.
Its checked stream synchronization completes the model outputs and compact
readback before returning to host policy. This is a greedy-decision interface;
sampled methods may require full distributions and a different selector contract.

## Changing the lowering or state representation

[Graph.attention()](speculative/graph.py) is the lowering boundary. Replacing its
attention primitives with `IAttention` can retain native linear KV updates and the
existing runtime ABI. Preserve Q scaling, RoPE, GQA, tree visibility, inactive-row
handling, cache aliases, row ordering and numerical qualification. This native
attention replacement is not yet qualified; an API change alone promises no
particular fused kernel or speedup.

Changing external cache layout, dtype, mask encoding or alias semantics requires
a new ABI or explicit adapter. Paging is independent of proposal policy, but
must declare pool geometry, logical-to-physical read/write mappings, partial-page
validity, shared-page ownership and in-flight lifetime. Shared tails require
copy-on-write or private tentative storage; page remapping cannot in general
replace compaction of arbitrary rows within a page. Allocation and reclamation
remain runtime responsibilities.

[Validation](tests/cpp/speculative_validation.cpp) checks AR/chain/tree output
consistency, reset and short prompts. The 1024-input fixture with 101 output IDs
includes one prefill prediction and 100 autoregressive decode calls. Agreement
on that repeated fixture does not establish general model quality or long-context
accuracy.
