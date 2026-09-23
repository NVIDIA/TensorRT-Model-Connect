# Llama speculative decoding prototype

This family owns an FP16, batch-one, single-GPU Llama 3.1 8B Instruct target
and EAGLE3 draft. The target retains its full 128256-token vocabulary. The
draft uses its trained 32000-token vocabulary and an explicit draft-to-target
map. The builder reads both checkpoints directly and produces a standalone
Model-Connect bundle.

The implementation uses TensorRT-native graph primitives, including
`IKVCacheUpdateLayer` and explicitly masked grouped-query attention. TensorRT
owns their lowering, fusion and kernel selection.

[Repeated GB100 performance measurements](SPECULATIVE_PERFORMANCE.md) compare
MC autoregressive and EAGLE3 execution separately from the correctness runs below.

## Ownership and entry points

- `speculative/contract.py`: versioned compiler/runtime tensor and state ABI.
- `speculative/graph.py`: target and draft graphs and the attention lowering seam.
- `speculative/selection.py`: optional TensorRT graph for compact greedy decisions.
- `speculative/build.py`: checkpoint mapping and bundle composition.
- `runtime/speculative/engine.*`: ABI validation, bindings, execution and KV copies.
- `runtime/speculative/eagle3.*`: method-specific conditioning, proposal and feedback.
- `runtime/speculative/pipeline.*`: request lifecycle and target verification.

The existing Llama family remains the owner. Speculative decoding is not a
new model family. Other methods can reuse the engine adapter when their
tensor/state semantics match, and supply their own graphs and scheduling.
Recurrent-state methods need a new state contract; this KV ABI does not claim
to cover every speculative technique.

The family-owned build command is discoverable through Model-Connect's CLI:

```bash
PYTHONPATH=core/builder:. python -m tensorrt_model_connect llama build-speculative \
  --model-dir /models/Llama-3.1-8B-Instruct \
  --draft-dir /models/EAGLE3-LLaMA3.1-Instruct-8B \
  --spec-dec eagle3 --max-sequence-length 2048 --max-query 64 \
  --draft-depth 4 --output /models/llama-eagle3.bundle

cmake --build build-sm100 --target trtmc trtmc_backend_trt trtmc_model_llama
```

The standard family factory loads the resulting bundle as `ITextGeneration`.
`text_generation_mode=auto` uses EAGLE3; `autoregressive` provides a baseline
with the same compiled target. Sampling beyond greedy selection and repetition
penalties are rejected. Existing non-speculative bundles use the original
pipeline.

The prototype expands a top-one chain and, by default, includes its top-two
sibling at each depth in the verification tree. This exercises branching
visibility, depth-based positions and noncontiguous accepted-path compaction.
Full beam scoring and dynamic tree selection are outside this prototype.

## Compiler/runtime ABI v1

`speculative.json` names two engine contracts and the EAGLE3 configuration.
For either engine, let `Q` be query rows, `M` selected logits rows, `C` fixed
cache capacity, `Hkv` KV heads, and `D` head dimension. All tensors are dense,
contiguous, batch one. All execution bindings are device pointers. Token and
visibility control inputs are uploaded; conditioning features stay on device.
Tensor names are part of the ABI.

| Binding | Direction/type/shape | Meaning |
|---|---|---|
| `token_id` | input INT32 `[Q]` | Full target-vocabulary IDs, including for the draft embedding lookup. |
| `position_id` | input INT32 `[Q]` | Logical RoPE positions. Siblings share a position; positions do not identify physical cache slots. |
| `cache_write_indices` | input INT32 `[1]` | Physical start `s` of the tentative contiguous write. Row `r` writes slot `s+r`. |
| `key_value_lengths` | input INT32 `[1]` | Initialized span after this invocation, `s+Q`; not the accepted length. |
| `attention_mask` | input INT32 `[Q,C]` | `1` means visible, `0` means hidden. Each row sees the committed prefix and its inclusive ancestors only. |
| `logits_indices` | input INT32 `[M]` | Query-row indices to run through the final norm and LM head, in requested order. |
| `cache_k_i`, `cache_v_i` | input FP16 `[1,Hkv,C,D]` | Runtime-owned persistent state for layer `i`; K is already rotated. |
| `present_k_i`, `present_v_i` | output FP16 `[1,Hkv,C,D]` | Alias the corresponding cache input. Only `[s,s+Q)` changes; all other slots are preserved. |
| `logits` | output FP32 `[M,V]` | Row `j` predicts the token after query row `logits_indices[j]`, under that row's visible history. Raw logits, not probabilities. |
| `features` (target) | output FP16 `[Q,12288]` | Concatenated inputs to target decoder layers 2, 16, 28, before their norms; identical query-row order. |
| `target_features` (draft) | input FP16 `[Q,12288]` | Verified target features from the preceding logical token, or zero during recurrent drafting. |
| `draft_features` (draft) | input FP16 `[Q,4096]` | Previous draft residual features, or zero when using target features. |
| `features` (draft) | output FP16 `[Q,4096]` | Unnormalized draft residual stream, in query-row order. |

By default both engines retain one optimization profile, with query and selected
logit row MIN/OPT/MAX bounds `1/16/64` (OPT is capped by `max_query`). Prefill is
chunked at `max_query`, 64 by default. A prefill call selects only its last logits
row while returning all feature rows. Verification selects every candidate row.
The runtime enforces `1 <= M <= Q`, `s >= 0` and `s+Q <= C`.
The two conditioning inputs must have the same `Q` as the draft's token input.

### Separate prefill and decode profiles

Build with `--execution-profiles split --prefill-query P` to specialize each
engine for both phases. With draft depth four, the shape contract is:

| Engine/profile | Query MIN/OPT/MAX | Selected logits MIN/OPT/MAX |
|---|---|---|
| Target 0: prefill | `1/P/P` | `1/1/1` |
| Target 1: decode/verify | `1/5/9` | `1/5/9` |
| Draft 0: prefill | `1/P/P` | `1/1/1` |
| Draft 1: propose/feedback | `1/1/5` | `1/1/1` |

Target decode is optimized for a four-candidate chain; its maximum also admits
the width-two, depth-four tree. Draft proposal consumes one row, while accepted
target-feature feedback can consume up to five rows. `P=64` preserves chunking;
`P=1024` allows one target and one draft prefill invocation for an IST=1024
request. Short prompts and final partial chunks use profile 0 even for one row.
`--max-query` remains the legacy profile bound and a speculative-depth safety
cap; split profile bounds are derived from `--prefill-query` and `--draft-depth`.

The optional `execution_profiles` array in each engine's `speculative.json`
contract records `phase`, `query` and `logits` MIN/OPT/MAX triplets in profile
index order. `max_query` is the largest query maximum across those profiles.
Missing or empty arrays select the legacy single-profile runtime. This is
additive execution metadata: state ABI v1 retains its existing tensor semantics,
storage layout and alias requirements. Other state ABI versions are rejected.
Split bundles require the updated runtime; existing single-profile bundles
remain readable.

The runtime selects the profile from an explicit `Phase`, validates every
dynamic input's MIN/OPT/MAX against the serialized engine, and checks invocation
row counts before enqueue. Each engine is deserialized once into two persistent
execution contexts sharing its weights, CUDA stream and runtime-owned KV
allocations. Context workspaces and ordinary input/output buffers are separate.
Both contexts bind the same KV addresses and validate the alias contract.
Prefill-to-decode transitions preserve state without copying or resetting KV.
Request reset resets logical lengths and invokes both modules' reset hooks;
the backend preserves the existing contexts and bindings. Target and draft still
own separate states and streams. Host synchronization preserves their existing
ordering. Profile selection changes neither commit/rollback nor EAGLE3 feature
alignment.

The compiler encodes KV reads, writes and aliases in the graph. The sidecar
describes that graph contract; it is not a replacement for compiler-visible
state effects. The runtime rejects engines whose cache outputs are not declared
as aliases of their inputs; manually binding unrelated tensors to the same
address does not satisfy this contract. TensorRT owns engine workspace.
The runtime owns KV buffers, input staging and reusable compaction storage.
Every execution and copy uses its module's CUDA stream. Feature outputs are
borrowed device views, completed before return and valid until the producer's
next invocation/reset/destruction. Draft inputs are staged into separate device
buffers before enqueue, preserving recurrent output lifetime without input/output
aliasing. Prompt features are accumulated on the target stream and completed
before draft prefill. See the [resident runtime contract](SPECULATIVE_RESIDENT_RUNTIME.md)
for the optional selector's cross-stream dependency and numerical semantics.

Only initialized visible slots may affect attention. The current lowering
sanitizes inactive cache rows before matmuls, so masked stale NaNs cannot
contaminate the result. Reset changes logical state; clearing all KV bytes is
not necessary. Unaccepted rows remain inaccessible until overwritten.

## EAGLE3 state alignment

After target prefill of `N` tokens, the target emits a root token whose KV has
not yet been computed. Draft prefill consumes the shifted tokens
`prompt[1:] + root` paired with the unshifted target features. Its positions
are `0..N-1`. Its last output predicts the first draft candidate.

A target verification invocation consumes `root + candidates`. Each logits
row predicts a child, not the token occupying that row. Greedy acceptance
walks the matching root-to-leaf path. The runtime commits KV for the root and
accepted candidates, emits accepted candidates and a target bonus token, and
leaves that bonus pending. For trees it first gathers accepted physical rows
to temporary storage, then compacts them into the committed prefix. This
two-phase copy is safe even when source and destination slots overlap.

Draft feedback uses verified target features for the committed path, paired
with the next accepted token or the bonus. It overwrites tentative recurrent
draft rows. Target and draft committed lengths therefore advance together,
despite the one-token offset in their conditioning semantics.

## Swapping attention implementations

The `Graph.attention()` path forms Q/K/V, applies RoPE, performs native
linear KV updates, and emits explicit masked attention. An implementation
replacing this v1 region without changing its external ABI must:

1. Preserve the same FP16 Q/K/V, scale `1/sqrt(D)`, rotate-half RoPE and
   grouped-query semantics.
2. Accept arbitrary declared query visibility and logical positions,
   including siblings with equal positions and different physical slots.
3. Read/write exactly the specified cache layout and write interval, expose
   the state effects and return the same aliases.
4. Preserve selected logits and feature row order, dtype and shape profiles.
5. Respect stream ordering, workspace ownership and output lifetime.

A future TensorRT `IAttention` lowering must encode these same semantics,
including explicit state effects. Compiler fusion and tactic selection can
then change internally without changing the runtime's logical operations.

Rebuilding an engine may change its internal attention implementation while
the runtime keeps the same bindings and algorithm. Changing public KV layout,
alias rules or feature semantics requires an ABI version or an explicit
adapter. A paged cache or a packed-mask encoding is not binary compatible
with this linear layout and dense visibility tensor. Fused native attention
remains qualification work.

## Design refinement: reusable state and model I/O

This September 21 refinement does not change the implemented version-1 ABI.

The compiler/runtime contract has three parts: model input/output semantics,
state representation and effects, and execution constraints. KV management is
the main externally mutable model-state mechanism for this dense Llama pair.
EAGLE3 also passes residual features between calls as ordinary tensor values;
those features are not hidden side effects or additional KV state. Hybrid
models may require recurrent and convolution state in addition to KV.

Speculative method, state representation, and attention implementation should
be independently selectable within qualified combinations:

- The method constructs candidate histories and selects the continuation.
- A state manager reserves/binds storage and retains or discards evaluated
  rows without interpreting the speculative method name.
- Each engine declares a concrete physical state ABI and its permitted
  reads, writes, aliases, initialization rules, and ordering requirements.
- Model-specific conditioning and draft refresh remain method/family logic.
  EAGLE3 feedback requires verified target features, not just KV compaction.

The current engine adapter and EAGLE3 policy provide an initial separation;
there is no general state-manager interface or paged implementation yet.
Iterative token-by-token drafting is not a universal requirement: block
drafters can propose multiple tokens per call. Target verification requires
the correct per-row history; a chain can use causal visibility, while a tree
needs ancestor visibility. Its encoding is an explicit engine ABI choice.

### Semantic state obligations and physical ABI

Reuse semantic obligations across cache implementations: evaluate the declared
history, isolate tentative writes, preserve published state, and make only
the retained continuation visible. A concrete engine still binds one exact
representation. Linear append indices, paged read/write mappings, cache dtype
and geometry, quantization metadata, and aliases cannot change silently.
Other state kinds may use fresh state outputs rather than this ABI's in-place
aliases; they need their own declared effects.

A future paged profile must specify page size and pool layout, mapping tensor
encoding, query-row write slots, valid rows in partial pages, and shared-page
ownership. Shared tails require copy-on-write or separate tentative storage.
Page remapping can retain a selected path only when token placement permits
it; arbitrary accepted rows inside a page can still require copies. Mapping
updates and reclamation must wait for in-flight consumers. Allocation,
reference counting, path retention, and discard are runtime operations, not
mandatory new compiler entry points. If a state utility is compiled, its
inputs/outputs and effects use the ordinary engine contract.

### Target/draft composition and position semantics

Exported feature semantics are part of the compiler/runtime I/O contract:
producer and exact tap, normalization point, concatenation order, dtype,
row/token association, and lifetime. Connecting those tensors to another
model and deciding when to execute it are runtime composition.

For this EAGLE3 pair, verified target feature `F_i` is paired with token
`x_(i+1)` at draft position `i`. The recurrent proposal path uses the previous
draft residual instead. Keep logical positions, materialized progress, and
physical storage slots separate; page IDs or flattened candidate indices do
not determine RoPE positions. A target-only prefix-cache hit also needs an
explicit path to restore or rebuild compatible draft/feature state.

Paging and speculation are not the complete requirement set. Extension
profiles must account for ragged batching, chunked prefill, prefix sharing,
sliding windows, quantized-cache metadata, state/checkpoint identity, and
eventually distributed/offloaded or hybrid state. These are declared
capabilities, not required additional bindings on the initial linear engine.

## Validation

Build `llama_speculative_validation` and `test_llama_speculative_policy` with
`TRTMC_BUILD_TESTS=ON`. The validation executable loads the public bundle and
family runtime, compares autoregressive, chain and branching outputs, then
checks a second request after speculative state has been used:

```bash
python families/llama/tests/prepare_speculative_fixture.py /models/target /tmp/spec-fixture
build-sm100/families/llama/llama_speculative_validation \
  /models/llama-eagle3.bundle /tmp/spec-fixture/input_ids.json 100 /tmp/result.json
```

The fixture is 1024 raw input IDs with no chat templating. A 100-token output
contains the first prefill prediction and 99 ordinary decode calls in the
autoregressive baseline. Use output count 101 for exactly 100 such calls.
The report includes complete output IDs, acceptance lengths, verification
rounds and wall times. Timings include host scheduling and copies; they are
not an optimized serving benchmark.

### Recorded native validation (September 22)

The cleanup at `a2638214e6c28a4b970178ce848fc73b480b744b` was validated on
Blackwell GB100 (SM100) with TensorRT 11.1.0.106 and CUDA 13.3, using FP16
engines and KV state. The target checkpoint was
`meta-llama/Llama-3.1-8B-Instruct` revision
`0e9e39f249a16976918f6564b8830bc894c89659`; the draft was
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` revision
`ada412b672e293d682423de84a095447bf38a637`.

- Exactly 1024 input IDs and 101 output IDs: MC autoregressive, chain and
  tree outputs matched for device selection, host selection and chunked prefill.
- The autoregressive baseline made 100 decode calls after its prefill prediction.
- Reusing the pipeline after speculative writes produced the same output.
  One- and 65-token prompts, zero-output and invalid-count checks passed.
- A fresh runtime build, four C++ tests, 23 focused Python tests, 168 GPU
  selection cases, Ruff and C++ formatting checks passed.
- Four small target/draft graphs compiled with single and split profiles.
  Full-model checks reused existing native 8B plans with the freshly built
  runtime; full-model plans were not rebuilt during this cleanup.

These checks establish consistency between MC execution modes for one
deterministic fixture. They do not establish tensorwise-logit accuracy or
general model quality. Broader prompts and long-context numerical validation
remain necessary. Full beam scoring, sampled decoding, batching, dynamic
page allocation, quantization and fused native paged attention remain
follow-up work.
