# TriAttention Native C++ Worklog

Date: 2026-04-16

:::info Point-in-time implementation record

Commands, temporary artifact paths, benchmark numbers, and hypotheses on this
page are evidence from the dated TriAttention bring-up. They are not current
setup instructions and are not automatically revalidated on later revisions.
Use current Qwen descriptors, tests, and CLI help for the live contract.

:::

## Purpose

This worklog records how TriAttention was brought from an upstream research
runtime into the native C++ TensorRT runtime in this repo, what design choices
were made along the way, which debugging hypotheses were wrong, and what
finally produced an end-to-end result with both accuracy and throughput
benefit.

This document replaces the older point-in-time superpowers spec notes for this
feature. Those snapshots were useful during implementation, but the website now
keeps the durable feature narrative here.

## Final outcome

The final native implementation is a real C++/CUDA runtime path, not a Python
debug shortcut:

- bundle build embeds TriAttention metadata and stats
- runtime compaction is handled by `TriAttentionKvCache`
- selection and repack run natively in C++/CUDA
- the same TRT decoder engine continues after compaction
- runtime KV allocation can be set at launch time with `--kv-cache-size`

On the corrected Qwen3-8B AIME25 pilot slice used for parity checks, native
TriAttention now matches dense TRT on answer accuracy while remaining faster.

Matched 6-sample long-budget slice:

- dense answers:
  `tmp/qwen3_dense_completed6.jsonl`
- TriAttention answers:
  `tmp/qwen3_tri_v14_completed6_long.jsonl`
  `tmp/qwen3_tri_v14_sample5_completed6_long_gpu2.jsonl`
  `tmp/qwen3_tri_v14_sample6_completed6_long_gpu3.jsonl`

Answer parity on that slice:

- `aime25_1 -> 70`
- `aime25_2 -> 588`
- `aime25_3 -> 16`
- `aime25_4 -> 117`
- `aime25_5 -> 279`
- `aime25_6 -> 504`

Throughput on that same slice:

- dense: `64.16 tok/s`
- TriAttention: `86.12 tok/s`

That is about `1.34x` faster while preserving the dense answers on the matched
pilot.

Current reproducible PR-tip validation:

- TriAttention bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri12288-b3072-r128-dynkv-fp16-manual-current.trtfb`
- dense control bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-dense12288-dynkv-fp16-manual-samefam.trtfb`
- runtime overrides:
  `TRTMC_TRIATTN_OVERRIDE_KV_BUDGET=6144`
  `TRTMC_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`
  `TRTMC_TRIATTN_RUNTIME_BUCKET_ROWS=32`
- validation outputs:
  `artifacts/triattention/policy-sweeps/2026-04-17-current-bundle-override-validation/`

Checked results from that current-tree validation:

- `aime25_2`: TriAttention answer `588`, dense answer `588`
- `aime25_3`: TriAttention answer `16`
- `aime25_2` no-stop throughput:
  TriAttention `56.658344 tok/s`
  dense same-family control `42.614745 tok/s`

That is `1.3295x` on the fair same-family dense baseline while preserving the
checked hard answers.

## High-level design choices

### 1. Treat TriAttention as a runtime/cache policy, not an attention-kernel swap

The first important conclusion from upstream was that TriAttention is not
"replace TensorRT attention with a new attention op." It is a cache scoring,
selection, and compaction policy with runtime metadata and lifecycle handling.

That drove the integration point:

- builder persists TriAttention config and calibration stats into the bundle
- runtime owns cache state, position tracking, selection, and compaction
- the TRT engine stays the decoder, but the cache semantics change underneath it

This was the right choice. It kept the project feasible and matched how the
upstream runtime is actually structured.

### 2. Move to native C++ instead of a Python evaluation shim

The first repo-local enablement path used a Python debug runner to validate the
idea. That was useful for initial understanding, but the user explicitly did
not want a Python shortcut. The implementation was therefore moved fully into
the native runtime.

This forced the real project decisions:

- native state object under `IInferenceState`
- native bundle config parsing
- native selector and compaction logic
- native tests and native benchmarks

### 3. Prefer a single dynamic-KV engine over multi-engine switching

There were two plausible ways to reduce real attention work:

- bucketed engine switching
- a single engine with dynamic KV row count

After reviewing TensorRT local docs and the repo's existing dynamic-shape use,
the single-engine path was chosen. That proved sufficient as long as the decode
graph was built with dynamic KV input shapes and the runtime rebound the live
row count.

Why this choice was correct:

- less engine-management complexity
- no explicit engine migration cost
- cleaner runtime model
- consistent with TensorRT's intended dynamic-shape usage

### 4. Separate physical cache limit from logical TriAttention budget

One important quality issue was prompt-time overflow. Using a tiny physical
window and a tiny logical budget caused compaction to happen during prompt
prefill, which damaged the prompt-state before decode started.

The fix was architectural:

- physical limit large enough to hold the prompt cleanly
- logical TriAttention budget smaller and enforced during decode

This is why configurations like `tri256-b128` were materially better than
`tri192-b128` on overflow prompts.

### 5. Make runtime KV allocation configurable in bytes

Dynamic shapes remove the fixed exact KV length at runtime, but not the build
upper bound. The runtime therefore needed a user-facing way to choose the
actual allocation below that upper bound.

The implemented runtime model is:

- build with a large dynamic upper bound
- allocate only the requested runtime amount
- expose that through `--kv-cache-size`

That became a must-have feature because otherwise large build-time bounds would
force wasteful memory allocation at startup.

### 6. Keep GPU work on GPU

The first native implementation was functionally correct enough to prove the
idea, but it was slow because it copied cache rows to host, converted them on
CPU, scored on CPU, and repacked with many tiny D2D row copies.

The final runtime moved the hot pieces to GPU:

- GPU candidate scoring
- GPU gather-style compaction
- host only for small normalization/combine work where still acceptable

This was essential to recovering throughput.

## What was implemented

The final feature spans four layers.

### Bundle/build layer

- TriAttention stats export into bundle JSON sections
- TriAttention runtime config persisted in bundle config
- dynamic-KV-capable decoder engine build

### Runtime state layer

- `TriAttentionKvCache` owns:
  - cache tensors
  - per-head position tracking
  - compaction trigger logic
  - reserved/recent/prefill protection
  - keep-index selection
  - compaction and cache rebound

### CUDA helper layer

- GPU scoring kernel
- GPU KV gather/repack kernel

### CLI/runtime config layer

- runtime KV byte budget
- dynamic external KV binding without eager allocation of the full engine max

## Debugging chronology

### Phase 1: understand upstream correctly

The first upstream reading established two things:

1. TriAttention is a runtime policy, not a TRT graph replacement.
2. Upstream's best performance story is tied to long reasoning workloads,
   GPU scoring, and serving-runtime cooperation.

This prevented an early wrong turn: trying to "just add a new attention layer"
in TRT.

### Phase 2: native C++ port with a shared-row approximation

The first native version worked as a C++ cache compaction path and proved that:

- compaction fired in native code
- the runtime could continue generation after compaction
- retained-memory prompts behaved better than undersized dense baselines

But this version still used a simplified shared-row interpretation compared to
the stronger upstream per-head behavior.

### Phase 3: sanity checking with retrieval-style prompts

The first correctness checks used marker/retrieval prompts. These were useful
because they had deterministic expected answers, but they were not persuasive as
general quality evidence.

That led to a better evaluation discipline:

- use retrieval-style prompts only for deterministic cache sanity checks
- present normal conversational prompts when discussing visible output quality

### Phase 4: performance regression investigation

The first native C++ compaction path was dramatically slower than dense once
compaction engaged.

The key finding from profiling was simple:

- pre-compaction decode was already fine
- nearly all slowdown came from the compaction event itself

The cost breakdown pointed at:

- host copy + dtype conversion
- CPU scoring
- thousands of tiny D2D repack copies

That led directly to the GPU scoring and GPU gather/repack implementation.

### Phase 5: dynamic-KV path and runtime KV sizing

Reducing cache residency alone is not enough if the engine still binds and
processes the full physical window. The runtime was therefore extended to bind a
live KV row count into a single dynamic-shape engine.

This was paired with runtime byte-budget control so the actual allocation could
be chosen at launch time instead of being fully materialized at bundle max.

### Phase 6: overflow bug during prompt prefill

One of the harder early failures was long-context overflow on normal prompts.
The issue was not initially obvious because the selector appeared reasonable.

The actual problem was compaction during prefill in too-small a physical cache.

Fix:

- use physical slack during prompt fill
- decouple physical capacity from logical keep budget
- ensure compaction starts after decode crosses the physical threshold, not
  while prompt state is still being built

### Phase 7: evaluation recipe was wrong

A separate accuracy confusion came from the benchmark recipe itself. Early AIME
results looked poor for dense, TriAttention, and HF because the decode recipe
was throughput-oriented:

- greedy or deterministic decode
- poor stopping behavior
- simplistic extraction
- non-chat prompting in some earlier runs

After switching to the corrected Qwen reasoning recipe, dense TRT recovered to
HF parity on the pilot slice. This mattered because it separated "bad eval
recipe" from "TriAttention bug."

### Phase 8: remaining TriAttention accuracy gap after compaction

At that point dense and HF were aligned, but TriAttention still failed hard
reasoning samples once real compression activated.

Several debugging hypotheses were tested.

#### Hypothesis: sampled-head mapping was being handled incorrectly

This was partly true.

The runtime had to preserve actual `sampled_heads` mappings instead of assuming
uniform contiguous grouping. Fixing this materially improved results and made a
previously bad AIME sample return the correct answer.

But it was not sufficient by itself.

#### Hypothesis: the benchmark stats file was sparse sampled-head-only

This turned out to be wrong for the actual Qwen3-8B AIME25 calibration file.
Inspection showed that the file effectively carried full per-attention-head
stats for all `36 * 32 = 1152` attention heads.

That changed the interpretation of the runtime semantics:

- the selector should behave like grouped per-head reduction over dense
  attention-head stats
- sparse sampled-head fallback logic alone was not the real upstream match

#### Hypothesis: the remaining gap was purely in per-head compaction layout

This was plausible, but still not the first blocker.

The decisive remaining mismatch was the scorer formula.

### Phase 9: final root cause - native scorer used the wrong formulation

Upstream's current vLLM/Triton runtime does not score by unrotating keys using
cached token positions. It scores directly on stored `K_rot`, where key
position is already baked into the rotated key, and only the query-side phase
term remains.

Our native runtime was still using the older formulation:

- recover key position
- unrotate key
- reapply phase with cached positions

That was the wrong match for the upstream runtime being emulated.

The fix was to change both the host selector and the CUDA kernel to the direct
`K_rot` formulation:

- use stored rotated key components directly
- compute `Q_mean * conj(K_rot)` directly
- apply only the query-side trig term
- keep the additive MLR term on `|K_rot|`

After that fix, the previously bad hard samples flipped to the correct answers.

## Why the final fix worked

The direct-`K_rot` scorer mattered because it matched the runtime semantics of
the upstream implementation we were actually trying to reproduce.

It also removed two fragility points:

- dependence on exact cached position bookkeeping for scoring correctness
- extra numeric reconstruction work not present in the upstream runtime path

Once the native scorer matched upstream, the remaining native compaction path
was good enough to preserve quality on the corrected pilot.

## Evidence that the feature now works

### Native tests

- `ctest --test-dir build --output-on-failure -R test_triattention_kv_cache`

### Hard-sample recovery

Correct after scorer alignment:

- `tmp/qwen3_tri_v14_sample1_5000_gpu1.jsonl`
- `tmp/qwen3_tri_v14_sample2_5000_gpu1.jsonl`

### Corrected 6-sample pilot at shorter budget

- `tmp/qwen3_tri_v14_completed6.jsonl`

This run showed 6/6 correct with materially higher tok/s than dense.

### Corrected 6-sample pilot at matched long budget

Artifacts:

- aggregate partial:
  `tmp/qwen3_tri_v14_completed6_long.jsonl`
- per-sample completions:
  `tmp/qwen3_tri_v14_sample3_completed6_long_gpu1.jsonl`
  `tmp/qwen3_tri_v14_sample4_completed6_long_gpu1.jsonl`
  `tmp/qwen3_tri_v14_sample5_completed6_long_gpu2.jsonl`
  `tmp/qwen3_tri_v14_sample6_completed6_long_gpu3.jsonl`

Dense comparison:

- `tmp/qwen3_dense_completed6.jsonl`

Result:

- same extracted answers as dense on all 6 samples
- higher average throughput than dense

## Important non-obvious lessons

### 1. Accuracy debugging needed a matched evaluation recipe first

Until dense and HF agreed, TriAttention debugging was underdetermined. Fixing
the benchmark recipe first was mandatory.

### 2. Upstream drift matters

The upstream runtime semantics had moved to direct `K_rot` scoring. Matching an
older formulation would have kept the implementation "reasonable" but still
wrong.

### 3. Cache semantics and attention-work reduction are related but distinct

There were really two separate projects:

- make compaction correct
- make compaction useful for throughput

Both had to be solved independently.

### 4. Physical and logical cache sizes should not be conflated

That one design choice explained a large part of the early overflow and prompt
quality failures.

## Final apples-to-apples check

The final fair comparison used the upstream plain-prompt recipe and a true
full-KV dense baseline:

- dense full-KV bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-dense32768-dynkv-fp16-manual-fullkv.trtfb`
- TriAttention bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri12288-b3072-r128-dynkv-fp16-manual-current.trtfb`

Prompt recipe:

- plain upstream math prompt, no chat template
- `temperature=0.6`
- `top_k=20`
- `top_p=0.95`
- `min_p=0.0`
- `seed=1234`

The raw probe inputs and outputs were generated under the ignored
`artifacts/triattention/` tree. The checked-in record keeps only the summarized
numbers below so benchmark outputs do not become source files.

Dense full-KV results:

- `aime25_2` no-stop:
  answer `588`, `38.89729 tok/s`
- `aime25_3` stop-on-answer:
  answer `16`, `53.068426 tok/s`

TriAttention results:

- `aime25_2` no-stop:
  answer `588`, `61.263541 tok/s`
- `aime25_3` stop-on-answer:
  answer `16`, `68.298592 tok/s`

Derived result:

- `aime25_2` no-stop speedup versus dense full-KV: `1.5750x`

### Why this is still fair even though the TriAttention bundle is `tri12288`

For these probes, the physical max length difference is inactive.

The prompt lengths are:

- `aime25_2`: `690` tokens
- `aime25_3`: `164` tokens

The runtime only begins compaction once
`cache_length_ >= compaction_trigger_length()`, and
`compaction_trigger_length()` is `kv_budget + divide_length = 3200` for this
bundle configuration; see
`src/runtime/models/<family>/triattention_kv_cache.cpp`.

Because both prompts are far below `3200`, the tested runs never depend on a
physical cache capacity larger than `12288`. After compaction starts, the live
cache stays near the logical TriAttention budget, again far below `12288`.
So on these probes a hypothetical `tri32768-b3072` bundle would follow the same
runtime path, and the comparison against `dense32768` is apples-to-apples for
the actual operating point being measured.

## Remaining limits

The current state is strong enough for native feature support, but there are
still follow-up opportunities:

- rerun a broader AIME25 sweep under the corrected recipe, not just the matched
  pilot slice
- revisit true KV-head-native cache layout for GQA models to eliminate the
  expanded query-head representation
- recover a more optimized long-window attention path where native TensorRT
  `IAttention` is unstable and the manual path is used instead
- consolidate the split pilot artifacts into one aggregate output file for
  easier future regression checking

## 2026-04-17 Follow-up: full-benchmark regression and current root-cause status

After the earlier probe-level success, a full 30-sample AIME25 rerun exposed a
real remaining problem:

- HF eager:
  accuracy `0.700`, wall `15.96 tok/s`
- dense full-KV:
  accuracy `0.667`, wall `38.08 tok/s`
- TriAttention:
  accuracy `0.267`, wall `70.28 tok/s`

Recorded summary:

- `tmp/qwen3_8b_aime25_fullkv_plain_summary.md`
- `tmp/qwen3_8b_aime25_fullkv_plain_summary.json`

That result was too large a quality loss to wave away as sampling noise, so the
next debugging pass focused on whether the runtime selector/compactor was still
wrong on the bad long-window cases.

### Selector correctness is no longer the leading suspect

For `aime25_23`, native score-cache dumps were replayed through the local
upstream selector implementation in
`artifacts/triattention/upstream/triattention/vllm/runtime/selector_hf.py`.

For both compaction 1 and compaction 2:

- exact match on the selected rows for every sampled head
- mean Jaccard overlap `1.0`

This ruled out the remaining "wrong keep-set" theory for that sample.

### K-cache repack correctness is also no longer the leading suspect

For the same `aime25_23` investigation, the post-compaction K cache was checked
against the pre-compaction cache by directly gathering the kept indices. The
result was exact:

- `max_abs_diff = 0.0`
- no bad layers

So the native path is not silently scrambling K rows after selection on that
sample.

### One old long-window artifact was proven invalid as an apples-to-apples control

The older artifact
`artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-apple.trtfb`
turned out not to be a valid same-family long-window baseline.

Even under conservative no-compaction settings and greedy decode, it diverged
from the dense full-KV bundle before compaction and also ran much faster than
dense in the pre-compaction region. That means it is not "the same engine path
plus TriAttention policy"; it is a materially different engine/runtime
combination. It must not be used for final parity claims.

This finding changed the next step of the investigation:

- stop trusting the stale `tri32768 ... manual-apple` artifact for full-benchmark
  conclusions
- rebuild a fresh same-family `dense32768` control from the current code
- rebuild the matching fresh `tri32768` bundle from the current code
- re-prove greedy no-compaction equivalence on the fresh pair before resuming
  full-benchmark tuning

At the time of writing this note, that fresh same-family rebuild is still in
progress and the full-benchmark parity question remains open.

### Same-bundle control proof: compaction is not the only remaining issue

After the stale-bundle baseline problem was identified, the next control was to
use the exact same `tri32768` bundle in two modes:

- `TRTMC_TRIATTN_FORCE_ENABLE=0`
- `TRTMC_TRIATTN_FORCE_ENABLE=1`

with compaction effectively disabled by runtime overrides:

- `TRTMC_TRIATTN_OVERRIDE_KV_BUDGET=16384`
- `TRTMC_TRIATTN_OVERRIDE_DIVIDE_LENGTH=2048`
- `TRTMC_TRIATTN_OVERRIDE_RECENT_WINDOW=256`
- `TRTMC_TRIATTN_RUNTIME_BUCKET_ROWS=32`

On `aime25_7`, greedy 3000-token generation from the same bundle was
byte-identical between the forced-off and forced-on modes:

- outputs identical
- extracted answer identical
- only a small throughput difference from extra TriAttention state allocation

This was an important control:

- when compaction is disabled, the native runtime no longer shows a separate
  "TriAttention corrupts output before compaction" problem
- the old pre-compaction divergence against dense was indeed caused by using a
  non-matched bundle pair

### But the same-bundle dense control still missed HF on long sampled decoding

The next focused test used that same `tri32768` bundle with
`TRTMC_TRIATTN_FORCE_ENABLE=0` on the full sampled stop-on-answer recipe for the
known bad sample `aime25_2`.

Result:

- same-bundle forced-off control extracted `5`
- dense full-KV and HF reference both extracted `588`

That ruled out another attractive but incorrect explanation:

- the remaining full-benchmark gap is not only "bad compaction keep-sets"

Because `TRTMC_TRIATTN_FORCE_ENABLE=0` takes the runtime back to the normal
`KvCache` state in `decoder_plugin.cpp`, this miss must come from the engine
family/build side, not from the TriAttention runtime state object.

### Dynamic-KV profile rows became the leading build-time suspect

The stale long-window bundles were then inspected directly. The dense full-KV
bundle and the `tri32768` bundle did not even share the same dynamic-KV
optimization profiles:

- dense full-KV:
  `[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]`
- `tri32768`:
  `[3072, 6144, 12288, 24576, 32768]`

Because the same-bundle force-off control already missed `aime25_2`, the
coarse TriAttention-specific profile schedule became the leading build-time
suspect for the remaining long-window drift. That led to the next experiment:

- add an explicit builder override for dynamic-KV profile rows
- rebuild a fresh `tri32768` bundle with the dense-style full profile ladder
- re-test sampled force-off control accuracy before returning to compaction
  tuning

### Dense-engine hybrid bundle proved the profile-row issue was real

Instead of waiting for another full engine rebuild, a direct control bundle was
constructed by taking the exact dense full-KV engine plan and tokenizer assets
from
`artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-dense32768-dynkv-fp16-manual-fullkv.trtfb`
and grafting only the TriAttention metadata block plus
`triattention_stats.json` from the stale `tri32768` artifact.

The resulting hybrid bundle was:

- `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-denseengine-hybrid.trtfb`

This gave a same-family long-window comparison point with:

- the dense engine plan
- the dense dynamic-KV optimization profile ladder
- the current native TriAttention runtime path

That hybrid bundle immediately recovered the long sampled control behavior that
the stale `tri32768` bundle had lost.

On `aime25_2`, with `TRTMC_TRIATTN_FORCE_ENABLE=0` and the full sampled
stop-on-answer recipe:

- hybrid force-off control extracted `588`
- generated `8592` tokens

So the dense-engine hybrid proved that the earlier long-window miss was not
intrinsic to the native runtime. It was tied to the stale engine-family build.

### Repaired-engine TriAttention then recovered the first hard cases

Using the same hybrid bundle with TriAttention enabled:

- default policy (`kv_budget=3072`, `divide_length=128`) also extracted `588`
  on `aime25_2`
- conservative policy
  (`TRTMC_TRIATTN_OVERRIDE_KV_BUDGET=6144`,
  `TRTMC_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`,
  `TRTMC_TRIATTN_RUNTIME_BUCKET_ROWS=32`) also extracted `588`

That meant the repaired engine family plus native TriAttention could preserve
the dense/HF answer on the previously bad sample `aime25_2`.

### Conservative policy fixed the three focused long-reasoning regressions

The next focused validation used the repaired hybrid bundle on the previously
bad slice `aime25_7`, `aime25_12`, and `aime25_23`.

With the repaired hybrid bundle and the conservative runtime policy:

- `aime25_7 -> 821`
- `aime25_12 -> 510`
- `aime25_23 -> 610`

Those are the same extracted answers as the dense/HF references for that
focused slice.

This is the first point in the long-window investigation where all three of the
known bad focus cases were simultaneously correct under the native runtime.

The exact conservative runtime recipe that achieved that state was:

- `TRTMC_TRIATTN_FORCE_ENABLE=1`
- `TRTMC_TRIATTN_OVERRIDE_KV_BUDGET=6144`
- `TRTMC_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`
- `TRTMC_TRIATTN_RUNTIME_BUCKET_ROWS=32`

That repaired-engine conservative recipe is now the right candidate for the
next full 30-sample apples-to-apples benchmark:

- same hybrid bundle as dense control with `TRTMC_TRIATTN_FORCE_ENABLE=0`
- same hybrid bundle as TriAttention run with the conservative policy above
- HF eager as the external reference

### A benchmark harness bug was then found in the seed schedule

The first full hybrid-bundle benchmark replay later showed a new regression on
`aime25_7`, but that run exposed a benchmark bug rather than a clean model
regression.

HF reference generation in
`python/tensorrt_model_connect/families/qwen/benchmark_qwen3_8b_aime25_vs_hf.py`
already reseeded per sample:

- `torch.manual_seed(seed + row_idx)` for the single-GPU case

But the TRT-side dataset benchmark binary was still reusing the exact same seed
for every row in a multi-sample run.

That meant the earlier "full benchmark" comparison was not even using the same
sampling schedule across HF and TRT. The benchmark runner was corrected so that
the TRT side now also advances the configured base seed by `sample_idx`.

### Seed fix changed the interpretation of the `aime25_7` regression

After rebuilding `trtmc_dataset_benchmark` with the corrected per-row seed
schedule, the dense control was rechecked on `aime25_7` as a standalone sample:

- seed `1234` still produced `821`
- seed `1240` also produced `821`

So the seed fix did not break the previous single-sample proof.

Then the crucial sequence-dependent focused replay was repeated with the rebuilt
benchmark binary:

- dataset: `aime25_2`, then `aime25_7`
- dense control: same hybrid bundle with `TRTMC_TRIATTN_FORCE_ENABLE=0`

Result:

- `aime25_2 -> 588`
- `aime25_7 -> 821`

And that same focused replay stayed correct even with CUDA Graphs disabled:

- `TRTMC_DISABLE_CUDA_GRAPH=1`
- still `aime25_2 -> 588`
- still `aime25_7 -> 821`

This invalidated the earlier "CUDA Graph reuse is the remaining root cause"
hypothesis. Once the benchmark seed schedule was corrected, the focused dense
control reproduced the right answer path with and without CUDA Graphs.

### Conservative TriAttention also passed the corrected focused replay

The same corrected focused replay (`aime25_2`, then `aime25_7`) was then run on
the native TriAttention path using the conservative runtime policy:

- `TRTMC_TRIATTN_FORCE_ENABLE=1`
- `TRTMC_TRIATTN_OVERRIDE_KV_BUDGET=6144`
- `TRTMC_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`
- `TRTMC_TRIATTN_RUNTIME_BUCKET_ROWS=32`

Result:

- `aime25_2 -> 588`
- `aime25_7 -> 821`

At that point, both dense and conservative TriAttention were again aligned on
the key focused pair under the corrected benchmark runner.

The next required step from that state is a fresh full 30-sample apples-to-
apples rerun using:

- the corrected TRT benchmark binary with per-row seed advancement
- dense control = hybrid bundle with `TRTMC_TRIATTN_FORCE_ENABLE=0`
- TriAttention = same hybrid bundle with the conservative runtime policy
- HF eager as the external reference

### Prompt mismatch later invalidated part of the focused `aime25_7` story

While chasing the remaining full-benchmark miss on `aime25_7`, I discovered
that the old standalone sample file used for several earlier spot checks did
not contain the same prompt string as the current benchmark dataset row.

The old standalone prompt began with:

- `You are given a math problem.`

The current benchmark prompt begins directly with the problem statement and the
final-answer instruction:

- `The twelve letters ...`
- `Please reason step by step, and put your final answer within \boxed{}`

That means several earlier "standalone sample 7" checks were not strict
apples-to-apples comparisons against the current benchmark harness. The dense
and TriAttention results on those old prompt files are still useful as local
signals, but they cannot be treated as proof for the current benchmark recipe.

### Current-prompt standalone checks changed the interpretation again

After extracting `aime25_7` directly from the current benchmark prompt file, I
reran the same-family dense control and conservative TriAttention as strict
single-sample standalone runs.

Dense control, current prompt:

- seed `1234` -> `pred_answer = 1`
- seed `1240` -> `pred_answer = 41`

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu2.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1240_gpu1.jsonl`

This was an important correction. The dense current-prompt path is already
wrong on `aime25_7` for these seeds as a standalone sample, so the later
`aime25_6 -> aime25_7` replay that produced `41` is no longer evidence of a
sequence-state corruption by itself.

### Conservative TriAttention still diverges further on the same prompt

Using the exact same current benchmark prompt row and the same hybrid bundle,
the conservative TriAttention path produced:

- seed `1234` -> `pred_answer = 68`

Artifact:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2.jsonl`

So the current sharp parity statement is:

- dense current-prompt standalone at seed `1234` gives `1`
- TriAttention current-prompt standalone at the same seed gives `68`
- gold answer remains `821`

That means the remaining problem is no longer "TriAttention loses parity while
dense stays correct" on this case. On the current benchmark recipe, both paths
are already off the gold answer on `aime25_7`, and TriAttention departs even
further from the dense answer on the exact same prompt and seed.

The unfinished seed-`1240` TriAttention run only reached pipeline load and did
not produce a sample row, so it is not evidence either way:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1240_gpu3.log`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1240_gpu3.jsonl`

### Current working interpretation

At this point the investigation has two distinct questions:

- why the current-prompt dense long-window path misses `aime25_7` under these
  seeds
- why conservative TriAttention departs from that dense path on the exact same
  prompt and seed

The next valid parity step is therefore not another broad benchmark rerun. It
is a same-prompt, same-seed runtime diff between dense and TriAttention on the
current benchmark sample until the first real divergence is localized.

### Same-prompt boundary runs localized the divergence to the first compaction

Using the current benchmark prompt row for `aime25_7` and the same hybrid
bundle:

- dense force-off, seed `1234`, `max_new_tokens=7000`
- conservative TriAttention, same seed, `max_new_tokens=7000`

produced byte-identical text:

- same `pred_answer = 10`
- same `generated_tokens = 7000`
- exact string equality on the generated text

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu1_max7000.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max7000.jsonl`

This ruled out any "prefill mismatch" or "early decode mismatch" theory for
this sample. The dense and TriAttention paths are identical through 7000
generated tokens.

Then a first-compaction abort run with keep-dump enabled showed the exact first
compaction point:

- `planned_prompt_length = 158`
- `prompt_end_position = 158`
- first compaction at `absolute_position = 7168`
- `keep_count = 6144`

So the first compaction happens after exactly:

- `7168 - 158 = 7010` generated tokens

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_firstcomp`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_firstcomp_abort.log`

That means the `7000`-token equality checkpoint is only 10 generated tokens
before the first compaction.

### Divergence appears immediately after the first compaction window

At `max_new_tokens=7600`, dense and TriAttention already diverge on the same
prompt and seed:

- dense `pred_answer = 10`
- TriAttention `pred_answer = 10`
- but text is no longer equal
- first textual difference is still the same `char=24807` boundary observed in
  the full outputs

Since the exact `7000`-token text length is `24690`, the first difference
appears only:

- `24807 - 24690 = 117` characters

after the last pre-compaction-equal checkpoint.

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu2_max7600_rerun.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max7600.jsonl`

This sharply localizes the problem: the remaining divergence begins almost
immediately after the first compaction, not thousands of tokens later.

### GPU-specific selection/compaction is no longer a viable root cause

The same `7600`-token current-prompt run was repeated with the entire GPU
TriAttention fast path disabled:

- `TRTMC_TRIATTN_DISABLE_GPU_SELECT=1`
- `TRTMC_TRIATTN_DISABLE_GPU_COMPACT=1`
- `TRTMC_TRIATTN_DISABLE_GPU_STATE=1`

Result:

- host-only TriAttention and normal GPU TriAttention were byte-identical
  at `7600` tokens

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_hostonly_max7600.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max7600.jsonl`

So the remaining issue is not in:

- the CUDA selector kernel
- the CUDA compaction kernel
- GPU-only state bookkeeping

It is shared host/GPU TriAttention logic above that layer.

### The first-compaction keep-set disagrees with the upstream selector on this sample

The exact first-compaction K-cache snapshot for `aime25_7` was replayed through
the local upstream per-head selector implementation in:

- `artifacts/triattention/upstream/triattention/vllm/runtime/selector_hf.py`

using both:

- `artifacts/triattention/qwen3_8b_aime25.pt`
- `artifacts/triattention/upstream/triattention/calibration/for_aime25_experiment/qwen3_8b.pt`

Both produced the same result:

- exact head matches: `0 / 8`
- mean Jaccard overlap: `0.778429853926756`

The protected prefix and recent tail still agree, but the selected interior
rows differ materially.

This is the first decisive proof that, on the current sample and compaction
state, the native selector semantics are still not matching the upstream
selector semantics.

So the working diagnosis is now:

- the first real divergence happens in the first compaction window
- host/GPU native paths agree with each other
- native keep-set does **not** agree with upstream keep-set on this sample

The next debugging step from here is score-level comparison:

- dump native per-layer / aggregated score values for the first compaction
- compare them directly against the upstream selector scores on the same
  snapshot
- identify whether the mismatch is in raw scoring, per-head grouping, or
  cross-layer aggregation

### Later correction: selector and compaction are now cleared on the aggressive sample7 path

The previous diagnosis above was too pessimistic. After replaying the same
`aime25_7` first-compaction snapshot more carefully, the remaining bug is no
longer in the selector itself.

First, the attempted "reduce stats to runtime KV heads" patch was the wrong
direction and was reverted. On the exact `sample7` first-compaction snapshot,
the original native `32`-score-head semantics match the official upstream
selector exactly:

- simulated original native vs upstream official selector: exact head matches
  `8 / 8`
- mean Jaccard overlap: `1.0`

That proof used the real dumped pre-compaction cache from:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert4.json.layer00.bin`
  through
- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert4.json.layer35.bin`

and the real runtime keep dump:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert4.json`

Second, the real C++ runtime keep-set is also exact with respect to the native
math on that same dump:

- runtime keep-set vs direct native-math replay: exact head matches `8 / 8`
- mean Jaccard overlap: `1.0`

So by this point the aggressive `sample7` first compaction has:

- official upstream selector parity
- native runtime selector parity

Third, the post-compaction cache contents are also exact gathers of the
selected rows.

Using the first-compaction dump with explicit pre/post K/V cache snapshots:

- `artifacts/triattention/investigation-2026-04-17/sample7_currentprompt_seed1234_debugprobe_revert5.json`

all `36` layers passed exact gather checks for both:

- `K`
- `V`

That means:

- selected indices are correct
- `K` repack is correct
- `V` repack is correct

Fourth, the boundary localization still holds on the aggressive operating
point:

- dense force-off and TriAttention are byte-identical through
  `max_new_tokens=3000`
- at `max_new_tokens=3400`, they diverge
- first text difference is still at `char=11217`

Artifacts:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu1_max3000_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max3000_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_force0_sample7_currentprompt_seed1234_gpu1_max3400_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max3400_revert.jsonl`

Finally, the current aggressive-path host-only replay still matches the normal
GPU TriAttention replay exactly at `3400` tokens:

- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_hostonly_max3400_revert.jsonl`
- `artifacts/triattention/investigation-2026-04-17/tri32768_hybrid_tri_sample7_currentprompt_seed1234_gpu2_max3400_revert.jsonl`

So the updated diagnosis is:

- the remaining `sample7` divergence still begins in the first compaction
  window
- but it is **not** caused by selector mismatch
- and it is **not** caused by incorrect K/V gather
- and it is **not** a GPU-only kernel bug

At this point the remaining explanations are narrower:

- genuine quality loss from the aggressive compression operating point itself
- or a downstream runtime effect outside selector/repack that still has not
  been identified

## Files most directly responsible for the final result

- `src/runtime/models/<family>/triattention_kv_cache.cpp`
- `src/runtime/models/<family>/triattention_kernels.cu`
- `src/runtime/models/<family>/triattention_kv_cache.h`
- `tests/builder/test_triattention_runtime.py`

## Short version

TriAttention only became accuracy-safe after the native scorer was aligned with
the upstream direct-`K_rot` semantics. Before that, the runtime was close in
structure but wrong in the core scoring math. Once that was fixed, the native
dynamic-KV C++ path achieved the intended state on the corrected pilot:

- dense-quality answers
- real native compaction
- real throughput gain

## 2026-04-18: later-compaction absolute-position bug isolated and fixed

The full `30`-sample AIME25 rerun had shown a severe TriAttention collapse even
though the short hard probes were still correct. At this point the next step
was to stop sweeping policy knobs and go back to direct tensor-level diffs.

### What remained true before the fix

On the sampled `aime25_7` repro:

- first compaction trigger was at absolute decode position `3200`
- sampled-token divergence happened before compaction `2`
- compaction `1` selection still matched the upstream selector exactly
- compaction `1` post-compaction `K`/`V` caches were still exact gathers of the
  pre-compaction cache

So compaction `1` had already been ruled out as a selector or gather bug.

### The concrete later-compaction bug

While reviewing `TriAttentionKvCache`, both selector paths were found to be
precomputing trig phases from the **current compacted row count** instead of the
true absolute decode position:

- `select_keep_indices_host()`
- `select_keep_indices_gpu()`

They used:

- `round_start = total_tokens`

instead of:

- `round_start = absolute_position_`

That mistake is silent on compaction `1` because, before any rows are dropped,
`total_tokens == absolute_position_`. But it becomes wrong on all later
compactions after the cache has already been compacted once.

The fix was to switch both host and GPU trig-prep paths to
`absolute_position_`.

### Live proof after the fix

Using the current sampled `aime25_7` repro and replaying the dumped score cache
through the upstream selector:

- compaction `2` dump:
  `artifacts/triattention/investigation-2026-04-18/profile_switch/sample7_compaction2_postfix_dump`
  - absolute position `3328`
  - exact head matches `8 / 8`
  - mean Jaccard `1.0`
- fresh compaction `3` dump from the current binary:
  `artifacts/triattention/investigation-2026-04-18/profile_switch/sample7_compaction3_currentfix_dump`
  - absolute position `3456`
  - exact head matches `8 / 8`
  - mean Jaccard `1.0`

The later-compaction cache gather was then checked directly on that same live
compaction-`3` dump using the recorded `keep_indices_by_head` together with the
dumped pre/post packed `K`/`V` cache snapshots:

- layers checked: `36 / 36`
- heads checked per layer: `8 / 8`
- exact gather matches:
  - `K`: yes
  - `V`: yes

So compaction `3` on the fixed path now also has:

- upstream selector parity
- exact native post-compaction `K` gather
- exact native post-compaction `V` gather

This also explained a confusing intermediate result: the old artifact

- `artifacts/triattention/qwen3-8b-nonflash/diff/current_native_r128_compact3_afterpatch_postcheck.json`

was stale from `2026-04-17`, so its compaction-`3` mismatch was no longer a
valid live signal after the absolute-position fix.

### Updated diagnosis after the fix

By this point:

- compaction `1` selector matches upstream exactly
- compaction `1` K/V gather matches exactly
- compaction `2` selector matches upstream exactly
- compaction `3` selector matches upstream exactly

So the major later-compaction selector bug is fixed. The remaining question is
how much of the full-benchmark accuracy collapse that bug was responsible for.
The next active step is a fresh full `30`-sample TriAttention rerun on the
fixed binary, using the same `upstream_plain` prompt recipe and decode settings
as the prior dense/HF comparison.

### Full-benchmark rerun after the absolute-position fix

The fixed `3072/128` native rerun was then launched on the same full
`upstream_plain` AIME25 benchmark recipe. By the time the investigation moved
 back to targeted diff testing, the live artifact had reached:

- result file:
  `artifacts/triattention/qwen3-8b-aime25-vs-hf-fullkv-plain-e2e-2026-04-18-currentfix/tri_results.jsonl`
- rows completed: `29 / 30`
- correct: `14 / 29`
- partial accuracy: `0.4828`
- mean decode throughput across completed rows: `77.33 tok/s`

This is a real improvement over the earlier broken full run (`0.267`), so the
absolute-position fix mattered materially. But it is still far from dense/HF
parity, so there must be either another implementation issue or a remaining
policy gap.

Comparing the completed fixed native rows against the dense/HF full-benchmark
reference, the following rows were dense-correct and HF-correct but still
native-wrong:

- `aime25_7`
- `aime25_9`
- `aime25_10`
- `aime25_11`
- `aime25_12`
- `aime25_13`
- `aime25_14`
- `aime25_15`
- `aime25_18`
- `aime25_20`
- `aime25_21`
- `aime25_22`
- `aime25_25`
- `aime25_28`
- `aime25_29`

The earliest clean failing probe after the fix is therefore `aime25_7`, and the
next independent failing probe is `aime25_9`.

### Upstream control checks on failing rows

To separate native-runtime bugs from budget/policy limitations, exact-prompt
upstream TriAttention worker probes were launched on the same benchmark prompt
format.

On `aime25_7` with the official upstream TriAttention worker at `3072/128`:

- output file:
  `tmp/upstream_worker_sample7_tri_local/shard00/run000.jsonl`
- result: wrong
- predicted boxed answer: `271`
- gold answer: `821`

On `aime25_12` with the official upstream TriAttention worker at `3072/128`:

- output file:
  `tmp/upstream_worker_sample12_tri_local/shard00/run000.jsonl`
- result: also not correct
- gold answer: `510`
- no clean final boxed answer was extracted

So `aime25_7` and `aime25_12` are not enough by themselves to prove another
native-only implementation mismatch at the aggressive `3072/128` operating
point. They are at least partly compatible with an upstream policy/quality
limit.

### New sample9 compaction anomaly to resolve next

The next clean candidate for a native-only bug became `aime25_9`, because dense
and HF both solve it while the fixed native `3072/128` run does not:

- gold answer: `62`
- fixed native current rerun answer: `2`
- dense full-KV answer: `62`
- HF eager answer: `62`

Two targeted native sample9 probes were then started:

- a `5000`-token native standalone run with
  `TRTMC_TRIATTN_DUMP_KEEP_PATH=/workspace/tensorrt-model-connect/tmp/aime25_9_plain_compaction1_dump`
  and `TRTMC_TRIATTN_ABORT_AFTER_DUMP=1`
- a focused trace around positions `3190..3220`

The first surprising result is that the `5000`-token standalone native run
finished without ever writing the requested compaction dump:

- output file:
  `tmp/aime25_9_plain_compaction1_abort.jsonl`
- generated tokens: `5000`
- predicted answer at cutoff: `3`
- expected dump file:
  `tmp/aime25_9_plain_compaction1_dump`
- observed status: dump file absent

That is suspicious, because the same bundle reports live TriAttention init with:

- `kv_budget=3072`
- `divide_length=128`
- `count_prompt_tokens=1`
- `protect_prefill=1`

and earlier sample7 traces from the same runtime clearly show the cache already
compacted to `3072` rows by the `3200`-position window:

- trace file:
  `tmp/aime25_7_plain_tri_trace_3200_3600.jsonl`
- first traced row:
  - `position_before = 3200`
  - `rows_before = 3072`
  - `rows_after = 3104`

So the next concrete question is not “which config sweep next?” but:

- does native sample9 actually compact live KV at the expected threshold?
- if it does, why was the compaction dump hook not hit?
- if it does not, what runtime condition is preventing compaction on this row?

### Sample9 compaction was then proven live

That sample9 compaction question was resolved by switching from the dump hook to
the built-in compaction profile logs.

Using the same native current bundle on the exact benchmark prompt:

- bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri12288-b3072-r128-dynkv-fp16-manual-current.trtfb`
- dataset row:
  `tmp/aime25_rescue_samples/aime25_09.jsonl`
- runtime flags:
  - `TRTMC_TRIATTN_FORCE_ENABLE=1`
  - `TRTMC_TRIATTN_PROFILE=1`
  - `TRTMC_TRIATTN_DEBUG=1`

The profile logs showed native compaction firing exactly on the expected
schedule:

- compact `#1`: `abs_pos=3200`, `old_rows=3200`, `kept_rows=3072`
- compact `#2`: `abs_pos=3328`, `old_rows=3200`, `kept_rows=3072`
- compact `#3`: `abs_pos=3456`, `old_rows=3200`, `kept_rows=3072`
- compact `#4`: `abs_pos=3584`, `old_rows=3200`, `kept_rows=3072`

So the missing sample9 dump file was not evidence that compaction was disabled.
Compaction is definitely occurring on sample9; the problem is elsewhere.

### Sample9 diverges only after the first compaction step

The next check compared the exact same bundle in two modes on the sample9
prompt:

- TriAttention enabled:
  `TRTMC_TRIATTN_FORCE_ENABLE=1`
- same bundle, force-off control:
  `TRTMC_TRIATTN_FORCE_ENABLE=0`

Both runs used:

- `--max-new-tokens 3300`
- `--temperature 0.6`
- `--top-k 20`
- `--top-p 0.95`
- `--min-p 0.0`
- `--seed 1234`

and recorded the same trace window:

- `TRTMC_TEXT_STEP_TRACE_START_POS=3190`
- `TRTMC_TEXT_STEP_TRACE_END_POS=3220`

Trace artifacts:

- TriAttention trace:
  `tmp/aime25_9_tri_trace_3190_3220.jsonl`
- force-off trace:
  `tmp/aime25_9_force0_trace_3190_3220.jsonl`

What that diff proved:

- positions `3190..3198`: logits match exactly
- position `3199`:
  - argmax and top logits still match exactly
  - only the cache-row count changes
  - TriAttention: `rows_after = 3072`
  - force-off: `rows_after = 3200`
- position `3200`:
  - first actual logit drift appears
  - argmax is still the same in both paths
  - differences are small (`~1e-2` scale in top logits)
- positions `3190..3220`:
  - argmax stays identical for the whole traced window

So sample9 is not another “explodes immediately at compaction” failure. It is a
much subtler case:

- native TriAttention matches the force-off control exactly until the first
  compaction boundary
- the first numerical drift begins one token *after* compaction
- that drift is initially small and does not immediately change argmax

That shifts the next investigation step again. The remaining question is no
longer “does sample9 compact?” but instead:

- are the selected rows themselves still aligned with upstream on sample9?
- if yes, is the residual accuracy gap on sample9 also reproducible upstream at
  the same budget?
- if not, what still differs in native sample9 selection or repack semantics?

### Fixed `3072/128` full benchmark completed at `14 / 30`

The full post-fix aggressive rerun eventually finished:

- result file:
  `artifacts/triattention/qwen3-8b-aime25-vs-hf-fullkv-plain-e2e-2026-04-18-currentfix/tri_results.jsonl`
- completed rows: `30 / 30`
- correct: `14 / 30`
- final accuracy: `0.4667`
- mean decode throughput: `77.18 tok/s`

Wrong rows in the completed fixed rerun:

- `aime25_7`
- `aime25_9`
- `aime25_10`
- `aime25_11`
- `aime25_12`
- `aime25_13`
- `aime25_14`
- `aime25_15`
- `aime25_18`
- `aime25_20`
- `aime25_21`
- `aime25_22`
- `aime25_25`
- `aime25_28`
- `aime25_29`
- `aime25_30`

So the absolute-position fix rescued a large chunk of the benchmark, but it did
not restore full accuracy parity on the aggressive bundle.

### Same-family `6144/1024` sample9 also fails in the live full run

The fair same-family run was already live on:

- bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-denseengine-hybrid.trtfb`
- runtime overrides:
  - `TRTMC_TRIATTN_OVERRIDE_KV_BUDGET=6144`
  - `TRTMC_TRIATTN_OVERRIDE_DIVIDE_LENGTH=1024`
  - `TRTMC_TRIATTN_RUNTIME_BUCKET_ROWS=32`

and by the time the focused debugging resumed it had reached:

- result file:
  `artifacts/triattention/qwen3-8b-aime25-vs-hf-fullkv-plain-e2e-2026-04-18-override6144/tri_results.jsonl`
- rows completed so far: `9`
- correct so far: `7`
- partial accuracy so far: `0.7778`

Crucially, sample9 is still wrong even on this more conservative same-family
path:

- `aime25_9`
  - predicted answer: `-3`
  - gold answer: `62`
  - generated tokens: `38912`

So sample9 remains a live failure on both:

- the aggressive current bundle
- the same-family conservative `6144/1024` hybrid run

### Same-family `6144/1024` sample9 reaches first compaction cleanly

To inspect the fair path directly, sample9 was then rerun on the same-family
hybrid bundle with the same `6144/1024` overrides and a focused trace window:

- trace file:
  `tmp/aime25_9_hybrid6144_tri_trace_7150_7190.jsonl`

That trace shows the first compaction exactly where it should happen:

- position `7150`:
  - `rows_before = 7168`
  - `rows_after = 7168`
- position `7167`:
  - `rows_before = 7168`
  - `rows_after = 6144`
- position `7190`:
  - `rows_before = 6176`
  - `rows_after = 6176`

So on the fair hybrid path as well:

- compaction is definitely active
- the first compaction boundary is clean and on schedule

The next pending comparison is the same hybrid bundle with
`TRTMC_TRIATTN_FORCE_ENABLE=0` over the same `7150..7190` window, to see whether
the first fair-path numerical drift is again only a small post-compaction
difference or something more severe.

### Same-family `6144/1024` sample9 vs force-off: same pattern again

That pending force-off comparison was then completed on the exact same hybrid
bundle and trace window:

- TriAttention trace:
  `tmp/aime25_9_hybrid6144_tri_trace_7150_7190.jsonl`
- force-off trace:
  `tmp/aime25_9_hybrid6144_force0_trace_7150_7190.jsonl`

Diff result:

- first any difference:
  - position `7167`
  - rows only
  - TriAttention: `rows_after = 6144`
  - force-off: `rows_after = 7168`
  - logits still match exactly
- first actual logit drift:
  - position `7168`
  - argmax still the same in both paths
  - top-id ordering still the same
  - top-logit deltas are again small (`~1e-2` to `1e-1`)
- across the whole traced window `7150..7190`:
  - argmax never changes between TriAttention and force-off

So the fair-path same-family result matches the earlier aggressive-bundle
sample9 finding:

- TriAttention stays numerically identical up to the compaction step itself
- the first drift appears one token later
- the initial drift is small and does not immediately flip argmax

This again argues against a gross native compaction bug on sample9. The
remaining miss is increasingly consistent with a long-horizon quality loss from
compression at this budget, unless the pending upstream sample9 control proves
otherwise.

### Upstream sample9 `3072/128` control also fails

To get a faster upstream control than the original shard worker, a dedicated
single-sample upstream harness was added:

- script: local scratch harness `run_upstream_triattention_single.py`

It loads the same local Qwen3-8B snapshot, applies the official
`apply_triattention_patch()`, and stops generation once a boxed answer appears
or the configured token cap is hit.

Running that harness on sample9 at the aggressive upstream point:

- sample file:
  `tmp/aime25_rescue_samples/aime25_09.jsonl`
- budget:
  `3072`
- divide length:
  `128`
- max new tokens:
  `8192`
- output:
  `tmp/upstream_single_sample9_b3072_d128.json`

Result:

- sample id: `aime25_9`
- gold answer: `62`
- predicted answer extracted: `""`
- generated tokens: `8192`
- no valid final boxed answer was extracted

So sample9 is **not** a clean native-only failure at `3072/128`. Official
upstream TriAttention also fails to produce the correct answer for this row
under the same prompt recipe and a long boxed-answer run.

That pushes the interpretation further toward:

- sample9 is a real quality miss for this operating point, not a clear native
  implementation bug

The next active upstream check is therefore sample9 at the more conservative
`6144/1024` setting, which is the operating point currently being compared
against the fair same-family native hybrid path.

### Upstream sample9 `6144/1024` control also fails

The same upstream single-sample harness was then run on sample9 at the more
conservative operating point:

- sample file:
  `tmp/aime25_rescue_samples/aime25_09.jsonl`
- budget:
  `6144`
- divide length:
  `1024`
- max new tokens:
  `8192`
- output:
  `tmp/upstream_single_sample9_b6144_d1024.json`

Result:

- sample id: `aime25_9`
- gold answer: `62`
- predicted answer extracted: `""`
- generated tokens: `8192`
- no valid final boxed answer was extracted

So sample9 also fails in the official upstream Python patch at the same
`6144/1024` point that is currently being used for the fair same-family native
hybrid run. This is a stronger discriminator than the earlier `3072/128`
result:

- sample9 remains a real quality miss for TriAttention on this prompt recipe
  even at the conservative `6144/1024` setting
- the native miss on sample9 is therefore still not sufficient evidence of a
  native implementation bug

### Same-family `6144/1024` full run: early partial still breaks the same rows

The full same-family native hybrid benchmark at `6144/1024` is still running:

- output:
  `artifacts/triattention/qwen3-8b-aime25-vs-hf-fullkv-plain-e2e-2026-04-18-override6144/tri_results.jsonl`

At the latest partial checkpoint:

- rows completed: `12 / 30`
- correct: `8 / 12`
- partial accuracy: `0.6667`
- mean decode throughput so far: `69.14 tok/s`

Wrong rows so far:

- `aime25_7` -> predicted `16`, gold `821`
- `aime25_9` -> predicted `-3`, gold `62`
- `aime25_10` -> predicted `70`, gold `81`
- `aime25_11` -> predicted `36`, gold `259`

This is materially better than the old aggressive `3072/128` benchmark result,
but it still breaks the same early hard rows that motivated the deeper
investigation.

### Next diff artifact in flight: native sample9 full token trace

To move from aggregate scores to token-level diffing, a dedicated native trace
run was started for sample9 on the same-family hybrid path:

- bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-denseengine-hybrid.trtfb`
- runtime overrides:
  `kv_budget=6144`, `divide_length=1024`, `bucket_rows=32`
- trace target:
  `tmp/aime25_9_hybrid6144_fulltrace_0_7180.jsonl`
- benchmark output:
  `tmp/aime25_9_hybrid6144_trace_run.jsonl`

That trace is intended to capture the exact native generated token ids from the
first decode step through the first `6144/1024` compaction boundary, so the
same token prefix can then be replayed teacher-forced through:

- dense HF
- upstream Python TriAttention

using the new helper:

- local scratch helper `trace_hf_teacher_forced.py`

The goal of that replay is to decide whether the remaining gap is:

- a native-vs-upstream implementation divergence, or
- a true quality loss shared by both native and upstream TriAttention under
  this benchmark recipe.

### Native sample9 `8192/1024` still fails

To test whether sample9 would recover under a larger decode budget on the
same-family native hybrid path, a dedicated native run was executed with:

- bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-denseengine-hybrid.trtfb`
- runtime overrides:
  `kv_budget=8192`, `divide_length=1024`, `bucket_rows=32`
- sample file:
  `tmp/aime25_rescue_samples/aime25_09.jsonl`
- output:
  `tmp/aime25_9_hybrid8192_stop.jsonl`

Result:

- sample id: `aime25_9`
- gold answer: `62`
- predicted answer: `-1848`
- generated tokens: `38912`
- decode throughput: `45.04 tok/s`

So sample9 is still wrong even at native `8192/1024`. This does **not** prove
that larger budgets can never recover the row, but it does rule out the
simplest “just move from `6144` to `8192`” explanation for the sample9 miss.

### Sample9 boundary replay exposed a prompt-token accounting mismatch

The new teacher-forced replay helper:

- local scratch helper `trace_hf_teacher_forced.py`

was first used to compare the native sample9 trace against dense HF and the
upstream Python patch around the first `6144/1024` compaction boundary:

- native trace:
  `tmp/aime25_9_hybrid6144_fulltrace_0_7180.jsonl`
- dense replay:
  `tmp/dense_teacherforced_sample9_trace_fast_7160_7180.jsonl`
- upstream replay:
  `tmp/upstream_teacherforced_sample9_6144_trace_fast_7160_7180.jsonl`

The first replay attempt used `count_prompt_tokens=True` in the Python patch,
matching the current builder/runtime config export. That exposed an immediate
semantic mismatch:

- sample9 prompt length is `157` tokens
- dense HF raw cache length at position `7166` is `7323`
- native `rows_before` at the same step is `7168`

That `7323 - 7168 = 155` gap is essentially the prompt length (modulo bucketed
row rounding in the native trace), which means the native trigger is behaving
like **decode-only accounting** even though the exported config still says:

- `count_prompt_tokens = true`

This makes sense after reading the native runtime code:

- `cache_length_` only grows in `append_present_to_cache()`
- prefill tokens do not contribute to `cache_length_`
- compaction triggering uses `cache_length_ >= compaction_trigger_length()`
- relevant code: `src/runtime/models/<family>/triattention_kv_cache.cpp` around the
  append/cache-length and compaction-trigger paths

So the earlier Python replay with `count_prompt_tokens=True` was not an
apples-to-apples comparison against the native runtime behavior.

### Aligned sample9 replay: native and upstream match through the boundary

The upstream replay was rerun with:

- `--no-count-prompt-tokens`

using:

- aligned upstream replay:
  `tmp/upstream_teacherforced_sample9_6144_noprompt_trace_fast_7160_7180.jsonl`
- aligned hidden dumps:
  `tmp/upstream_teacherforced_sample9_6144_noprompt_hidden_fast_7160_7180`

Key result on the window `7160..7180`:

- native and upstream-no-prompt have **no argmax mismatch** in this window
- native and dense first differ on argmax at position `7177`
- native and upstream-no-prompt stay argmax-identical through the whole window

Representative positions:

- position `7167` (compaction step):
  - native argmax: `7196`
  - upstream-no-prompt argmax: `7196`
  - native rows: `7168 -> 6144`
  - upstream-no-prompt rows: `7324 -> 6144`
- position `7168` (first post-compaction token):
  - native argmax: `356`
  - upstream-no-prompt argmax: `356`
- position `7177`:
  - native argmax: `30`
  - dense argmax: `476`
  - upstream-no-prompt argmax: `30`

So for sample9, once prompt-token accounting is aligned, the native runtime
matches the upstream Python patch at the token-logit level through the first
post-compaction window.

### Dense-vs-TriAttention hidden states stay identical until compaction, then drift slightly

Dense hidden dumps:

- `tmp/dense_teacherforced_sample9_hidden_fast_7160_7180`

Aligned upstream-no-prompt hidden dumps:

- `tmp/upstream_teacherforced_sample9_6144_noprompt_hidden_fast_7160_7180`

Comparing dense vs upstream-no-prompt hidden states:

- positions `7166` and `7167`:
  - relative L2 difference is exactly `0.0` across all dumped layers
- position `7168`:
  - first post-compaction hidden drift appears
  - max sampled relative L2 across layers is about `1.05%`
- position `7171`:
  - max relative L2 is about `1.71%`
- position `7177`:
  - max relative L2 is about `2.83%`

This is the cleanest sample9 proof so far:

- dense and TriAttention are numerically identical right up to the compaction
  boundary
- the first dense-vs-TriAttention hidden-state drift appears immediately after
  compaction
- the drift starts small
- native and upstream-no-prompt track each other on that same boundary window

So sample9 now looks much more like a **shared quality loss from compression**
than a remaining native implementation bug.

### Later correction: the teacher-forced replay harness was double-counting prompt tokens

The earlier "full native trace" HF/upstream replays were still contaminated by a
bug in the replay harness itself.

The root cause was in:

- local scratch helper `trace_hf_teacher_forced.py`

That helper always:

1. prefills the prompt through HF, and then
2. replays all rows from the native TRT trace as warmup/replay tokens

But the native TRT step trace already includes the prompt-side prefill steps.
So for full traces like:

- `tmp/aime25_7_currentfix_fulltrace_0_3480.jsonl`
- `tmp/aime25_9_hybrid6144_fulltrace_0_7180.jsonl`

the prompt was being counted twice in the HF/upstream replay.

The symptom was obvious in the replay outputs:

- native sample9 trace position `7150`
- replay dense/upstream rows_before `7307`
- prompt length `157`

That is exactly the prompt-length offset caused by replaying prompt rows after
the prompt had already been prefetched.

The harness was corrected so that when a native trace starts at position `0`,
the replay skips all trace rows below the prompt token count after the initial
HF prompt prefill.

### Corrected sample9 replay: native, dense HF, and upstream all stay aligned

After fixing the local scratch helper `trace_hf_teacher_forced.py`, sample9
was rerun on the exact same native token trace and window:

- native trace:
  `tmp/aime25_9_hybrid6144_fulltrace_0_7180.jsonl`
- corrected dense replay:
  `tmp/dense_teacherforced_sample9_hybrid6144_promptfix_gpu3_trace_7150_7190.jsonl`
- corrected upstream replay:
  `tmp/upstream_teacherforced_sample9_hybrid6144_promptfix_gpu3_trace_7150_7190.jsonl`
- corrected dense hidden dumps:
  `tmp/dense_teacherforced_sample9_hybrid6144_promptfix_gpu3_hidden_7150_7190`
- corrected upstream hidden dumps:
  `tmp/upstream_teacherforced_sample9_hybrid6144_promptfix_gpu3_hidden_7150_7190`

Corrected result:

- native vs dense first argmax diff: `None`
- native vs upstream first argmax diff: `None`
- dense vs upstream first argmax diff: `None`

Across the first compaction boundary:

- position `7167`:
  - native argmax `7196`
  - dense argmax `7196`
  - upstream argmax `7196`
- position `7168`:
  - native argmax `356`
  - dense argmax `356`
  - upstream argmax `356`
- positions `7177`, `7179`, `7180`:
  - all three paths still have identical argmax decisions

Dense vs upstream hidden-state comparison is also exact on the sampled
positions:

- `7150`, `7166`, `7167`, `7168`, `7176`, `7177`, `7179`, `7180`
  all have relative L2 `0.0`

So the earlier sample9 claim that native/upstream diverged after compaction was
an artifact of the replay bug, not a real runtime mismatch.

### Corrected sample7 replay: the previously reported `3455` divergence also disappears

The same corrected replay was then applied to sample7:

- native trace:
  `tmp/aime25_7_currentfix_fulltrace_0_3480.jsonl`
- corrected dense replay:
  `tmp/dense_teacherforced_sample7_currentfix_promptfix_gpu3_trace_3440_3479.jsonl`
- corrected upstream replay:
  `tmp/upstream_teacherforced_sample7_currentfix_promptfix_gpu3_trace_3440_3479.jsonl`
- corrected dense hidden dumps:
  `tmp/dense_teacherforced_sample7_currentfix_promptfix_gpu3_hidden_3440_3479`
- corrected upstream hidden dumps:
  `tmp/upstream_teacherforced_sample7_currentfix_promptfix_gpu3_hidden_3440_3479`

Corrected result:

- native vs dense first argmax diff: `None`
- native vs upstream first argmax diff: `None`
- dense vs upstream first argmax diff: `None`

Representative positions that had previously looked suspicious are now all
aligned on argmax:

- `3444` -> `43778`
- `3455` -> `4325`
- `3456` -> `6524`
- `3479` -> `387`

Dense vs upstream hidden states are not exactly identical on this sample window
(sampled relative L2 is a few percent on some positions), but the token-level
argmax path remains identical across native, dense, and upstream throughout the
entire dumped range.

So the earlier sample7 "native diverges at 3455 and snaps onto upstream" claim
was also caused by the same prompt-double-count bug in the replay harness.

### Current implication

The two strongest tensor-level diff targets checked after fixing the replay
harness now say the same thing:

- sample7 currentfix window: native == dense == upstream on argmax
- sample9 hybrid `6144/1024` window: native == dense == upstream on both
  argmax and sampled dense-vs-upstream hidden states

So the debugging target has moved again:

- the previously reported token-level divergence windows are no longer valid
  evidence of a native implementation bug
- any remaining benchmark-level misses now need either
  - a later trace window on the true failing sample/path, or
  - a fresh full-benchmark rerun on the trustworthy hybrid bundle after
    correcting the replay methodology

### Exact current-prompt sample7 is not a valid native-only isolation target

To avoid mixing old prompt recipes with the current benchmark prompt file, a
single-row repro was rebuilt directly from:

- `artifacts/triattention/qwen3-8b-aime25-vs-hf-hybrid-conservative-e2e-seedfix-2026-04-18/aime25_prompts.jsonl`
- sample id: `aime25_7`
- injected `seed_index = 6`
- seed: `1234`
- effective sample seed: `1240`
- same-family bundle:
  `artifacts/triattention/qwen3-8b-nonflash/qwen3-8b-tri32768-b3072-r128-dynkv-fp16-manual-denseengine-hybrid.trtfb`

Standalone results on that exact prompt/seed:

- force-off same-family control:
  `tmp/aime25_7_hybrid_force0_seed1234_stop.jsonl`
  - predicted answer: `41`
  - gold answer: `821`
- TriAttention `6144/1024`:
  `tmp/aime25_7_hybrid_tri6144_seed1234_stop.jsonl`
  - predicted answer: `247`
  - gold answer: `821`

So on the exact current benchmark prompt, `aime25_7` is not a clean
TriAttention-only regression. The same-family dense control is already wrong,
which means sample7 is not a trustworthy tensor-diff target for isolating a
native TriAttention bug.

### Exact current-prompt sample9: first live split is far after compaction

The next target was rebuilt from the same current prompt file:

- `tmp/aime25_9_hybrid_seedfix_seedidx8.jsonl`
- sample id: `aime25_9`
- injected `seed_index = 8`
- effective sample seed: `1242`

Using the same-family hybrid bundle with:

- force-off control:
  `tmp/aime25_9_hybrid_force0_fulltrace_0_10000.jsonl`
- TriAttention `6144/1024`:
  `tmp/aime25_9_hybrid_tri6144_fulltrace_0_10000.jsonl`

and matching `10000`-token no-stop runs:

- force-off output:
  `tmp/aime25_9_hybrid_force0_run_10000.jsonl`
- TriAttention output:
  `tmp/aime25_9_hybrid_tri6144_run_10000.jsonl`

The direct native-vs-native trace diff shows:

- first structural difference:
  - position `7167`
  - compaction only
  - force-off `rows_after = 7168`
  - TriAttention `rows_after = 6144`
  - logits still match
- first live sampled-token split:
  - position `7427`
  - force-off token id: `1430`
  - TriAttention token id: `1221`
- first live argmax split in those native traces:
  - also position `7427`

So on the exact current prompt, the first real live split is not at the
compaction boundary. The two native paths stay aligned for roughly `260` more
tokens after compaction before the first sampled-token deviation appears.

### Current-prompt sample9 replay: no clean native-only tensor drift before the split

The exact current-prompt sample9 TriAttention trace was then replayed
teacher-forced through:

- dense HF:
  `tmp/dense_teacherforced_sample9_seedfix_from_tri_7400_7460.jsonl`
- upstream Python TriAttention:
  `tmp/upstream_teacherforced_sample9_seedfix_from_tri_7400_7460.jsonl`

using the native trace:

- `tmp/aime25_9_hybrid_tri6144_fulltrace_0_10000.jsonl`

and the same current sample file:

- `tmp/aime25_9_hybrid_seedfix_seedidx8.jsonl`

The only earlier argmax discrepancy reported by this replay window is at
position `7404`, but that turns out to be a top-logit tie, not a meaningful
split:

- native dense and native tri both report:
  - top-1 `1477`
  - top-2 `7942`
  - equal top logits at that step
- dense HF replay and upstream replay report the same two tied tokens in the
  opposite order:
  - top-1 `7942`
  - top-2 `1477`
  - equal top logits at that step

That is tie-breaking noise, not evidence of a semantic native/runtime drift.

On the last shared live prefix before the first sampled split (`7426`), all
four views remain aligned in substance:

- native dense live
- native tri live
- dense HF replay
- upstream Python TriAttention replay

At position `7426` they all still have:

- token id `323`
- argmax token `400`
- same top-id ordering at the head:
  `400`, `1430`, `1221`, `1490`, `1779`

So the current exact-prompt sample9 replay again fails to expose a clean
native-only tensor bug before the first live split. The evidence currently
points toward a sampling-sensitive/shared compression effect, not an obvious
native runtime mismatch.

### Live upstream current-prompt sample9 also fails at `6144/1024`

To check that conclusion outside the native TRT runtime, the official upstream
Python TriAttention patch was run live on the exact same current prompt/seed:

- sample file:
  `tmp/aime25_9_hybrid_seedfix_seedidx8.jsonl`
- effective sample seed: `1242`
- stats:
  `artifacts/triattention/qwen3_8b_aime25.pt`
- budget:
  `6144`
- divide length:
  `1024`
- output:
  `tmp/upstream_live_sample9_seedfix_6144_d1024_seed1242_10000.json`

Result:

- sample id: `aime25_9`
- gold answer: `62`
- predicted boxed answer extracted: `""`
- generated tokens: `10000`
- no valid boxed final answer was produced in the capped run

So on the exact current benchmark prompt and seed, upstream Python
TriAttention also fails to produce a clean solved answer for sample9 at the
same `6144/1024` operating point.

That further weakens the case for a native-only runtime bug on this row. The
current evidence stack for sample9 is now:

- same-family native dense and TriAttention stay aligned until long after the
  first compaction
- corrected teacher-forced dense HF and upstream replay do not expose a clean
  native-only tensor drift before the first live split
- live upstream TriAttention on the exact same current prompt/seed also fails

The most defensible interpretation at this point is still that sample9 is
primarily a shared TriAttention quality loss on this benchmark recipe, not an
isolated native TRT implementation bug.

### Exact current-prompt sample10 shows the same delayed post-compaction split shape

To check whether sample9 was a one-off, the same exact-prompt trace cycle was
run on the next independent failing row:

- sample file:
  `tmp/aime25_10_hybrid_seedfix_seedidx9.jsonl`
- effective sample seed: `1243`
- same-family force-off trace:
  `tmp/aime25_10_hybrid_force0_fulltrace_0_8000.jsonl`
- same-family TriAttention `6144/1024` trace:
  `tmp/aime25_10_hybrid_tri6144_fulltrace_0_8000.jsonl`

The direct native-vs-native trace diff shows the same broad shape as sample9:

- first structural difference:
  - position `7167`
  - compaction only
  - force-off `rows_after = 7168`
  - TriAttention `rows_after = 6144`
- first live sampled-token split:
  - position `7477`
- first live argmax split:
  - also position `7477`

So sample10 also does **not** explode at the compaction boundary itself. It
stays aligned for roughly `300` more tokens after compaction before the first
sampled-token deviation appears. That further supports the interpretation that
the current failures are dominated by long-horizon compression effects rather
than a simple native compaction corruption bug.

### Exact current-prompt sample9 improves materially at `8192/1024`

Because the current evidence points more toward shared compression loss than a
native-only bug, the next targeted check was whether a more conservative
current-prompt operating point materially delays or removes the sample9 split.

Using the same exact current sample9 prompt and seed:

- sample file:
  `tmp/aime25_9_hybrid_seedfix_seedidx8.jsonl`
- effective sample seed: `1242`
- force-off trace:
  `tmp/aime25_9_hybrid_force0_fulltrace_0_10000.jsonl`
- TriAttention `8192/1024` trace:
  `tmp/aime25_9_hybrid_tri8192_fulltrace_0_10000.jsonl`
- TriAttention `8192/1024` short run:
  `tmp/aime25_9_hybrid_tri8192_run_10000.jsonl`

Compared with the old `6144/1024` result:

- at `6144/1024`
  - first live sampled-token split: `7427`
  - `10000`-token extracted answer: `21`
- at `8192/1024`
  - first structural difference: `9215` (first compaction boundary)
  - first live sampled-token split: `9745`
  - no argmax split before that sampled-token deviation
  - `10000`-token extracted answer: `062`

So on the exact current prompt, raising the budget from `6144` to `8192`
substantially improves sample9:

- the first split moves later by more than `2300` positions
- the `10000`-token extracted answer matches the dense control's extracted
  value

This is the strongest evidence so far that the remaining misses may be
recoverable by a more conservative TriAttention operating point, even though it
is still not enough for full parity.

### Exact current-prompt sample9 full run at `8192/1024` still misses late

The improved sample9 point was then rerun under the full benchmark-style capped
decode:

- output:
  `tmp/aime25_9_hybrid_tri8192_stop_full.jsonl`

Result:

- sample id: `aime25_9`
- gold answer: `62`
- predicted answer: `32`
- generated tokens: `25104`

So `8192/1024` materially improves the trajectory on current-prompt sample9,
but it still does not fully recover the row under the full long decode. That
keeps the main conclusion intact:

- the failure is now looking less like a native runtime bug
- and more like a quality-vs-budget tradeoff that still needs a better
  operating point for parity

### Sample2 dense-vs-HF split was mostly a sampler bug, then a precision gap

The next debugging cycle switched from config sweeps to direct dense-vs-HF diff
testing on the exact current-prompt sample2 repro:

- sample file:
  `tmp/aime25_2_seedfix_seedidx1.jsonl`
- dense trace:
  `tmp/aime25_2_densefullkv_torchsampler_trace.jsonl`
- captured HF token ids:
  `tmp/hf_sample2_seed1235_2400_ids.json`

The first important finding was that the old native sampler bridge was still
wrong even after moving to libtorch:

- CUDA `torch.multinomial` over the **full masked vocabulary** is not
  equivalent to sampling over the compressed kept-slice with the same
  renormalized probabilities.
- Direct container probes showed large mismatches for the same seed on the same
  sparse distributions. For example on the exact step2-style two-way
  distribution:
  - full-vocab sparse tensor: token `419`
  - compressed kept-slice tensor: token `279`

That proved the bridge had to sample over a reusable full-vocab probability
tensor, not the compressed kept subset.

The sampler was then fixed to:

- build the filtered kept set as before
- scatter kept probabilities back into a reusable full-vocab CUDA tensor
- call `at::multinomial(..., replacement=false, generator_)` on that full
  tensor

Regression coverage was added to lock in the sampler's sparse full-vocab
behavior.

After that fix:

- the first dense-vs-HF sampled-token split on exact sample2 moved from
  generated token `2` to generated token `48`
- the first `64` native generated tokens became:
  `2014, 11625, 279, 3491, ... , 29208, 32711, 1447, ...`

The next question was whether the remaining split at generated token `48`
(`32711` vs earlier HF `29985`) was still a sampler bug or real model drift.

That was answered with two direct probes:

1. Dense teacher-forced replay on the corrected no-prompt-last trace:

   - trace:
     `tmp/aime25_2_densefullkv_torchsampler_trace_nopromptlast.jsonl`
   - replay:
     `tmp/dense_teacherforced_sample2_torchsampler_653_704_nopromptlast.jsonl`

   Result:

   - dense TRT and dense HF argmax/top-k still match through positions
     `653..704`
   - so there is no early native-only tensor corruption on this window

2. Single-GPU HF generate probes at different dtypes:

   - HF `bfloat16`, eager, exact current prompt/seed:
     - generated token `48`: `29985`
     - processed probabilities at step `48`:
       `[0.47153, 0.25239, 0.16639, 0.10969]`
   - HF `float16`, eager, exact current prompt/seed:
     - generated token `48`: `32711`
     - first `64` generated tokens match the native dense trace exactly

So the remaining sample2 split after the sampler fix is not a TRT-only logic
bug. It is explained by precision:

- native dense bundle is `fp16`
- HF “golden” runs had been checked in `bf16`
- on this exact row, HF `float16` matches native dense while HF `bfloat16`
  samples a different token later in the run

This materially changes the interpretation of the dense-vs-HF gap:

- there **was** a real sampler bug in native
- after fixing it, the residual sample2 disagreement is best explained as
  `fp16` vs `bf16` sampling sensitivity, not an obvious native runtime defect

### Sparse exact CUDA sampler replaces the slow full-vocab torch bridge

The next step was to remove the large runtime cost of the temporary
full-vocab torch bridge without giving up parity.

Two direct probes established the target semantics:

1. CUDA `torch.multinomial(..., replacement=False)` for one sample matches the
   exponential-race form exactly:

   - `argmax(probs / exponential_(1.0))`

2. For contiguous float vocab tensors on our current GPU and Qwen3-8B vocab
   size, PyTorch's CUDA `exponential_` path is effectively:

   - one thread per vocab index
   - `curand_uniform4(&state).x` for that absolute index
   - generator offset increment of `4` per decode step

That led to a new exact sparse CUDA sampler:

- keep the existing host-side filter/top-p/top-k/min-p logic
- copy only the kept token ids and kept probabilities to GPU
- reproduce PyTorch's full-vocab RNG semantics for those absolute indices
- sample with the same exponential-race rule that PyTorch uses internally

The implementation is in:

- `src/runtime/models/<family>/sparse_multinomial_kernel.cu`
- `src/runtime/models/<family>/sampler.cpp`

New sampler regression coverage was added for:

- sparse full-vocab semantics
- offset progression across repeated draws
- a synthetic three-way case derived from the live sample2 first decode step

### Seed-index correction for dataset benchmark repros

One false alarm in the first live validation came from the benchmark driver:

- `examples/trtmc_dataset_benchmark.cpp` adds each sample's `seed_index` to the
  CLI `--seed`

So for:

- `tmp/aime25_2_seedfix_seedidx1.jsonl`

using:

- `--seed 1235`

actually means:

- effective seed `1236`

The correct base seed for the earlier HF sample2 reference is:

- `--seed 1234`

which yields effective seed `1235` because `seed_index=1`.

### Live sample2 validation after the sparse exact sampler

After correcting that seed-index mistake, the new sparse exact sampler was
rerun on the exact current-prompt dense sample2 repro.

Artifacts:

- `tmp/aime25_2_densefullkv_sparseexact_seed1234_trace.jsonl`
- `tmp/aime25_2_densefullkv_sparseexact_seed1234_out.jsonl`

Result:

- the first `64` generated native tokens exactly match the previously validated
  dense trace and the saved HF `float16` reference
- the first dense-vs-HF `bfloat16` split remains at generated token `48`
  (`32711` vs `29985`), exactly as before

So the sparse exact sampler preserves the already-proven dense parity behavior
while avoiding the expensive full-vocab scatter plus `at::multinomial` call.
