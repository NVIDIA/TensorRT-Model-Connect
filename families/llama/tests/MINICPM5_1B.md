# MiniCPM5-1B checkpoint coverage

This adds a pinned E2E selection for `openbmb/MiniCPM5-1B` through the existing
Llama family. It does not add a new architecture. GPU evidence below covers the three declared cases.

The configuration fixture is copied from the Apache-2.0 publisher checkpoint:
https://huggingface.co/openbmb/MiniCPM5-1B/blob/87179e5c1f455ef22e6223592d2d61351b525bfc/config.json

Unlike the existing 2B checkpoint, the 1B model has hidden width 1536 but
16 attention heads of width 128 (attention width 2048), with two KV heads and
24 layers. CPU tests preserve that explicit width, full-context KV byte
geometry and the two stop IDs. The E2E uses the existing family build/runtime
and reference comparison with thinking disabled, FP16 candidate, FP32 reference,
a 256-token build limit and ten generated tokens. These limits are a small
parity experiment, not a long-context or performance claim.

The checkpoint was built and run on an authorized H200 with the normal family
E2E runtime environment. The exact selection and qualification limits are below.

The Llama tokenizer's Sequence/Split classification fix was merged upstream
as `11ed7177`. This branch includes current main while preserving the original
dependency history. The recorded target-GPU parity covers the declared checkpoint
revision, precision and short prompts only.

## Tokenizer-sensitive premerge coverage

The second premerge case uses raw Chinese punctuation, a ten-digit sequence and
an English continuation prompt. Its `expected_prompt_token_ids` come from the
pinned publisher tokenizer, including its `<s>` post-processor token. It uses the
same FP16/FP32 generation comparison and limits as the chat case; no acceptance
threshold is changed. A separate native prefill receipt assertion requires
16 tokens, including BOS, so the observed 18-token baseline cannot pass merely
by generating similar text. This assertion retains the subsequent Hugging Face
generation comparison; it does not turn the case into a KV-contract-only test.
The current gate checks both the count and every ordered native input ID;
a same-length ID substitution fails. The earlier count-only H200 run remains
identified below. Selecting `--e2e-model minicpm5-1b` selects all three cases.

CPU comparison against Hugging Face Tokenizers 0.22.2 found six mismatches out
of 28 text probes on the initial PR source (f0c70eac). Replacing only the Llama
tokenizer source with PR #1423 at f509329e735f89acb05a24da96e1fa72461a35de made
all 28 probes match. That comparison disabled special-token insertion on both
sides and covered digits, Chinese punctuation, code, whitespace and multilingual
text. The new manifest prompt was also checked separately against the publisher
post-processor. These are tokenizer-only CPU results, not model inference proof.

For example, `1234567890` encodes as `[5645, 12740, 17371, 37]` without special
tokens in the pinned tokenizer. The initial PR runtime instead produced
`[5645, 12740, 1877, 1609]`. The dependency fixes that discrepancy. The branch now includes the original #1423 commit as a history-preserving
dependency merge. The tokenizer implementation retains its original author and
commit; this change does not reimplement it. The scoped target-GPU results are
recorded below.

## Staged checkpoint prerequisite

In offline mode, the Llama E2E lookup consumes a pre-populated Hugging Face
snapshot cache without refreshing Hub metadata. Community GPU CI stages that cache
before entering its offline test container. For a manual run, populate the cache
with the pinned checkpoint first and set `HF_HUB_OFFLINE=1`; an absent revision or missing `config.json`
still fails rather than selecting another checkpoint.

This avoids a Hub 1.32 repository-tree lookup that otherwise raises
`OfflineModeIsEnabled` even when the pinned snapshot is already cached. The
regression tests use temporary local snapshots without any tree index or model
weights. No manifest revisions, comparisons or acceptance thresholds are changed.


## Empty raw prompt framing

The pinned publisher tokenizer returns BOS `[0]` for empty raw text with special
tokens enabled. The native tokenizer previously returned before applying its
post-processor, producing no input IDs. The fix retains post-processing for
empty text. CPU tests cover BOS-only, EOS-only, combined framing, disabled
special tokens, nonempty controls and repeated calls.

The third premerge case uses an empty raw prompt, requires the reference's
single BOS token and a one-token native prefill receipt, then retains the same
FP16/FP32 generated-output comparison. No parity threshold is changed. A CPU
comparison against the pinned Tokenizers assets covers 28 inputs with and without
special tokens. The H200 run below also checks generation for all three cases.

## Single-token prefill mask contract

A run of the three cases on one NVIDIA H200 at `fb2d3312` passed chat and
multilingual/digit generation but failed the empty case before prefill. The
single BOS token selected the one-row mask path, which required the static
decoder mask shape even when bound to a fixed-cache prefill engine. That
engine's dynamic mask buffer initially has the profile OPT shape.

The runtime now accepts a dynamic mask whose current profile minimum is
`[1, cache_capacity + 1]`. Static decoder validation and mask construction
remain in place. The native KV regression reproduces the error with concrete
OPT buffer dimensions, checks both BOS-only and one-token-tail masks, and
rejects four malformed minimum profiles. It failed before the correction and
passed on H200 afterward. All three full checkpoint cases were then rerun at
`3e149b2bb4e5c1ee3e6b324591a65dfecf7ebd83`.

## Recorded H200 validation

On 2026-11-19, the following unchanged family selection passed at `3e149b2b`:

```bash
python -m pytest families/llama/tests/test_e2e.py \
  --e2e-testcase minicpm5-1b,minicpm5-1b-multilingual-digits,minicpm5-1b-empty \
  -vv -s -p no:cacheprovider --tb=short
```

| Case | Native prefill tokens | Generated token IDs versus FP32 reference |
| --- | ---: | --- |
| Chat | 25 | 2/2 identical, including EOS; text `Paris` |
| Multilingual/digits | 16 | 10/10 identical |
| Empty raw text | 1 | 10/10 identical |

Result: 3 passed, 11 unselected cases skipped. Candidate FP16, reference FP32,
TP1, 256-token bundle capacity, at most ten generated tokens; checkpoint and
tokenizer revision `87179e5c1f455ef22e6223592d2d61351b525bfc`. Comparison
thresholds were unchanged. The receipt proves native input token counts,
not native input token-ID equality.

Environment: one NVIDIA H200, x86_64 Ubuntu 24.04, driver 595.71.05,
CUDA 13.4.3 development image, TensorRT 11.4.0.106, PyTorch 2.12.0+cu130,
Transformers 5.2.0 and Tokenizers 0.22.2. The repository Dockerfile was used
with x86_64 TensorRT include paths and Hopper architecture settings. Production
CLI, TensorRT backend and Llama DSO compiled; four native CTests passed,
including the new KV regression. The focused Python group passed 17 tests;
source-quality passed all 312 tool tests at the same source head.

This does not qualify other checkpoints, precisions, tensor-parallel sizes,
long contexts, sampling, concurrency or performance. Rebuild the Llama DSO;
no engine format change is required. The documentation-only evidence commit
leaves the tested runtime and test source blobs unchanged.
## Native prompt-ID assertions

For cases declaring `expected_prompt_token_ids`, E2E now enables
`TRTMC_LLAMA_TRACE_PREFILL_IDS=1` only in that native subprocess. After successful
prefill, the Llama pipeline emits its actual input vector in one
`[trtmc.prefill_ids]` receipt. The checker compares the ordered IDs exactly,
including BOS, in addition to the existing count receipt and unchanged
FP16/FP32 generated-output comparison. This guards against same-length token
substitutions that a count assertion alone cannot detect.

Tracing is off by default. Cases without expected IDs explicitly remove an
inherited flag, so normal generation and the chat control do not emit input IDs.
Ten added CPU cases cover exact BOS/multitoken receipts, same-length wrong IDs,
missing/duplicate/malformed receipts, length mismatches and subprocess opt-in.
The focused Python group now passes 27 tests. The initial H200 input evidence
above was count-only. The stricter gate was rerun at
`76275b9848a4a6a8a9cb868322cdea77f5ba9bb8` on the same H200 environment:
3 passed, 11 unselected skipped. Both raw cases matched all expected native
prompt IDs (16 multilingual/digit IDs and BOS `[0]`), the chat control emitted
no ID trace, and all 22 generated IDs still matched the FP32 reference.
Native trace controls `0`, `invalid` and `1` passed; the old implementation
failed the requested trace probe after real prefill. Four native CTests,
27 focused Python tests and all 312 source-quality tool tests passed at this
source head. These statements cover the exact test inputs, not arbitrary
prompts or runtime performance.

The copied configuration's publisher, revision and license are recorded in
`families/llama/NOTICE`. The pinned checkpoint also has a documentation-only
entry in `website/data/hf-model-metadata.json`, as required by the contributor
guide; that inventory does not control support ownership or qualification.

## Recorded integration recheck at b1db0296

After merging NVIDIA `main` at `b1db0296`, the same strict three-case E2E was
rerun at `c76e393b544c9fbb560034fd0630d7bdf714a610`: 3 passed, 11 unselected
skipped. Both raw prompt-ID receipts and all 22 generated IDs matched, and the
chat control emitted no ID trace. Four native CTests and 27 focused Python
tests passed. Source-quality passed 322 tool tests, and the documentation
inventory's 15 tests and full website build passed on Linux with Node 22.23.2.
The metadata conflict resolution retained every upstream checkpoint plus this
pinned 1B entry. The final evidence-only documentation update changes no tested
runtime, builder, test or snapshot record.

## Conflict-resolution integration at 65281216

The later merge at `a9c5e1e0` retains all 152 canonical metadata records plus
this checkpoint, and preserves upstream packed-KV initialization and dtype
validation alongside the prefill-mask guard and opt-in ID trace.
The H200 results above remain evidence for `c76e393b`; this merge does not add
a GPU result. Current-head target-hardware validation remains outstanding.

## Consumer and profile scope after integration

The empty-text behavior is family-wide, not restricted to this checkpoint.
With special tokens enabled, the old `encode("")` returned `[]` before any
post-processing; it now returns the configured prefix/suffix framing, such as
`[BOS]` or `[BOS, EOS]`. The pinned MiniCPM tokenizer returns `[0]`.
With special tokens disabled or no configured framing, the result remains `[]`.
Nonempty effective text retains its segmentation and framing path.

The checked-in runtime has two tokenizer encode call sites. The following are
source-level consequences, not new generation-equivalence measurements:

| Consumer | Empty effective text before / after framing correction |
| --- | --- |
| Normal `generate`, through `encode_prompt` in `runtime/pipeline.cpp` | Empty IDs previously returned without generation. Nonempty framing now enters prefill/generation; no-framing empty text still returns without generation. |
| `speculative::Pipeline::generate`, including auto/EAGLE3/autoregressive modes | Empty IDs previously failed `generate_ids` validation. Nonempty framing can now enter target and, when enabled, draft execution. This behavior has not received checkpoint validation here. |
| CLI, SDK/server callers using either string-generation implementation | They inherit the corresponding behavior above; callers are not assumed unable to submit raw empty text. |
| IDs-only generation and internal single-forward | They bypass tokenizer encode. Normal empty IDs still return without generation; speculative empty IDs still reject. Single-forward string generation rejects before encode and empty IDs reject. |
| E2E | Raw-prompt cases use normal string generation. The historical empty MiniCPM case tests BOS framing; it does not validate speculative empty text or every other postprocessor. |

If a chat template produces nonempty effective text, it does not exercise the
empty-text branch. This audit does not claim every template or external caller
is equivalent. Edge-LLM's separate SDK tokenizer/runtime is outside this audit.

For the mask guard, let `C` be the fixed cache capacity. The eligible builder
outputs and existing alternative routes are:

| Bundle/profile class | Mask profile and effect of the added guard branch |
| --- | --- |
| Fixed-cache standard dynamic-sequence prefill, including the prefill profile of a dual-profile engine and split prefill engine | `dual_profile_decoder_builder.py` defines dynamic mask `[-1,-1]`, MIN `[1,C+1]`, OPT `[Sopt,C+Sopt]`, MAX `[Smax,C+Smax]`. The added branch permits a one-token prefill or tail even when the current OPT/buffer shape is not `[1,C+1]`. |
| Static standard decode, or fixed one-token decode profile | Its current mask already has `[1,C+1]`; the original current-shape check and decoder binding validation remain. |
| Runtime-sized KV | Its mask MIN is `[1,2]` and existing dynamic binding bypasses the fixed-cache comparison; this is not the newly added branch. |
| Native/active-prefix KV, cache-free single-forward and speculative engines | They use different mask contracts or bypass `write_decode_mask`; this guard does not qualify those routes. |

This is a profile-based rule, not a checkpoint allowlist. Qualifying fixed-cache
bundles for non-MiniCPM checkpoints can also reach it. Every non-MiniCPM5-1B
manifest selection, other precision/profile, speculative path and other
postprocessor remains outside this PR's recorded checkpoint qualification.
No claim is made that every existing serialized bundle was inspected or run.

The native KV fixture uses `C=11`, current `[64,75]` / tail `[4,15]`, and valid
MIN `[1,12]`. Its negative controls reject MIN `[1,11]`, `[2,13]`, `[-1,12]` and
`[1,12,1]`; the old static decode control remains. More generally, fixed-cache
inputs with an incorrect current shape must still reject when the mask is
static or its dynamic MIN differs from the exact rank-two `[1,C+1]` predicate.
These controls are not proof of all possible bundle contracts. Their current
CUDA/native-KV execution remains outstanding; the prior H200 result is at c76.

At `ea51a82e`, an isolated CPU harness compiled the actual repository BPE/chat
sources and the existing tokenizer-special-token, pretokenizer-variant and
chat-template tests: all three passed. This is not a production DSO build or
CUDA KV CTest. Compiling the same special-token fixture with canonical
`65281216` BPE source produced six expected empty-framing assertion failures;
the branch factory passed. Local production CMake configuration stopped at
the missing `nlohmann_json` package configuration, so DSO linking remains
unrun. CUDA KV tests and checkpoint GPU rechecks still require available
authorized target hardware; Radix authentication is expired.
