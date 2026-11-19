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
It checks token count, not equality of every native input ID. Selecting `--e2e-model minicpm5-1b` selects all three cases.

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
