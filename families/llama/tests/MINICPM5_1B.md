# MiniCPM5-1B checkpoint coverage

This adds a pinned E2E selection for `openbmb/MiniCPM5-1B` through the existing
Llama family. It does not add a new architecture or claim GPU qualification.

The configuration fixture is copied from the Apache-2.0 publisher checkpoint:
https://huggingface.co/openbmb/MiniCPM5-1B/blob/87179e5c1f455ef22e6223592d2d61351b525bfc/config.json

Unlike the existing 2B checkpoint, the 1B model has hidden width 1536 but
16 attention heads of width 128 (attention width 2048), with two KV heads and
24 layers. CPU tests preserve that explicit width, full-context KV byte
geometry and the two stop IDs. The E2E uses the existing family build/runtime
and reference comparison with thinking disabled, FP16 candidate, FP32 reference,
a 256-token build limit and ten generated tokens. These limits are a small
parity experiment, not a long-context or performance claim.

Validation remaining: build the pinned checkpoint on authorized GPU hardware
and run `families/llama/tests/test_e2e.py` with `--e2e-model minicpm5-1b`, using
the normal family E2E runtime environment. No large weights were downloaded locally.

The existing Llama tokenizer's Sequence/Split classification issue is tracked
in upstream PR #1423. This checkpoint coverage must not be treated as complete
runtime qualification until tokenizer compatibility and target-GPU parity are
verified. This change does not duplicate that tokenizer implementation.

## Tokenizer-sensitive premerge coverage

The second premerge case uses raw Chinese punctuation, a ten-digit sequence and
an English continuation prompt. Its `expected_prompt_token_ids` come from the
pinned publisher tokenizer, including its `<s>` post-processor token. It uses the
same FP16/FP32 generation comparison and limits as the chat case; no acceptance
threshold is changed. A separate native prefill receipt assertion requires
16 tokens, including BOS, so the observed 18-token baseline cannot pass merely
by generating similar text. This assertion retains the subsequent Hugging Face
generation comparison; it does not turn the case into a KV-contract-only test.
It checks token count, not equality of every native input ID. Selecting `--e2e-model minicpm5-1b` selects both cases.

CPU comparison against Hugging Face Tokenizers 0.22.2 found six mismatches out
of 28 text probes on the initial PR source (f0c70eac). Replacing only the Llama
tokenizer source with PR #1423 at f509329e735f89acb05a24da96e1fa72461a35de made
all 28 probes match. That comparison disabled special-token insertion on both
sides and covered digits, Chinese punctuation, code, whitespace and multilingual
text. The new manifest prompt was also checked separately against the publisher
post-processor. These are tokenizer-only CPU results, not model inference proof.

For example, `1234567890` encodes as `[5645, 12740, 17371, 37]` without special
tokens in the pinned tokenizer. The initial PR runtime instead produced
`[5645, 12740, 1877, 1609]`. The dependency fixes that discrepancy. Until #1423
is integrated, the newly declared case exposes a known tokenizer compatibility
gap; this PR remains a draft and does not copy the dependency's implementation.
