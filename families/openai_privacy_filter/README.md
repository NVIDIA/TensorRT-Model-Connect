# openai_privacy_filter

Token classifiers of the Transformers `openai_privacy_filter` architecture
(`OpenAIPrivacyFilterForTokenClassification`), the PII detectors derived from
`openai/privacy-filter`, for example
[`OpenMed/privacy-filter-nemotron-v2`](https://huggingface.co/OpenMed/privacy-filter-nemotron-v2)
(221 BIOES labels over 55 PII categories).

## Task

`text_to_head_scores`: one request is one text (or one list of token ids). The
family tokenizes it without framing tokens, runs it through the engine once and
returns the raw label logits of every token as a `[tokens, labels]` matrix
(`score_kind` `logit`, `pooling` and `normalization` `none`). Column `j` is the
checkpoint's `id2label[j]`; the bundle's `runtime.json` lists the label names in
that order. Span decoding (BIOES, Viterbi, merging) is left to the caller; the
Task carries no token offsets.

```bash
trtmc build OpenMed/privacy-filter-nemotron-v2 --output privacy-filter.bundle
trtmc encode privacy-filter.bundle --text "Call Sarah at 415-555-0123."
```

## Graph

Float32 throughout (TF32 disabled), constants upcast exactly from the stored
BF16:

- token embedding gather, then per layer: RMSNorm, Q/K/V with bias, rotary
  embedding on interleaved pairs from YaRN tables, Q and K each scaled by
  `head_dim ** -0.25`, grouped-query attention over the bidirectional band
  `|i - j| <= sliding_window` with one sink logit per head appended to the
  softmax, output projection;
- RMSNorm, router with bias, top-k over the raw router logits, softmax over the
  selected logits, clamped SwiGLU experts (`(up + 1) * gate * sigmoid(alpha *
  gate)`), weighted sum of the selected expert outputs;
- final RMSNorm and the `score` head.

Every expert is evaluated for every token (one GEMM over all experts, then a
gather of the selected outputs), which is exact but costs
`num_local_experts / num_experts_per_tok` times the routed work; attention
materializes the `[heads, tokens, tokens]` scores. Both bound the practical
`max_sequence_length` (default 4096, at most `max_position_embeddings`).
Longer inputs are refused rather than truncated or windowed.

## Tokenizer

The runtime implements the checkpoint's `tokenizer.json` natively: added tokens
first (leftmost-longest), then the o200k split rule, byte-level symbols and BPE
with whole-piece lookup (`ignore_merges`). It refuses any other tokenizer
configuration. The split rule's character classes come from the Unicode 16.0
[UnicodeData.txt](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt)
and [PropList.txt](https://www.unicode.org/Public/16.0.0/ucd/PropList.txt)
(White_Space), redistributed under Unicode License v3 in `UNICODE_LICENSE.txt`,
which is installed with this family. Regenerate the table with:

```bash
python families/openai_privacy_filter/tools/generate_unicode_classes.py \
  UnicodeData.txt PropList.txt families/openai_privacy_filter/runtime/unicode_classes.h
```

## Tests

- `tests/test_support.py`, `tests/test_unicode_classes.py`: CPU, no checkpoint.
- `tests/cpp/test_openai_privacy_filter_tokenizer.cpp`,
  `tests/cpp/test_openai_privacy_filter_task_contract.cpp`: CPU (`ctest`).
- `tests/test_e2e.py`: builds the manifest's bundle and compares the native
  tokenizer, CLI, `--token-ids` path and C/C++ SDK consumers with the
  Transformers float32 reference on the GPU. It needs `requirements.txt`.

Repository source retains its Apache-2.0 notices. Checkpoints are distributed by
their publishers under their own licenses and are not included.
