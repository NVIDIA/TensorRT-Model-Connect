# Native tokenizer normalization

This family is implemented by the existing semantic `text_to_embedding` Task.
Its graph, truncation and CLS pooling remain defined by the
family builder and pipeline. This change only corrects native WordPiece
normalization and character classification.

The family-local `runtime/unicode_data.h` reproduces the independently versioned
components of the fixed Tokenizers 0.22.2 oracle:

| Data | UnicodeData version | SHA256 |
| --- | --- | --- |
| Nonspacing-mark categories | [8.0.0](https://www.unicode.org/Public/8.0.0/ucd/UnicodeData.txt) | `38b17e1118206489a7e0ab5d29d7932212d38838df7d3ec025ecb58e8798ec20` |
| Canonical decomposition and combining classes | [9.0.0](https://www.unicode.org/Public/9.0.0/ucd/UnicodeData.txt) | `68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f` |
| Simple lowercase mappings | [17.0.0](https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt) | `2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c` |

The reference's `unicode-normalization-alignments` 0.1.12 crate declares Unicode
9.0.0. Canonical decompositions are identical in the cited versions 8 and 9.
Hangul decomposition is algorithmic; U+0130 lowercases to `i` plus U+0307. Stable
combining-class ordering precedes nonspacing-mark removal and lowercase, while
`strip_accents: false` preserves the original order. The Unicode notice is in
`UNICODE_LICENSE.txt`, also installed with native builds under
`share/trtmc/families/gte`; no extra runtime dependency is introduced.

The native tokenizer test retains the family framing checks and adds 60 fixed
oracle cases with distinct vocabulary IDs across three lowercase/accent modes.
The fixture in `tests/fixtures/unicode_tokenizer.json` records its source revision,
Tokenizers version and tokenizer SHA256. It preserves the official normalizer and
post-processor with a synthetic vocabulary that distinguishes normalization
outcomes. No model weights or network access are required by the native test.

A separate source-only probe uses the full tokenizer from
[`Alibaba-NLP/gte-base-en-v1.5@a829fd0e060bb84554da0dfd354d0de0f7712b7f`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5/tree/a829fd0e060bb84554da0dfd354d0de0f7712b7f).
Padding and truncation are disabled to compare the native tokenizer's unpadded
output before the existing pipeline applies its window. Tokenizer agreement does
not establish TensorRT embedding parity, latency or memory equivalence. Canonical
ordering uses a temporary codepoint buffer and stable sorting within each
combining sequence.

Rebuild the family runtime to obtain corrected token IDs. No bundle schema or
public Task API change is required.
