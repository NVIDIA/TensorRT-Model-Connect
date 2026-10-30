# Native tokenizer normalization

This family is implemented by the existing semantic `text_to_embedding` Task.
Its graph, runtime role prefixes, truncation and pooling remain defined by the
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
`UNICODE_LICENSE.txt`; no extra runtime dependency is introduced.

Native installation also carries the notice under `share/trtmc/families/nomic_bert`.

The existing native tokenizer test retains prefix/framing assertions and adds
distinct-vocabulary normalization cases, independent normalizer flags and
punctuation boundaries. Matching unknown tokens cannot stand in for the expected
IDs in those cases. A source-only probe of the production normalizer matches all
1,112,064 Unicode scalars in three lowercase/accent modes and 3,630 constructed
mark sequences per mode against Tokenizers 0.22.2, with cleaning and Chinese
spacing disabled to isolate those stages. This does not cover every
Unicode string, model inference parity or performance. Canonical ordering needs
a temporary codepoint buffer and stable sorting within each combining sequence.

Rebuild the family runtime to obtain corrected token IDs. No bundle schema or
public Task API change is required.
