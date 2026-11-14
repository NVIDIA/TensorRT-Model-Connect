# ministral3 — bidirectional Ministral 3 sentence encoders

This family serves `Ministral3Model` checkpoints that are trained as
bidirectional text-embedding encoders and published with a Sentence
Transformers head of mean pooling and L2 normalization, such as
[`nvidia/Nemotron-3-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16).
A causal backbone (`is_causal` absent or true), sliding-window attention, an
unknown RoPE type or another Sentence Transformers head is rejected at build
time.

## Tasks

One bundle holds one encoder engine and serves three Tasks; the build task
selects the primary one (default `text_to_embedding`).

| Task | Result |
| --- | --- |
| `text_to_embedding` (default) | the checkpoint's sentence embedding: the role prompt from `config_sentence_transformers.json` (`query: ` for queries, `passage: ` for documents, none for the default role) is prepended, every token's final hidden state is averaged, the mean is L2-normalized |
| `text_to_pooled_features` | the mean of the final hidden states of the text, without prompt or normalization |
| `text_to_token_features` | the final hidden state of every token with its token ID and UTF-8 byte offsets |

`text_to_pooled_features` and `text_to_token_features` also accept token IDs.
Runtime Config is empty. A request must have at least one token and at most the
bundle's `max_sequence_length`; longer inputs are rejected, never truncated.

## Build options

- Precision `bf16` (default, the checkpoint's stored precision) or `fp32`.
- `max_sequence_length` from 1 to the checkpoint's Sentence Transformers
  `max_seq_length` (32768 for Nemotron-3-Embed); default 8192. Positions at and
  beyond `original_max_position_embeddings` (16384) use the Llama-4 query scale
  `1 + beta * log(1 + floor(position / 16384))` of the reference.
- One sequence per engine call: `max_batch_size`, tensor or context
  parallelism, quantization, KV-cache and image/video options are rejected.

## Graph

Every token attends to every token. Rounding points follow the Transformers
module at the engine precision: RMSNorm normalizes in FP32, rounds to the
working type, then multiplies by the stored weight; YaRN inverse frequencies are
formed in FP32 in the reference's operation order and the rotary tables are
rounded to the working type; the query scale (Llama-4 factor times
`1/sqrt(head_dim)`) and SiLU are evaluated in FP32 and rounded once. Pooling and
normalization run on the host in FP64 from the FP32 hidden states.

## Tokenizer

`runtime/tokenizer.cpp` implements exactly the `tokenizer.json` contract of the
checkpoint and refuses any other: added tokens are split first
(leftmost-longest); a `Split` with the Tekken regex (`Isolated`) and
`ByteLevel` without regex pre-tokenize; BPE with `ignore_merges` and without
unknown token, byte fallback or dropout; a template that adds no tokens. The
regex is evaluated by a hand-written matcher of its seven alternatives with
Oniguruma's leftmost-first, greedy backtracking order.

Exactness was established against Tokenizers 0.22.2 (within the
`>=0.22.0,<=0.23.0` range that Transformers 5.2.0 accepts). Its `\p{..}` classes
follow Unicode 16.0.0 and its `\s` is the Unicode `White_Space` property: among
Unicode 14.0, 15.0, 15.1, 16.0 and 17.0 tables with either White_Space or ASCII
spaces, only Unicode 16.0 with White_Space reproduced the oracle's splits on all
of about 238,000 probes. `runtime/unicode_data.h` is generated from:

| Data | Source | SHA256 |
| --- | --- | --- |
| General categories L*, M*, N* | [UnicodeData.txt 16.0.0](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt) | `ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f` |
| White_Space | [PropList.txt 16.0.0](https://www.unicode.org/Public/16.0.0/ucd/PropList.txt) | `53d614508e2a0b2305a8aa21cd60d993de9326cdf65993660dfcce4503548583` |

The Unicode notice is in `UNICODE_LICENSE.txt`, installed with native builds
under `share/trtmc/families/ministral3`. Third-party provenance (the Tekken
pattern from Mistral AI's Apache-2.0 tokenizer, the reimplemented Transformers
and Tokenizers algorithms, the checkpoint's OpenMDW-1.1 terms) is in `NOTICE`.

The native tokenizer test checks 634 Split-regex cases recorded from the oracle
on the checkpoint's own pre-tokenizer
(`tests/fixtures/tekken_pre_tokenizer.json`, with the tokenizer revision and
SHA256) and the token IDs of a small synthetic `tokenizer.json` with the same
contract (`tests/fixtures/tekken_tokenizer.json`), and that contract
violations and invalid UTF-8 are refused.

## Validation

`tests/test_e2e.py` builds the bundle and runs the CLI `embed` Task for every
query and document of a case (with their role prompts), the token-feature Task
for every query and document without prompt, the public C and C++ consumers on
the first query and the pooled-feature Task on the first document. The oracle is
the checkpoint's documented Transformers recipe in FP32 with TF32 disabled; the
case texts are written for these tests. Every case requires token IDs identical
to `AutoTokenizer`, byte offsets that tile each input and the FP32 truth's
retrieval ranking.

- FP32 bundles must meet absolute thresholds: embedding cosine at least
  0.999999, query-document score difference at most 1e-5, token-feature cosine at
  least 0.99999 and pooled-feature cosine at least 0.999999.
- BF16 bundles are judged only relative to the official BF16 Transformers run's
  drift from the same FP32 truth (1 - cosine, or the absolute score difference),
  on statistics pooled over the case (at least 100 tokens of token features):
  means at most 1.5x and maxima at most 3x for embeddings, scores and token
  features; the pooled feature and every repeated long document at most 3x.

The long-context cases reach positions beyond 16384, where the Llama-4 query
scale acts: a bundle built without that scale drifts 5x the official BF16 drift
(BF16) and 70x the FP32 bound on the long document, and both cases reject it.
