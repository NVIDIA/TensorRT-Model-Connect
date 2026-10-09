# ModernBERT runtime Tasks

Rebuild existing ModernBERT bundles after this migration. Retired build names
`encoding`, `embedding` and `reranking` are replaced by:

| Build Task | Runtime result |
| --- | --- |
| `text_to_pooled_features` (default) | First-token CLS hidden features, without normalization |
| `text_to_embedding` | Mean of all input-token hidden features, L2 normalized |
| `text_pair_to_relevance` | Existing first-hidden-feature score for `question:QUERY   passage:DOCUMENT` |

A relevance bundle also exposes `text_query_documents_to_relevance`, preserving
document order and returning an empty result for no documents. Execution remains
serial. The score retains the legacy behavior: this family does not load a trained
reranking head or establish calibrated relevance quality.

Each bundle advertises only its selected Task, plus the document-list Task for
relevance. Runtime Config is empty. Default/query/document embedding roles use
the same input processing; no checkpoint-specific prefixes are inferred.
The pooled-feature Task accepts UTF-8 text or token IDs. Inputs must be nonempty
after tokenization, within the built sequence profile and vocabulary. The TP1
dynamic encoder receives exactly the actual input length. Fixed TP plans receive
zero-padded IDs and a fresh validity mask; pooling excludes padding rows.

The existing FP32 TP1 and TP4 E2E cases retain their official CLS reference and
cosine thresholds. They execute the CLI and public C/C++ consumers against the
same bundle. Native CPU contracts cover bindings, masks, pooling, serial relevance,
invalid inputs and engine failures; CPU results do not establish GPU model parity.
