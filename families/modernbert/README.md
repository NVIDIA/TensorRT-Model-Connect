# ModernBERT runtime Tasks

Rebuild existing ModernBERT bundles after this migration. Retired build names
`encoding` and `embedding` are replaced by:

| Build Task | Runtime result |
| --- | --- |
| `text_to_pooled_features` (default) | First-token CLS hidden features, without normalization |
| `text_to_embedding` | Mean of all input-token hidden features, L2 normalized |

Each bundle advertises only its selected Task. Legacy `reranking`,
`text_pair_to_relevance` and `text_query_documents_to_relevance` bundles are
rejected before engine loading. The encoder returns hidden states and has no
trained scoring head. Its first hidden feature is not a relevance score; future
reranking support needs a scoring head and checkpoint-specific pair tokenization.

Runtime Config is empty. Default/query/document embedding roles use
the same input processing; no checkpoint-specific prefixes are inferred.
The pooled-feature Task accepts UTF-8 text or token IDs. Inputs must be nonempty
after tokenization, within the built sequence profile and vocabulary. The TP1
dynamic encoder receives exactly the actual input length. Fixed TP plans receive
zero-padded IDs and a fresh validity mask; pooling excludes padding rows.

The existing FP32 TP1 and TP4 E2E cases retain their official CLS reference and
cosine thresholds. They execute the CLI and public C/C++ consumers against the
same bundle. Native CPU contracts cover bindings, masks, pooling, unsupported relevance rejection,
invalid inputs and engine failures; CPU results do not establish GPU model parity.
Set `TRTMC_NATIVE_BUILD_DIR` to the CMake build root containing both SDK consumers,
including when `TRTMC_BINARY` points to a CLI installed from a wheel.
