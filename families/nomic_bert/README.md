# Nomic text embeddings

The `nomic_bert` family builds the postnorm, full-rotary, SwiGLU encoder used by
`nomic-ai/nomic-embed-text-v1.5`. Validation pins revision
`e9b6763023c676ca8431644204f50c2b100d9aab`.

```bash
python -m tensorrt_model_connect nomic_bert build /path/to/snapshot \
  --output nomic.bundle --max-sequence-length 512
trtmc embed nomic.bundle --runtime-root build \
  --task text_to_embedding --text "Where is Paris?" --role query
```

The native C/C++ Task is `TextToEmbedding`. Query and Document roles prepend
`search_query: ` and `search_document: ` respectively. Default embeds the text
as supplied, allowing explicit clustering/classification prefixes. Special
tokens participate in mean pooling, padding does not, and the 768-dimensional
result is L2-normalized. The embedding-space identifier is empty because a local
snapshot does not establish the original checkpoint identity.

Only FP32 on the generic TensorRT backend is supported. The fixed engine profile
accepts 2 to 2048 tokens, including special tokens and role prefixes. Overlong
input fails instead of truncating. Text is limited to 1024 bytes per profile token
before tokenization. Extended-context rotary scaling, reduced Matryoshka
dimensions, batches, quantization and tensor parallelism are not implemented.

The builder requires `config.json`, FP32 `model.safetensors` and `tokenizer.json`.
No custom publisher code executes during the build or native inference. The
family owns its graph, checkpoint mapping, WordPiece tokenizer and native DSO;
TensorRT lowers and executes the graph.

The native tokenizer keeps its Unicode normalization data family-local, with no
new runtime dependency. `runtime/unicode_data.h` derives its nonspacing-mark
ranges, canonical decompositions and simple lowercase mappings from fields 2,
5 and 13 of [UnicodeData 8.0.0](https://www.unicode.org/Public/8.0.0/ucd/UnicodeData.txt)
(SHA256 `38b17e1118206489a7e0ab5d29d7932212d38838df7d3ec025ecb58e8798ec20`).
The category version matches the pinned Tokenizers 0.22.2 oracle; newer mark
categories must not be silently substituted. Hangul decomposition is algorithmic.
The Unicode data notice is in `UNICODE_LICENSE.txt`. This is not a claim of
exhaustive compatibility with later Unicode versions or every normalization sequence.
