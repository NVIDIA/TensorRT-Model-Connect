# Pinned offline publisher reference

These test-only modules come from
[`nomic-ai/nomic-bert-2048@7710840340a098cfb869c4f65e87cf2b1b70caca`](https://huggingface.co/nomic-ai/nomic-bert-2048/tree/7710840340a098cfb869c4f65e87cf2b1b70caca).
The embedding checkpoint's `auto_map` points to this separate repository.
Caching the checkpoint alone does not cache its Python code for offline E2E.
Importing these local classes preserves the publisher's reference computation
without requiring network access during validation or changing the builder.

| File | SHA256 of downloaded source |
| --- | --- |
| `configuration_hf_nomic_bert.py` | `f7871694b8de3d3df4ac6640313d5799ce323261a0fb90c5cc567ecc34a0039e` |
| `modeling_hf_nomic_bert.py` | `3b24a366c4cc31b869466ccfb7bbb8879e138c97f8de06c83d4fa1e31a21f149` |

Only trailing whitespace and the final newline are normalized. The publisher's
code and existing notices are otherwise preserved. The source repository declares
Apache-2.0; `LICENSE` includes that license. The modeling module preserves Tri Dao's
copyright and derived attention code; `FLASH_ATTENTION_LICENSE.txt` retains the BSD
notice from [FlashAttention v2.7.4](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4/LICENSE).
The exact-path legal and Ruff exceptions apply only to these two publisher files.

The offline CPU test checks local configuration, tokenizer and weight loading,
finite masked outputs and exact reload consistency on a tiny synthetic checkpoint.
It does not qualify the official checkpoint, TensorRT or GPU execution.
