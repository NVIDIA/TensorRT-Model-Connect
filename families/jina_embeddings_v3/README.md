# jina_embeddings_v3

The native builder uses the project's shared Transformers version. Upstream
parity runs in a separate interpreter so reference dependencies cannot replace
the native build environment's packages.

Prepare a reference environment before offline validation, including a
CUDA-enabled PyTorch installation for the host:

```bash
python3 -m venv /tmp/trtmc-jina_embeddings_v3-reference
/tmp/trtmc-jina_embeddings_v3-reference/bin/python -m pip install -r families/jina_embeddings_v3/tests/requirements.txt
export TRTMC_JINA_EMBEDDINGS_V3_REFERENCE_PYTHON=/tmp/trtmc-jina_embeddings_v3-reference/bin/python
```

Run the family E2E with the pinned checkpoint and built native runtime configured.
The reference entry point requires Transformers 5.20.0; it does not install
packages or access the network during validation. Its token/embedding comparison
criteria are unchanged. An explicit local checkpoint override is reported as
unpinned; pinned runs resolve the exact Hub snapshot in the recipe.
