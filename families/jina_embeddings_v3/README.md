# jina_embeddings_v3

The native builder uses the project's shared Transformers version. Upstream
parity runs in a separate interpreter so reference dependencies cannot replace
the native build environment's packages.

The family manifest declares `reference_environment: isolated`. With the native
wheel available in `dist/`, prepare the runtime and reference environment before
offline validation:

```bash
MODEL=jina_embeddings_v3 python3 -m tools.ci pipeline setup
```

Setup installs `tests/requirements.txt` in `.ci/reference/jina_embeddings_v3`
(under `TRTMC_CI_STATE_DIR` when configured). The E2E selects that environment
through the repository's prepared-interpreter resolver, which checks its setup
receipt against the current requirements. No manual reference-interpreter
override is needed. Reference subprocesses do not inherit `PYTHONPATH` or
`PYTHONHOME` from the builder environment.

Run the family E2E with the pinned checkpoint and built native runtime configured.
The reference entry point requires Transformers 5.20.0; setup requires package
access, while offline validation does not install packages or access the network.
Its token/embedding comparison criteria are unchanged. An explicit local
checkpoint override is reported as unpinned; pinned runs resolve the exact Hub
snapshot in the recipe.
