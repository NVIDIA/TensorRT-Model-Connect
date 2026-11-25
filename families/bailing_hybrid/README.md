# bailing_hybrid

The native builder uses the project's shared Transformers version. Upstream
parity runs in a separate interpreter so reference dependencies cannot replace
the native build environment's packages.

Prepare a reference environment before offline validation, including a
CUDA-enabled PyTorch installation for the host:

```bash
python3 -m venv /tmp/trtmc-bailing_hybrid-reference
/tmp/trtmc-bailing_hybrid-reference/bin/python -m pip install -r families/bailing_hybrid/tests/requirements.txt
export TRTMC_BAILING_HYBRID_REFERENCE_PYTHON=/tmp/trtmc-bailing_hybrid-reference/bin/python
```

Run the family E2E with the pinned checkpoint and built native runtime configured.
The reference entry point requires Transformers 4.57.6; it does not install
packages or access the network during validation. Its token/embedding comparison
criteria are unchanged. An explicit local checkpoint override is reported as
unpinned; pinned runs resolve the exact Hub snapshot in the recipe.

The native family library requires the host ICU shared libraries (`uc`, `i18n`,
and their `data` dependency), including for wheel installations. Install the
runtime package matching the wheel's ICU ABI; an Ubuntu 24.04 build uses
`libicu74`. Building from source additionally requires `libicu-dev`.
See the [host-library check](../../website/docs/getting-started/quick-start.md#host-runtime-libraries)
before loading the family.
