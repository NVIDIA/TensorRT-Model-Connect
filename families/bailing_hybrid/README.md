# bailing_hybrid

The native builder uses the project's shared Transformers version. Upstream
parity runs in a separate interpreter so reference dependencies cannot replace
the native build environment's packages.

The family manifest declares `reference_environment: isolated`. The standard
setup stage provisions `tests/requirements.txt` before offline validation:

```bash
MODEL=bailing_hybrid python -m tools.ci pipeline setup
```

With `TRTMC_BAILING_HYBRID_REFERENCE_PYTHON` unset, the E2E uses the prepared
interpreter under `${TRTMC_CI_STATE_DIR:-.ci}/reference/bailing_hybrid` and
validates its readiness receipt against the current reference requirements.
The builder keeps its own Transformers version. Missing or stale reference
environments fail before reference inference; E2E does not install packages.

For a manually provisioned reference, include a CUDA-enabled PyTorch
installation for the host and select that environment explicitly:

```bash
python3 -m venv /tmp/trtmc-bailing_hybrid-reference
/tmp/trtmc-bailing_hybrid-reference/bin/python -m pip install -r families/bailing_hybrid/tests/requirements.txt
export TRTMC_BAILING_HYBRID_REFERENCE_PYTHON=/tmp/trtmc-bailing_hybrid-reference/bin/python
```

Run the family E2E with the pinned checkpoint and built native runtime configured.
An explicitly selected interpreter must exist and be executable; invalid or
empty overrides fail instead of falling back to the builder's interpreter.
The reference entry point requires Transformers 4.57.6; it does not install
packages or access the network during validation. Its token/embedding comparison
criteria are unchanged. An explicit local checkpoint override is reported as
unpinned; pinned runs resolve the exact Hub snapshot in the recipe.

The native family library requires the host ICU shared libraries (`uc`, `i18n`,
and their `data` dependency), including for wheel installations. Install the
runtime package matching the wheel's ICU ABI; an Ubuntu 24.04 build uses
`libicu74`. Building from source additionally requires `libicu-dev`.

Check this family's shared library before loading a bundle. Replace the path
below with the library in the installed wheel's `bin` directory or your native
build directory:

```bash
python - /path/to/runtime/libtrtmc_model_bailing_hybrid.so <<'PYTHON'
import ctypes
import os
import sys

ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW | os.RTLD_LOCAL)
PYTHON
```

A missing or incompatible dependency raises `OSError` with the dynamic linker's
diagnostic. Install the matching host library before retrying the Bailing runtime.
