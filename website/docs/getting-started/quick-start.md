---
title: Quick Start
---

Install the wheel produced by the release or local package stage:

```bash
python -m pip install /path/to/tensorrt_model_connect-0.1.0-*.whl
```

If the selected family owns extra build or reference dependencies, install its
plain requirements file. From a checkout or unpacked source release:

```bash
python -m pip install -r families/sana_wm/requirements.txt
```

The wheel carries the same owner file. After installing the wheel, locate it
from the installed `families` package:

```bash
FAMILY_REQUIREMENTS="$(python -c 'from pathlib import Path; import families; print(Path(families.__file__).parent / "sana_wm" / "requirements.txt")')"
python -m pip install -r "$FAMILY_REQUIREMENTS"
```

There is no central family extra or dependency registry. A family without a
`requirements.txt` needs only the pinned base environment and the wheel.

Build a bundle directly from a Hugging Face model ID:

```bash
python -m tensorrt_model_connect build openai-community/gpt2 \
  --precision fp16 \
  --output gpt2.bundle
```

The CLI downloads the snapshot, reads `config.json` or `model_index.json`, and
asks every dependency-free family `support.py`. Exactly one family must claim
the checkpoint. That family supplies the default task; pass `--task` only when
selecting another task supported by the same family. The build then imports
only the selected `families.gpt2.model` and calls `build(request, writer)` once.
A prepared local snapshot can be passed in place of the model ID.

Run the bundle directly:

```bash
trtmc run gpt2.bundle \
  --prompt "Hello" \
  --max-new-tokens 32
```

The CLI reads the bundle family and backend, then selects the first complete,
single-directory runtime in this order:

1. the current directory, when all required libraries match the active build
   cohort;
2. the runtime belonging to the active `trtmc` selected through `PATH`,
   including native CMake and wheel install layouts;
3. colon-separated directories in `TRTMC_RUNTIME_PATH`.

A complete GPT-2 TensorRT runtime contains matching `libtrtmc_core.so`,
`libtrtmc_runtime.so`, `libtrtmc_backend_trt.so`, and
`libtrtmc_model_gpt2.so` files. Candidates are never combined across
directories, and the CLI prints the automatically selected directory. If more
than one installed wheel runtime matches, select one with `--runtime-root DIR`.
An explicit root bypasses discovery.

Every native artifact carries a build-cohort identity, and automatic discovery
accepts a directory only when the identity matches the core and runtime already
loaded by `trtmc`. The platform loader evaluates `LD_LIBRARY_PATH` before the
CLI starts, so it can determine that active cohort before the search above.
