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

Run the bundle:

```bash
trtmc run gpt2.bundle \
  --prompt "Hello" \
  --max-new-tokens 32
```

The loader reads the bundle header and loads exactly `libtrtmc_backend_trt.so`
and `libtrtmc_model_gpt2.so` from the directory containing the active
`libtrtmc_runtime.so`: the package `bin` directory for a wheel install, or the
library directory for a native CMake install. It returns the abstract task
interface declared by the bundle.

To load from another directory, pass `--runtime-root DIR` naming a directory
that contains the matching `libtrtmc_core.so`, `libtrtmc_runtime.so`,
`libtrtmc_backend_trt.so`, and `libtrtmc_model_gpt2.so`. The CLI never searches
environment variables, the current directory, or a fallback runtime.
