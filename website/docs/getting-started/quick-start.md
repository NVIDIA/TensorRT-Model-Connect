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

For a wheel install, resolve its native runtime directory directly from the
installed package:

```bash
TRTMC_RUNTIME_ROOT="$(python -c 'import pathlib, tensorrt_model_connect as m; print(pathlib.Path(m.__file__).parent / "bin")')"
trtmc run gpt2.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "Hello" \
  --max-new-tokens 32
```

For a native CMake install, point the loader at the directory containing the matching
`libtrtmc_core.so`, `libtrtmc_runtime.so`, `libtrtmc_backend_trt.so`, and
`libtrtmc_model_gpt2.so`. The loader reads the bundle header, loads exactly
those DSOs, and returns the abstract task interface declared by the bundle.

```bash
trtmc run gpt2.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello" \
  --max-new-tokens 32
```

The shell variable above is only a convenient explicit argument. The CLI never
searches environment variables, the current directory, or an installed
fallback runtime.
