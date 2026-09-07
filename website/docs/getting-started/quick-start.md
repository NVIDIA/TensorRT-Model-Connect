---
title: Quick Start
---

Install the wheel produced by the release or local package stage:

```bash
python -m pip install /path/to/tensorrt_model_connect-0.1.0-*.whl
```

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

Inspect the shared routing header and family-owned section inventory without
loading native code:

```bash
trtmc inspect gpt2.bundle
```

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
TRTMC_RUNTIME_ROOT="${TRTMC_RUNTIME_ROOT:-/opt/trtmc/lib}"
trtmc run gpt2.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "Hello" \
  --max-new-tokens 32
```

The shell variable above is only a convenient explicit argument. The CLI never
searches environment variables, the current directory, or another install.

GPT-2 needs no family-specific Python packages beyond the base environment.
For another family that owns a `requirements.txt`, install that exact file
before its build. From a checkout or unpacked source release:

```bash
python -m pip install -r families/sana_wm/requirements.txt
```

The wheel carries the same family-owned file. Locate it from the installed
`families` package:

```bash
FAMILY_REQUIREMENTS="$(python -c 'from pathlib import Path; import families; print(Path(families.__file__).parent / "sana_wm" / "requirements.txt")')"
python -m pip install -r "$FAMILY_REQUIREMENTS"
```

A family without that file needs only the pinned base environment and the
project wheel.
