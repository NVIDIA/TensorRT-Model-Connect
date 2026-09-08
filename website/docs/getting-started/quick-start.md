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
single-directory plugin root in this order:

1. the directory containing the active `libtrtmc_runtime.so`;
2. colon-separated directories in `TRTMC_RUNTIME_PATH`.

The wheel console command replaces itself with the native CLI stored beside
Core, Runtime, backend, and family DSOs. Native installs resolve their already
loaded Runtime directory in the same way, so discovery does not inspect Python
installation layouts or scan `PATH` directories.

A complete GPT-2 TensorRT plugin root contains root-local
`libtrtmc_backend_trt.so` and `libtrtmc_model_gpt2.so` files. Candidates are
never combined across directories, and discovery never loads a candidate just
to inspect it. After selection, the Runtime Loader loads those exact paths and
requires their descriptors to match the active product build, plugin kinds,
and bundle IDs before it calls either factory. A mismatched selected root fails
immediately without falling back to another installation.

Before bundle inspection crosses a C++ interface, the CLI verifies that its
own product-build identity matches the already loaded Runtime and Core. The
Runtime Loader repeats the Core check and validates selected plugins at load.

The CLI prints the automatically selected directory. Use `--runtime-root DIR`
to override it with one exact root. An explicit root bypasses discovery but not
build and identity validation. The current directory is not searched
implicitly; use `TRTMC_RUNTIME_PATH=.` when that behavior is intended.
`LD_LIBRARY_PATH` remains a platform-loader setting evaluated before the CLI
starts.
