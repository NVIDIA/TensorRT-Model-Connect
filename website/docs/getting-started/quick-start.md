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

After the wheel is installed, build and run a supported model in two commands:

```bash
trtmc build Qwen/Qwen3-0.6B \
  --max-sequence-length 16384 \
  --output qwen3-0.6b.bundle
trtmc run qwen3-0.6b.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --enable-thinking false
```

The CLI downloads the snapshot, reads `config.json` or `model_index.json`, and
asks every dependency-free family `support.py`. Exactly one family must claim
the checkpoint. That family supplies the default task; pass `--task` only when
selecting another task supported by the same family. The build then imports
only the selected `families.qwen.model` and calls `build(request, writer)` once.
A prepared local snapshot can be passed in place of the model ID.

The installed `trtmc` command routes `build` to the existing Python builder and
executes the native CLI packaged beside the runtime DSOs for every other
command. A text family that owns a supported chat template enables it by
default; pass `--use-chat-template false` to request raw completion instead.

For a native CMake install, point the loader at the directory containing the matching
`libtrtmc_core.so`, `libtrtmc_runtime.so`, `libtrtmc_backend_trt.so`, and
selected family DSO. The loader reads the bundle header, loads exactly
those DSOs, and returns the abstract task interface declared by the bundle.

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello"
```

An explicit `--runtime-root` always wins. Without it, the native CLI accepts a
directory only when the exact core, runtime, backend, and family files are all
present. It checks the current directory first and the real executable
directory second. It does not scan environment variables or other install
locations.
