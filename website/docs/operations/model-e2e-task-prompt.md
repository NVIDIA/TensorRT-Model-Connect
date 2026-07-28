# Model E2E Task Checklist

Use this checklist for a native single-model bring-up. Three linked model-owned
descriptors, not a generic strategy table or an assumed shared directory name,
define that contract. A delegated optimized-runtime qualification instead adds
a family-owned implementation manifest/profile and `QUALIFICATION.*.toml`; use
its producer proof in addition to, not as a replacement for, the native support
inventory.

## Inputs

- Builder family: the Python owner directory name, `<builder-family>`
- Runtime owner: the C++ DSO owner directory name, `<runtime-owner>`
- E2E family: the test owner directory name, `<e2e-family>`
- Hugging Face model ID and immutable revision
- Python descriptor:
  `python/tensorrt_model_connect/families/<builder-family>/MODEL.toml`
- Runtime descriptor:
  `src/runtime/models/<runtime-owner>/MODEL.toml`
- E2E descriptor:
  `tests/e2e/models/<e2e-family>/MODEL.toml`
- Runtime strategy: an exact key declared by the runtime descriptor
- Task strategy: the runner/comparator contract declared by the E2E manifest

The three names usually match, and each descriptor's `id` must match its own
directory. They are not required to be one physical name, however. The Python
plugin's `runtime_strategy` and the E2E manifest's `runtime_strategy` select
the runtime owner. Current compatibility examples are
`magpie_tts` → `magpie` and `wan_t2v` → `wan`; in both cases the builder and
E2E names use the left-hand value while the runtime directory uses the
right-hand value.

Do not substitute generic labels such as `decoder_kv_cache` or
`vision_language` for a family-owned strategy. For example, current Qwen text
generation uses `qwen_decoder_kv_cache`.

## Required implementation evidence

1. The Python descriptor resolves the intended checkpoint family.
2. The native bundle records the exact family-owned `runtime_strategy`.
3. The runtime descriptor maps that strategy to a model DSO and registration
   symbol.
4. CMake discovers the descriptor without a central plugin-list edit.
5. The E2E descriptor declares the manifest.
6. The JSON manifest contains `runtime_strategy`, `task_strategy`, and a
   non-empty `testcases` array.
7. Each testcase declares its user contract, CI tier, inputs, and comparison
   expectations.
8. Tests demonstrate meaningful reference parity; compilation alone is not
   parity evidence.

## Local checks

Replace `<e2e-family>` and `<manifest-name>` with literal values before
running:

```bash
DOC_REMOTE="github"
if ! git remote get-url "$DOC_REMOTE" >/dev/null 2>&1; then
  DOC_REMOTE="origin"
fi
DOC_REMOTE_URL=$(git remote get-url "$DOC_REMOTE")
case "$DOC_REMOTE_URL" in
  https://github.com/NVIDIA/TensorRT-Model-Connect|\
  https://github.com/NVIDIA/TensorRT-Model-Connect.git|\
  git@github.com:NVIDIA/TensorRT-Model-Connect.git|\
  ssh://git@github.com/NVIDIA/TensorRT-Model-Connect.git) ;;
  *)
    echo "Refusing non-canonical remote: $DOC_REMOTE_URL" >&2
    exit 1
    ;;
esac
git fetch "$DOC_REMOTE" main

PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py

PYTHONPATH=python:. python3 -m pytest \
  tests/builder/test_manifest_validation.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_model_plugin_encapsulation_static.py -q

PYTHONPATH=python:. python3 tools/model_ci.py validate

PYTHONPATH=python:. python3 tools/model_ci.py impact \
  --base "$DOC_REMOTE/main" \
  --head HEAD

PYTHONPATH=python:. python3 -m pytest \
  tests/e2e/models/<e2e-family> \
  --e2e-model <manifest-name> \
  --engine-dir /path/to/engines \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  -v
```

Check `python3 tools/model_ci.py --help` and
`python3 -m pytest --help` in the execution environment before adding
model-specific options. E2E execution additionally requires the checkpoint,
TensorRT/CUDA, suitable GPU capacity, the built CLI, and built model DSOs.

## Completion gate

Record the exact tested commit, command, hardware, generated bundle, comparison
artifact, performance artifact when claimed, and unresolved limitations. Do
not relax an acceptance threshold merely to obtain a passing result.
