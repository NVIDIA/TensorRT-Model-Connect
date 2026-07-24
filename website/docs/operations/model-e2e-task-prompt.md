# Model E2E Task Checklist

Use this checklist for single-model bring-up. The three model-owned
descriptors, not a generic strategy table, define the contract.

## Inputs

- Family ID: the common `<family>` directory name
- Hugging Face model ID and immutable revision
- Python descriptor:
  `python/tensorrt_model_connect/families/<family>/MODEL.toml`
- Runtime descriptor:
  `src/runtime/models/<family>/MODEL.toml`
- E2E descriptor:
  `tests/e2e/models/<family>/MODEL.toml`
- Runtime strategy: an exact key declared by the runtime descriptor
- Task strategy: the runner/comparator contract declared by the E2E manifest

Do not substitute generic labels such as `decoder_kv_cache` or
`vision_language` for a family-owned strategy. For example, current Qwen text
generation uses `qwen_decoder_kv_cache`.

## Required implementation evidence

1. The Python descriptor resolves the intended checkpoint family.
2. The bundle records the exact family-owned `runtime_strategy`.
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

Replace `<family>` and `<manifest-name>` with literal values before running:

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
  tests/e2e/models/<family> \
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
