# Qwen TensorRT Edge-LLM adapter

This is an explicit, Qwen-owned runtime adapter. It is never selected automatically.
Native Qwen builds remain unchanged unless `--backend edge_llm` is present.

Install the family dependency and build the official TensorRT Edge-LLM native
targets first. The `tensorrt-edgellm-export` and `llm_build` executables must be
on `PATH`. Point `EDGELLM_PLUGIN_PATH` at the plugin produced by that same
native build when invoking `llm_build` requires it.

Configure ModelConnect with those existing source and build directories, then
build the native CLI and Qwen Edge-LLM runtime:

```bash
python -m pip install -r families/qwen/edge_llm/dependencies.txt

EDGE_LLM_SOURCE=/path/to/TensorRT-Edge-LLM
EDGE_LLM_BUILD="$EDGE_LLM_SOURCE/build"
export PATH="$EDGE_LLM_BUILD/examples/llm:$PATH"
export EDGELLM_PLUGIN_PATH="$EDGE_LLM_BUILD/libNvInfer_edgellm_plugin.so.1.0"

cmake -S . -B build-edge \
  -DTRTMC_EDGE_LLM_SOURCE_DIR="$EDGE_LLM_SOURCE" \
  -DTRTMC_EDGE_LLM_BUILD_DIR="$EDGE_LLM_BUILD"
cmake --build build-edge --target trtmc trtmc_qwen_edge_llm_runtime
```

Build and inspect a bundle:

```bash
python -m tensorrt_model_connect build /path/to/Qwen3-0.6B \
  --backend edge_llm \
  --precision fp16 \
  --max-sequence-length 4096 \
  --output qwen3-edge.bundle

build-edge/trtmc inspect qwen3-edge.bundle
```

Run through the independent runtime directory. The bundle contains the engine
directory files, but not the Edge-LLM runtime bridge or TensorRT plugin:

```bash
build-edge/trtmc run qwen3-edge.bundle \
  --runtime-root build-edge \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 16 \
  --temperature 0 \
  --top-k 1
```

The runtime root contains the normal ModelConnect runtime and Qwen family DSO,
plus `libtrtmc_qwen_edge_llm.so` and `libNvInfer_edgellm_plugin.so`. Edge-LLM
owns engine execution, so this path does not install a fake Engine API backend.

The current integration is FP16, text-only, and single-device. A missing tool,
unsupported request, export failure, engine-build failure, or runtime-library
failure is terminal; the command does not switch to native Qwen.

The real GPU entry is opt-in and is not part of the CPU contract suite:

```bash
TRTMC_QWEN_EDGE_LLM_E2E=1 \
TRTMC_EDGE_LLM_QWEN3_0_6B_DIR=/path/to/Qwen3-0.6B \
TRTMC_BINARY="$PWD/build-edge/trtmc" \
TRTMC_RUNTIME_ROOT="$PWD/build-edge" \
PYTHONPATH=core/builder:. \
python -m pytest \
  families/qwen/tests/test_e2e.py::test_qwen3_0_6b_edge_llm_build_inspect_and_run -q
```
