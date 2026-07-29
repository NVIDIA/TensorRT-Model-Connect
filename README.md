# TensorRT-Model-Connect

![TensorRT-Model-Connect build and runtime map](website/static/img/diagrams/trtmc-system-map.svg)

[Documentation site](https://nvidia.github.io/TensorRT-Model-Connect/) |
[First NLP Inference](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start) |
[GitHub Actions](https://github.com/NVIDIA/TensorRT-Model-Connect/actions) |
[Docs source](website/docs/intro.md)

TensorRT-Model-Connect turns HuggingFace-style checkpoints into deployable
`.trtfb` TensorRT bundles and runs them from a native C++ runtime. Native
bundles dispatch through `runtime_strategy` to a model DSO and backend DSO;
exact qualified optimized bundles carry `optimized_runtime.json` and an
embedded implementation DSO instead.

## Start Here

The current public path builds from source in the repository container, which
is pinned to NVIDIA's official TensorRT 11.1.0.106 distribution. Release
wheels, when published, install the native `trtmc` executable, Python builder
dependencies including TensorRT, and the TensorRT backend DSO. Neither a wheel
nor a bundle is a hermetic operating-system or GPU-runtime image: the execution
environment must still resolve a compatible NVIDIA driver, CUDA/TensorRT
cohort, dynamic loader, and system libraries.

To build and run native Wan2.2 TI2V-5B at 720p, follow the
[optional Jetson Thor example](website/docs/getting-started/quick-start.md#optional-advanced-example-jetson-thor-wan22-720p).

For source development, open Codex or another repo-aware coding agent and ask:

```text
Clone https://github.com/NVIDIA/TensorRT-Model-Connect, set up the dev
container for this machine, build the project, build a Qwen/Qwen3-0.6B bundle,
and run the C++ smoke test. Report the commands you ran.
```

Success means the final `./build/trtmc run` command prints generated text.

## Manual Fallback

Use [Getting Started](website/docs/getting-started/overview.md) for the full
prerequisites-to-first-inference path. The short version is:

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect

./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh

pip install -e . -C py-only=true
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/trtmc build Qwen/Qwen3-0.6B
./build/trtmc run Qwen3-0.6B.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

The editable install is intentionally Python-only. It points imports at
`python/tensorrt_model_connect/` for fast builder iteration; CMake still builds
the source-tree `./build/trtmc` executable used by this workflow. Release and
CI validation use built wheels instead.

If CMake says the TensorRT backend was skipped, follow the [Installation](website/docs/getting-started/installation.md) TensorRT path instructions before running a model.

Source-built release wheels use `py310-none-manylinux_2_39_aarch64` and
`py312-none-manylinux_2_39_aarch64`; use the tag matching your Python
interpreter. The `manylinux_2_39_aarch64` platform tag matches the TensorRT
11 CUDA 13 aarch64 stack and requires a glibc 2.39 or newer Linux host. CI
builds and tests the wheel in the repository Dockerfile image before release.

## Useful Docs

| Need | Link |
| --- | --- |
| Complete the newcomer path | [Getting Started](website/docs/getting-started/overview.md) |
| Learn from beginner to advanced | [Learn & Tutorials](website/docs/learning-path.md) |
| Use CLI, Python, or C++ APIs | [API Reference](website/docs/api/overview.md) |
| Understand internals | [Architecture & Design](website/docs/architecture/overview.md) |
| Add or contribute functionality | [Contribute & Extend](website/docs/extend/overview.md) |
| Research features and design history | [Feature Reference & Context](website/docs/features/overview.md) |
