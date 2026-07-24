# TensorRT-Model-Connect

![TensorRT-Model-Connect overview](website/static/img/trtmc-landing.png)

[Documentation site](https://sturdy-broccoli-y7zg5w9.pages.github.io/) |
[Quick Start](https://sturdy-broccoli-y7zg5w9.pages.github.io/getting-started/quick-start) |
[GitHub Actions](https://github.com/NVIDIA/TensorRT-Model-Connect/actions) |
[Docs source](website/docs/intro.md)

TensorRT-Model-Connect turns HuggingFace-style checkpoints into deployable `.trtfb` TensorRT bundles and runs them from a native C++ runtime.

## Start Here

Nightly GitHub Releases publish Linux aarch64 wheels for Python 3.10 and
Python 3.12. On a compatible NVIDIA GPU host, download the wheel that matches
your Python version and run:

```bash
python3.12 -m venv .venv-trtmc
. .venv-trtmc/bin/activate
pip install ./tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_aarch64.whl

trtmc version
trtmc build Qwen/Qwen3-0.6B
trtmc run qwen3-0.6b.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

For the exactly qualified `Qwen/Qwen3-0.6B` and
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` revisions, the model-only build
automatically emits split prefill/decode engines with runtime-owned KV
memory on the initial qualified target: GB300 (`sm103`) with TensorRT
`11.2.0.113`. A recognized model on another target fails with the expected and
actual target instead of silently producing a different kind of bundle.
Runtime defaults to 90% of the safely usable free GPU memory measured after
engine loading, capped by the model's context limit. Use
`--kv-cache-memory 80%` or `--kv-cache-memory 8GiB` to override it, and
optionally add the runtime-only `--max-sequence-length 4K` admission cap.
Other checkpoints retain their existing build and runtime route.

The wheel installs the native `trtmc` executable into the environment, the
Python builder dependencies including TensorRT, and the TensorRT backend DSO.
CUDA driver/runtime libraries still come from the host system.

To build and run native Wan2.2 TI2V-5B at 720p, follow the
[two-command Wan2.2 Quick Start](website/docs/getting-started/quick-start.md).

For source development, open Codex or another repo-aware coding agent and ask:

```text
Clone https://github.com/NVIDIA/TensorRT-Model-Connect, set up the dev
container for this machine, build the project, build a Qwen/Qwen3-0.6B bundle,
and run the C++ smoke test. Report the commands you ran.
```

Success means the final `./build/trtmc run` command prints generated text.

## Manual Fallback

Use the [Environment and First Repro](website/docs/getting-started/environment-and-repro.md) and [Quick Start](website/docs/getting-started/quick-start.md) guides for the full manual path. The short version is:

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect

./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh

pip install -e . -C py-only=true
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/trtmc build Qwen/Qwen3-0.6B
./build/trtmc run qwen3-0.6b.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

The editable install is intentionally Python-only. It points imports at
`python/tensorrt_model_connect/` for fast builder iteration; CMake still builds
the source-tree `./build/trtmc` executable used by this workflow. Release and
CI validation use built wheels instead.

If CMake says the TensorRT backend was skipped, follow the [Installation](website/docs/getting-started/installation.md) TensorRT path instructions before running a model.

Nightly wheels are tagged `py310-none-manylinux_2_39_aarch64` and
`py312-none-manylinux_2_39_aarch64`; use the tag matching your Python
interpreter. The `manylinux_2_39_aarch64` platform tag matches the TensorRT
11 CUDA 13 aarch64 stack and requires a glibc 2.39 or newer Linux host.
CI package jobs build and test wheels in the repository Dockerfile image
(`TRTMC_CI_IMAGE`, derived from repository variable `TRTMC_MANYLINUX_CI_IMAGE`
or default `trtmc-dev-gb300:manylinux_2_39`) so the compiled
native executable is actually checked against that platform floor.

## Useful Docs

| Need | Link |
| --- | --- |
| Learn the project | [Learning Path](website/docs/learning-path.md) |
| Build and run models | [Build and Run](website/docs/getting-started/build-and-run.md) |
| Check model coverage | [Model Support](website/docs/getting-started/model-support.md) |
| Use CLI, Python, or C++ APIs | [API Overview](website/docs/api/overview.md) |
| Understand internals | [Architecture](website/docs/architecture/overview.md) |
| Run validation | [Testing](website/docs/reference/testing.md) |
| Measure model performance | [Performance Benchmarking](website/docs/reference/benchmarking.md) |
