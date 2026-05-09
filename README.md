# TensorRT-Model-Connect

![TensorRT-Model-Connect overview](website/static/img/trtmc-landing.png)

[Documentation site](https://sturdy-broccoli-y7zg5w9.pages.github.io/) |
[Quick Start](https://sturdy-broccoli-y7zg5w9.pages.github.io/getting-started/quick-start) |
[GitHub Actions](https://github.com/NVIDIA/TensorRT-Model-Connect/actions) |
[Docs source](website/docs/intro.md)

TensorRT-Model-Connect turns HuggingFace-style checkpoints into deployable `.trtfb` TensorRT bundles and runs them from a native C++ runtime.

## Start Here

For the fastest setup, open Claude Code, Codex, or another repo-aware coding agent and ask:

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
cd TensorRT-Model-Connct

./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh

pip install -e tensorrt_model_connect/
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

trtmc-build build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256
./build/trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

If CMake says the TensorRT backend was skipped, follow the [Installation](website/docs/getting-started/installation.md) TensorRT path instructions before running a model.

## Useful Docs

| Need | Link |
| --- | --- |
| Learn the project | [Learning Path](website/docs/learning-path.md) |
| Build and run models | [Build and Run](website/docs/getting-started/build-and-run.md) |
| Check model coverage | [Model Support](website/docs/getting-started/model-support.md) |
| Use CLI, Python, or C++ APIs | [API Overview](website/docs/api/overview.md) |
| Understand internals | [Architecture](website/docs/architecture/overview.md) |
| Run validation | [Testing](website/docs/reference/testing.md) |
