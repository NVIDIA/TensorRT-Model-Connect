<div align="center">

<h1>TensorRT-Model-Connect</h1>

<p><strong>Self-contained model families that build TensorRT bundles and implement native C++ task APIs.</strong></p>

[Documentation](https://nvidia.github.io/TensorRT-Model-Connect/)&nbsp;&nbsp;|&nbsp;&nbsp;[Supported Models](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)&nbsp;&nbsp;|&nbsp;&nbsp;[Architecture](https://nvidia.github.io/TensorRT-Model-Connect/architecture/ai-native-horizontal-scaling)

</div>

## Architecture

Every model family is one complete vertical slice:

```text
families/<family>/
├── support.py        # checkpoint ownership + supported/default tasks
├── model.py          # plain build(request, writer); inheritance forbidden
├── requirements.txt  # optional family-owned build/reference/test dependencies
├── runtime/          # one libtrtmc_model_<family>.so
└── tests/            # family-owned manifests and validation
```

The shared core is intentionally narrow. Python reads standard Hugging Face
metadata, asks every lightweight family `support.py`, requires exactly one
owner, and imports only that family's `model.py`. C++ reads the bundle header,
loads exactly the named backend and family DSO, and returns an abstract
interface from `trtmc/task.h`. Family implementations depend on those
interfaces; they do not depend on each other.

There is no central model registry, runtime-strategy switch, builder base
class, compatibility layer, or fallback path.

GPU build/reference/E2E environments use one pinned base image. A family may
add a plain `requirements.txt` in its own directory; changing it does not
publish another image digest or modify a central dependency registry.

## Build a bundle

```bash
python -m tensorrt_model_connect build openai-community/gpt2 \
  --precision fp16 \
  --output gpt2.bundle
```

The positional model may be a Hugging Face model ID or local snapshot. The
matching family owns its model identities, supported tasks, meaningful default
task, TensorRT graph, weights, section names, and runtime configuration. Use
`--task` only to override the family default. The bundle container stores only
`format`, `family`, `task`, `backend`, and section offsets and lengths.

Two existing opt-in paths remain direct rather than becoming configuration
systems:

```bash
# Llama runtime-sized KV cache
python -m tensorrt_model_connect build MODEL -o model.bundle --dynamic-kv-cache
trtmc run model.bundle --runtime-root /opt/trtmc/lib \
  --kv-cache-size 1GiB --prompt "Hello"

# TensorRT-RTX, when its Python package and optional backend DSO are installed
python -m tensorrt_model_connect build MODEL -o model.bundle --backend trt_rtx
trtmc run model.bundle --runtime-root /opt/trtmc/lib \
  --runtime-cache kernels.cache --cuda-graphs --prompt "Hello"
```

## Load from C++

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>

auto task = trtmc::load_task("gpt2.bundle", "/opt/trtmc/lib");
auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
if (text == nullptr) throw std::runtime_error("unexpected task");
auto result = text->generate("What is TensorRT?");
```

The runtime directory must contain the native core, runtime loader, requested
backend, and exact family DSO produced by the same build.

## Applications and tools

Applications depend one way on public ModelConnect APIs; core and model
families never depend on application code.

- [Bring your own kernel](examples/byok/README.md) connects an explicit
  TVM-FFI kernel DSO to a family-owned TensorRT graph, either directly in the
  family or through the optional pre-serialization graph transform.
- [Persistent audio streaming](examples/audio_streaming/README.md) loads one
  bundle once and serves one prompt per input line through the public Task API.
- [Cosmos3 dual Spark](examples/models/cosmos3/dual_spark/README.md) runs the
  CP=2 video pipeline across two DGX Sparks.
- [Nemotron VoiceChat full duplex](examples/models/nemotron_voicechat/full_duplex/README.md)
  provides the live ALSA microphone/speaker application.
- `trtmc-bench` and [the performance matrix](apps/benchmark/performance/README.md)
  measure the public Task APIs without adding behavior to core.

## Add a family

Adding a family means adding one directory, including its own `support.py`. It
must not require editing core source lists, registries, or sibling families.
Similar code is copied until a real shared contract—not code similarity—proves
a shared boundary is needed.

If the family needs packages beyond the base environment, install them with
`python -m pip install -r families/<family>/requirements.txt` before its build
or reference test. Native bundle deployment never reads this file or launches
Python.

See [Add a Model Family](website/docs/extend/add-model-family.md) and the
[full architecture](website/docs/architecture/ai-native-horizontal-scaling.md).

## Validate

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest -q
git diff --check
```

Then build and run the affected family's native target and real E2E recipe.
Do not weaken a correctness threshold to make CI pass.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md). Contributions require DCO sign-off.
TensorRT-Model-Connect is licensed under [Apache-2.0](LICENSE).
