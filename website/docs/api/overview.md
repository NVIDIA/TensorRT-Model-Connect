---
title: API and CLI Reference
description: Entry points for building, inspecting, loading, and running a family-owned bundle.
---

TensorRT-Model-Connect has three public surfaces:

| Surface | Use it for | Canonical implementation |
| --- | --- | --- |
| Python build CLI and API | resolve one family and build a `.bundle` | `core/builder/tensorrt_model_connect/` |
| Native `trtmc` CLI | inspect or execute a bundle | `apps/cli/` |
| C++ Task API | embed loading and task calls in an application | `core/runtime/include/trtmc/` |

Model families implement these contracts. They do not create new public entry
points or require callers to include family headers.

## Build

```bash
python -m tensorrt_model_connect build openai-community/gpt2 \
  --precision fp16 \
  --output gpt2.bundle
```

The CLI accepts a Hugging Face ID or local snapshot, resolves exactly one
`families/*/support.py`, and calls only that family's plain `model.py::build`
function.

See [Python Build API](python-builder.md).

## Inspect and run

```bash
trtmc inspect gpt2.bundle

trtmc run gpt2.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello" \
  --max-new-tokens 32
```

Inspect reads only the bounded bundle header. Every execution command requires
an explicit runtime root containing the matching core, loader, backend, and
family DSO.

See [CLI Reference](cli-reference.md).

## Embed from C++

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>

auto task = trtmc::load_task("gpt2.bundle", "/opt/trtmc/lib");
auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
if (text == nullptr) throw std::runtime_error("unexpected task");
auto result = text->generate("Hello");
```

See [C++ Task API](cpp-api.md).

## Contract boundary

The bundle header carries `format`, `family`, `task`, and `backend`.
The loader returns the abstract interface named by `task`. Family-specific
options, section meanings, tokenizers, execution order, and postprocessing stay
inside the family.

There is no shared runtime-strategy registry, config-schema registry, provider
profile, Python runtime fallback, or compatibility API.
