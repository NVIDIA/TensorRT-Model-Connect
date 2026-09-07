---
title: C++ Task API
---

`load_task()` returns the abstract task declared by the bundle. A user casts
to that task interface, never to a family implementation:

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>

auto task = trtmc::load_task("gpt2.bundle", "/opt/trtmc/lib");
auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
if (text == nullptr) throw std::runtime_error("not a text-generation bundle");
trtmc::TextGenerationConfig config;
config.use_chat_template = text->default_use_chat_template();
auto result = text->generate("Hello", config);
```

The chat-template default is selected by the loaded family and checkpoint. An
application can set `config.use_chat_template` explicitly when it needs raw
completion or a template regardless of that default.

Each family DSO directly implements one or more interfaces from
`trtmc/task.h`. The loader verifies that `ITask::task()` exactly matches the
bundle header.

`load_task()` lives in `libtrtmc_runtime.so`; bundle reading and engine
primitives live in `libtrtmc_core.so`. A family factory receives a
`FamilyContext` containing a `const BundleReader&` and an abstract `IBackend&`.
The reader exposes immutable metadata and on-demand section reads only. The
factory must copy the lightweight reader into its pipeline if it will read a
section after the factory returns; it must never retain the context reference.
