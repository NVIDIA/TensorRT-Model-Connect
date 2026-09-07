---
title: Runtime Lifecycle
description: How a bundle names one backend and one family DSO and returns an abstract Task interface.
---

import Diagram from '@site/src/components/Diagram';

Runtime loading is explicit and bounded. The loader never searches the current
working directory, environment variables, or fallback locations.

<Diagram
  src="/img/diagrams/architecture/native-bundle-load.svg"
  alt="The runtime loader reads a bundle header, opens the exact backend and family DSO from an explicit root, and returns an abstract Task interface"
  caption="Bundle identity selects one backend and one family; failures are terminal."
/>

## Load sequence

1. The application calls `trtmc::load_task(bundle, runtime_root)`.
2. `BundleReader` validates bounded header and section ranges.
3. The loader accepts only safe family and backend identifiers.
4. It opens `libtrtmc_backend_<backend>.so` from `runtime_root`.
5. It opens `libtrtmc_model_<family>.so` from the same root.
6. The loader calls the family factory with a read-only bundle reader and the
   abstract backend.
7. The factory returns an `ITask` implementation.
8. The loader verifies that `ITask::task()` matches the bundle header.

Any missing DSO, rejected identifier, task mismatch, or factory error stops the
load. There is no registry search or fallback family.

## Command-line execution

```bash
trtmc run qwen3.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello" \
  --max-new-tokens 32
```

`trtmc inspect BUNDLE` reads the header without loading a backend or family.
Every execution command requires `--runtime-root`.

## Request execution

<Diagram
  src="/img/diagrams/architecture/text-generation-request.svg"
  alt="A family-owned text pipeline tokenizes, executes prefill and decode engines, samples, and returns a TextResult"
  caption="Request loops, tokenizer state, KV state, sampling, and stopping policy belong to the family."
/>

The application calls an abstract interface such as `ITextGeneration`,
`ITranscription`, or `IImageGeneration`. The concrete family pipeline owns
preprocessing, bindings, engine order, state, and postprocessing. Engines are
created and enqueued through the abstract Engine API.

The optional TensorRT-RTX backend uses the same Engine interface and load
sequence. It is a backend selection, not a second model-runtime framework.

<Diagram
  src="/img/diagrams/architecture/optimized-bundle-load.svg"
  alt="The optional TensorRT-RTX backend is loaded through the same Engine and family factory contracts"
  caption="TensorRT-RTX follows the same Task and family path; only the Engine implementation changes."
/>
