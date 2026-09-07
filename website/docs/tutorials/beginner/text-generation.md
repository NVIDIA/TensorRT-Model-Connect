---
title: Beginner Tutorial - Text Generation
---

import Diagram from '@site/src/components/Diagram';

Build and run one family through the same public boundaries used by native
applications.

## 1. Build a bundle

```bash
python -m tensorrt_model_connect build openai-community/gpt2 \
  --output gpt2.bundle \
  --precision fp16
```

The resolver selects `families/gpt2/support.py`, then imports only
`families/gpt2/model.py`. The builder is a plain function; it does not inherit
from a shared model base.

## 2. Read the bundle

```bash
trtmc inspect gpt2.bundle
```

Confirm `family=gpt2`, `task=text_generation`, and the intended backend before
loading the runtime.

<Diagram
  src="/img/diagrams/trtmc-inference-loop.svg"
  alt="A family text pipeline tokenizes once, prefills the prompt, then decodes and samples until a stop condition"
  caption="The concrete loop stays in the family behind the abstract ITextGeneration interface."
/>

## 3. Run generation

```bash
trtmc run gpt2.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "What is TensorRT?" \
  --max-new-tokens 32
```

The loader opens only the `gpt2` family and the backend named in the bundle.
The family factory returns `ITextGeneration`; the CLI calls `generate()` through
that abstract interface.

## 4. Change one sampling control

```bash
trtmc run gpt2.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "What is TensorRT?" \
  --max-new-tokens 32 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 42
```

Sampling flags are request inputs. A family must implement their semantics; the
shared Task API does not make all families behave identically.

## 5. Explain the result

Record the checkpoint, bundle command, runtime root, prompt, generation values,
and output. If comparing two runs, change one variable at a time. A readable
answer is a useful spot check, not a replacement for the exact family E2E.
