# Hugging Face versus TensorRT-Model-Connect

Hugging Face execution and TensorRT-Model-Connect serve different roles in the
repository's validation loop.

| Concern | Hugging Face reference | TensorRT-Model-Connect |
| --- | --- | --- |
| Model execution | Framework model executes eagerly/compiled | Python builds TensorRT; C++ executes bundle |
| Family selection | Transformers auto classes/config | Family `MODEL.toml` plus plugin matching |
| Weights | Framework modules load checkpoint tensors | Family checkpoint mapper feeds TensorRT graph |
| Artifact | Checkpoint/config/tokenizer | Self-describing `.trtfb` bundle |
| Runtime dispatch | Python model class | Bundle strategy to model DSO/plugin |
| State | Framework cache/model objects | Family-owned C++ runtime state |
| Validation role | External reference oracle | System under test |

## Build and run

```bash
PYTHONPATH=python python3 -m tensorrt_model_connect build \
  Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-0.6b.trtfb

./build/trtmc run /tmp/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 16
```

The first command needs the Python/TensorRT build environment and checkpoint
access. The second needs the compiled CLI, TensorRT/CUDA, a compatible GPU, and
the Qwen model DSO.

## Parity

The E2E manifest's `task_strategy` selects the appropriate runner/comparator.
Text generation can compare tokens or logits; other modalities use
task-specific contracts. A model is not parity-qualified merely because it
builds or produces plausible output.

Use a focused logit check for supported decoder families:

```bash
PYTHONPATH=python:. python3 tools/diff_logits.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --json /tmp/qwen3-logits.json
```

Retain the exact model revision, prompt, precision, bundle, tested code
revision, and comparison artifact.
