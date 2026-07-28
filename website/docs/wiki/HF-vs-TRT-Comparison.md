# Hugging Face versus TensorRT-Model-Connect

Hugging Face execution and TensorRT-Model-Connect serve different roles in the
repository's validation loop.

| Concern | Hugging Face reference | TensorRT-Model-Connect |
| --- | --- | --- |
| Model execution | Framework model executes eagerly/compiled | Python builds native TensorRT plans or invokes an exact qualified provider; C++ executes the bundle through `IPipeline` |
| Family selection | Transformers auto classes/config | Family `MODEL.toml` plus plugin matching |
| Weights | Framework modules load checkpoint tensors | Native family mapper feeds a TensorRT graph, or a qualified family adapter owns conversion |
| Artifact | Checkpoint/config/tokenizer | Self-describing `.trtfb` bundle |
| Runtime dispatch | Python model class | Native strategy to model DSO/plugin, or `optimized_runtime.json` to the embedded implementation DSO |
| State | Framework cache/model objects | Native family-owned state or delegated implementation-owned state behind `IPipeline` |
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
access. This model-only dense Qwen3 request follows Qwen's native default route,
so it skips optimized-provider probing and uses the native Qwen builder. The
second needs the compiled CLI, TensorRT/CUDA, and a compatible GPU. A native
bundle additionally needs the Qwen model/backend DSOs; an optimized bundle
carries its exact implementation DSO and artifact tree.

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
