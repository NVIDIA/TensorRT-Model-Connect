# Phonon 31m English

This family implements `gradium/phonon-31m-en` revision
`92741c5b07d085ff840e26fea8ce43110b87852f` as native TensorRT text-to-speech.
The bundle contains three FP32 plans: the causal transformer, the one-step LSD
flow sampler, and the Mimi waveform decoder. Native inference requires no
Python or PyTorch. The checkpoint's default voice is projected into the bundle.

The initial implementation supports one GPU, batch size one, normalized ASCII
English prose, and complete 24 kHz mono waveforms. It uses full-context
transformer evaluation and decodes the collected latents once. Streaming,
custom voices, Unicode text normalization, quantization, and reduced precision
are not implemented. Unsupported build options and languages are rejected.
Supply spelled-out numbers and normalized punctuation when needed. Input
whitespace is collapsed, its first character capitalized, and a missing final
period appended to text ending in an alphanumeric character.

## Build and run

Prepare the checkout using `apps/devtoolkit`; select `family="phonon"` to install
this family's optional reference dependency. The builder uses the base
PyTorch, ONNX, safetensors and TensorRT packages. The additional Transformers
dependency is pinned to the official implementation used by the tests.

From the checkout root in the prepared development environment:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100-real \
  -DTRTMC_BUILD_TESTS=ON -DTRTMC_BUILD_EXAMPLES=OFF -DTRTMC_BUILD_SERVER=OFF
cmake --build build --parallel 8 --target \
  trtmc trtmc_backend_trt trtmc_model_phonon \
  test_phonon_tokenizer test_phonon_sdk_c test_phonon_sdk_cpp
```

Set `CMAKE_CUDA_ARCHITECTURES` to the executing GPU's compute capability.
Download the checkpoint with `huggingface_hub.snapshot_download`, supplying
the exact revision above, then build through the public Python API:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from tensorrt_model_connect import BuildRequest, build

checkpoint = snapshot_download(
    "gradium/phonon-31m-en",
    revision="92741c5b07d085ff840e26fea8ce43110b87852f",
    allow_patterns=["*.json", "model.safetensors", "default-voice.safetensors"],
)
build(
    BuildRequest(
        model_dir=Path(checkpoint),
        output_path=Path("phonon.bundle"),
        family="phonon",
        task="text_to_speech",
        precision="fp32",
        max_sequence_length=256,
    )
)
```

Run Python with `PYTHONPATH=core/builder:.` when using the source checkout.
The sequence budget includes voice frames, text tokens, BOS and generated
audio frames. Requests that cannot complete within their budget fail explicitly.
The runtime also applies the upstream text-dependent frame cap and EOS tail.

```bash
build/trtmc generate-audio phonon.bundle --runtime-root build \
  --prompt "Hello from Phonon." --language en --output speech.wav
```

The family declares `temperature` (Gaussian variance, default `0.3`) and `seed`
(an integer from `0` to `4294967295`, default `0`) as Task configuration fields.
Add `--set temperature=0` for the
deterministic reference path, or `--set seed=42` for another stochastic sample.
Both the C `TRTMC_TASK_TEXT_TO_SPEECH` API and C++ `trtmc::TextToSpeech` wrapper
use these same fields. Every call starts with fresh generation state.

## Validation

```bash
export TRTMC_BINARY="$PWD/build/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build"
export TRTMC_NATIVE_BUILD_DIR="$PWD/build"
# Optional: explicit local copy of the pinned checkpoint.
export TRTMC_PHONON_MODEL_DIR=/absolute/path/to/checkpoint
# Optional: a separate interpreter containing the pinned family requirements and torch.
export TRTMC_PHONON_REFERENCE_PYTHON=/absolute/path/to/reference/python

PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/phonon/tests --e2e-model phonon -q
ctest --test-dir build -R '^test_phonon_tokenizer$' --output-on-failure
git diff --check
```

Without `TRTMC_PHONON_MODEL_DIR`, tests require the exact revision in the local
Hugging Face cache. Without a separate reference interpreter, the test
interpreter must contain `families/phonon/requirements.txt`. Missing execution
prerequisites fail an explicitly selected E2E instead of producing a skip.

The independent oracle is the official Transformers Phonon implementation and
its official Mimi checkpoint conversion. It imports no family graph code.
Tests compare transformer conditions and EOS logits, a nonzero-noise flow
sample, the decoded official latents, and complete native waveforms. Short and
longer sentences exercise both EOS tail policies and Mimi's attention window.
The C/C++ consumers test public Task execution, repeated requests, unsupported
language rejection and retained result storage. Acceptance limits are owned
by `tests/test_e2e.py`; no upstream reference or passing criterion is relaxed.
