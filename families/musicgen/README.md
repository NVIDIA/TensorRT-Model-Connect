# MusicGen medium

This family implements `facebook/musicgen-medium` as the native `text_to_audio`
Task. The bundle contains three FP32 TensorRT engines: the T5 text encoder and
projection, MusicGen's four-codebook causal decoder, and the non-causal 32 kHz
EnCodec decoder. Both codec LSTMs execute inside TensorRT loops. Native inference
does not invoke Python or PyTorch.

The initial path uses one GPU, one prompt, mono PCM, and a bounded sequence that
is recomputed at each generation step. It does not implement KV caching, tensor
parallelism, quantization, audio continuation, stereo, or melody conditioning.
Other MusicGen model sizes are not claimed by this owner.

## Build and run

Prepare the repository's documented development environment, install this
family's `requirements.txt`, then download the checkpoint:

```python
from huggingface_hub import snapshot_download

checkpoint = snapshot_download(
    "facebook/musicgen-medium",
    revision="d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "*token*.json",
        "special_tokens_map.json",
        "spiece.model",
        "pytorch_model.bin",
    ],
)
```

Build the one-second qualification case from the repository root:

```python
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build

build(
    BuildRequest(
        model_dir=Path(checkpoint),
        output_path=Path("musicgen-medium.bundle"),
        family="musicgen",
        task="text_to_audio",
        precision="fp32",
        max_sequence_length=54,
    )
)
```

The optional sequence bound includes the four delayed BOS/EOS positions. A
bound of 54 produces 50 codec frames, or exactly one second of mono 32 kHz audio.
The default bound is 256 (5.04 seconds); accepted bounds are 11 through 1504.
The checked-in qualification covers bound 54. Prompts must fit in 64 T5 tokens,
including EOS; longer prompts fail without silent truncation. The bundle stores
the tokenizer backend produced by the repository-pinned Transformers version,
including its actual normalization and special-token framing.

Build the `trtmc`, `trtmc_backend_trt`, `trtmc_model_musicgen`, and
`test_musicgen_tokenizer` CMake targets. Enable `TRTMC_BUILD_TESTS` for the last
target and the public SDK consumer. Then run:

```bash
build-musicgen/trtmc generate-audio musicgen-medium.bundle \
  --runtime-root build-musicgen \
  --prompt "A mellow jazz piano with soft drums." \
  --set do_sample=false --set guidance_scale=3.0 --output musicgen.wav
```

The Task defaults to sampling with `top_k=50`, `temperature=1.0`,
`guidance_scale=3.0`, and `seed=0`. Each request owns its RNG and sequence state.
Equal sampling seeds reproduce native output; native sampled tokens are not
claimed to match PyTorch's RNG bit for bit. Set `do_sample=false` for the
official-reference greedy comparison.

## Validation

```bash
export TRTMC_MUSICGEN_MODEL_DIR=/absolute/path/to/pinned/checkpoint
export TRTMC_BINARY="$PWD/build-musicgen/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build-musicgen"
export TRTMC_E2E_ARTIFACT_DIR="$PWD/artifacts/musicgen"
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/musicgen/tests --e2e-model musicgen -q
build-musicgen/test_musicgen_sdk musicgen-medium.bundle build-musicgen
git diff --check
```

The E2E test builds a fresh bundle unless `TRTMC_MUSICGEN_BUNDLE` explicitly names
an existing one. It compares two distinct prompts with official Transformers
5.2.0 FP32 greedy generation at guidance scale 3.0. Both outputs must have the
exact sample count, finite PCM, waveform cosine at least 0.999, and relative
RMSE at most 0.01. A separate check requires prompt-dependent audio. Native
tokenizer fixtures check Unicode and whitespace against the official backend.
The C++ SDK consumer tests seeded sampling, repeat calls on one loaded model,
invalid configuration, prompt bounds, and recovery after invalid requests.

Checkpoint identity and upstream implementation:
[model card](https://huggingface.co/facebook/musicgen-medium),
[MusicGen reference](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/musicgen/modeling_musicgen.py),
[EnCodec reference](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/encodec/modeling_encodec.py).
The upstream model weights use CC-BY-NC-4.0.
