# Qwen3-VL MoE

This family owns `Qwen3VLMoeForConditionalGeneration` with model type
`qwen3_vl_moe`. Its recipe uses `Qwen/Qwen3-VL-30B-A3B-Instruct` at revision
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` for image-language generation.

The decoder implements grouped-query attention, interleaved multimodal RoPE,
top-8 routed experts, and the three vision DeepStack injections. Expert
weights use the checkpoint's packed input/output layout. BF16 constants retain
the checkpoint's BF16 range. TensorRT lowers the graph; the family supplies no
GPU kernels. The current graph evaluates all experts and weights their outputs
with the sparse router probabilities.

One serialized decoder engine has separate prefill and decode profiles. The
native runtime shares that engine between its two execution contexts. A second
engine owns dynamic vision encoding and the main and DeepStack features.

## Build and run

Download the pinned checkpoint, including all thirteen safetensors shards, to
`/path/to/checkpoint`. Prepare the repository's matched TensorRT environment
using the environment setup skill, then build from the repository root:

```python
from tensorrt_model_connect import BuildRequest, build

build(
    BuildRequest(
        model_dir="/path/to/checkpoint",
        output_path="qwen3-vl-30b-a3b-bf16.bundle",
        family="qwen3_vl_moe",
        task="vision_language_generation",
        precision="bf16",
        max_sequence_length=512,
    )
)
```

Configure the native build with the matched SDK settings described in the
[source-build guide](../../website/docs/getting-started/source-build.md), then
build the CLI, TensorRT backend, and family DSO. Run using their build directory:

```bash
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_qwen3_vl_moe -j 8

build/trtmc run qwen3-vl-30b-a3b-bf16.bundle \
  --runtime-root build \
  --image families/qwen3_vl_moe/tests/data/test_img.jpeg \
  --prompt 'What color is the vehicle in this image? Answer in one word.' \
  --max-new-tokens 10 --temperature 1 --top-k 1
```

The recipe uses one GPU, batch one, BF16, and a 512-token cache. The multirow
prefill profile accepts up to 64 tokens; longer prompts currently prefill one
token at a time. The declared image cases exercise that sequential path.
Images preserve their aspect ratio and are resized within 65,536 to 200,704
pixels. Prompt and generated tokens together must fit the cache. Quantization,
tensor or context parallelism, video, and runtime-sized KV caches are
unsupported. FP32 is available for component debugging; this checkpoint recipe
uses BF16.

## Validation

The manifest contains original and mirrored vehicle-image cases. The official
Transformers reference uses the same processor bounds, BF16 weights, prompt,
and greedy generation. The native answer must pass the existing Qwen-VL
canonicalization and normalized edit-distance limit of 0.15. Separate small
graphs compare routed-expert and dynamic vision/DeepStack outputs against the
official reference modules.

```bash
PYTHONPATH=core/builder:apps/benchmark:. \
  TRTMC_QWEN3_VL_MOE_COMPONENTS=1 \
  python -m pytest families/qwen3_vl_moe/tests/test_components.py \
    families/qwen3_vl_moe/tests/test_vision.py -q

PYTHONPATH=core/builder:apps/benchmark:. \
  TRTMC_QWEN3_VL_MOE_MODEL_DIR=/path/to/checkpoint \
  TRTMC_BINARY="$PWD/build/trtmc" TRTMC_RUNTIME_ROOT="$PWD/build" \
  python -m pytest families/qwen3_vl_moe/tests \
    --e2e-model qwen3_vl_moe -q
```

The E2E fixture builds a bundle unless `TRTMC_QWEN3_VL_MOE_BUNDLE` explicitly
selects an existing bundle built from the current family source.
