# Qwen3-VL MoE

This family owns `Qwen3VLMoeForConditionalGeneration` with model type
`qwen3_vl_moe`. Its recipe uses `Qwen/Qwen3-VL-30B-A3B-Instruct` at revision
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` for image-language generation.

The decoder implements grouped-query attention, interleaved multimodal RoPE,
top-8 routed experts, and the three vision DeepStack injections. Expert
weights use the checkpoint's packed input/output layout. BF16 constants retain
the checkpoint's BF16 range. TensorRT lowers the graph; the family supplies no
GPU kernels. The graph gathers each token's selected expert weights before
GEMMs and accumulates their weighted outputs in ascending expert order.

One serialized decoder engine has separate prefill and decode profiles. The
native runtime shares that engine between its two execution contexts. A second
engine owns dynamic vision encoding and the main and DeepStack features.
Native builds require ICU's `uc` component for Unicode tokenizer categories;
its upstream attribution is in [NOTICE](NOTICE).

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
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_qwen3_vl_moe \
  test_qwen3_vl_moe_sdk -j 8

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
pixels. Prompt and requested generated tokens together must fit the cache;
oversized requests are rejected before prefill, and the cache does not slide. Quantization,
tensor or context parallelism, video, and runtime-sized KV caches are
unsupported. FP32 is available for component debugging; this checkpoint recipe
uses BF16.

## Validation

The Task SDK exposes `text_continuation` for UTF-8 prefixes and
`images_text_to_text` for one user turn with one RGB image followed by text. The image
Task accepts contiguous UInt8 RGB pixels or Float32 RGB values in [0,1]. Tools,
additional turns, additional images, and text before the image are rejected.
When `max_new_tokens` is omitted, the SDK bounds its default budget by the cache
space remaining after the encoded prompt. Explicit budgets that exceed this
space are rejected. LoRA management is advertised only when the loaded decoder
has adapter inputs.

The manifest contains original and mirrored vehicle-image cases. The official
Transformers reference uses the same processor bounds, BF16 weights, prompt,
and greedy generation. The native answer must pass the existing Qwen-VL
case, whitespace, and edge-punctuation normalization and the normalized
edit-distance limit of 0.15. Full normalized answers are compared; shared words
do not excuse negation or conflicting colors. The fixture also loads the bundle
through the public C++ Task SDK and checks an image answer before CLI execution. Separate small
graphs compare routed-expert and dynamic vision/DeepStack outputs against the
official reference modules.

```bash
PYTHONPATH=core/builder:apps/benchmark:. \
  python -m pytest families/qwen3_vl_moe/tests/test_components.py \
    families/qwen3_vl_moe/tests/test_vision.py -m 'gpu or trt' -q

PYTHONPATH=core/builder:apps/benchmark:. \
  TRTMC_QWEN3_VL_MOE_MODEL_DIR=/path/to/checkpoint \
  TRTMC_BINARY="$PWD/build/trtmc" TRTMC_RUNTIME_ROOT="$PWD/build" \
  TRTMC_NATIVE_BUILD_DIR="$PWD/build" \
  python -m pytest families/qwen3_vl_moe/tests \
    --e2e-model qwen3_vl_moe -q
```

The E2E fixture builds a bundle unless `TRTMC_QWEN3_VL_MOE_BUNDLE` explicitly
selects an existing bundle built from the current family source.
Build the native tests with `TRTMC_BUILD_TESTS=ON` (the CMake default). On a
CUDA-capable host, the clean-build CI path can be exercised with:

```bash
cmake --build build --target test_qwen3_vl_moe_kv_capacity \
  test_qwen3_vl_moe_runtime test_qwen3_vl_moe_thread_unwind -j 8
ctest --test-dir build -R '^qwen3_vl_moe_' --output-on-failure
```

These targets also build the tokenizer and SDK helpers. Tests find the helpers
in `TRTMC_NATIVE_BUILD_DIR` while `TRTMC_RUNTIME_ROOT` supplies the runtime
libraries. A combined local build directory remains supported. GPU component
tests use the standard `gpu`/`trt` markers, so CI executes them in its hardware
stage and rejects skips through its existing JUnit validator.

## Quantization study (not shipped)

FP8 and NVFP4 were built and measured for `Qwen3-VL-30B-A3B-Instruct`. Neither
meets the rules for a shipped build (faster than BF16 on B200 and B300 for FP8;
at least 0.8x the FP8 speed on both GPUs and at most about 2x the FP8 loss for
NVFP4), so the family keeps BF16 only.

Checkpoints: there is no NVIDIA FP8 or NVFP4 export of this model. Qwen's
`Qwen3-VL-30B-A3B-Instruct-FP8` uses 128x128 block weight scales and dynamic
activations, which TensorRT runs only as weight-only FP8 (slower than BF16), so
the weights were quantized from the BF16 checkpoint at build time. Layer list
from the official exports (Qwen FP8; `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4`
for NVFP4): every language-model linear, but not the routers, `lm_head` or the
vision tower. `o_proj` stays BF16 here because an FP8 Quantize right after the
fused attention does not build on TensorRT 11.4.

Recipe: FP8 W8A8 with a per-projection weight scale `amax / 448` and a static
per-tensor activation scale (max calibration over 384 COCO images, each with a
question and a cnn_dailymail or WikiText-2 text as the answer, up to 512
tokens; no layer was above 20x the median). NVFP4 W4A4 with block-16 e4m3 weight scales,
`weight_scale_2 = amax / (6 x 448)` and dynamic activation quantization with
the FP8 calibration amax / 6 as global scale. Routed experts use one weight
scale and one input scale per layer and projection.

Setup: TensorRT 11.4, B200 and B300, `max_sequence_length` 512 (the manifest),
the family builder settings unless noted. Decode is one engine step after the
204-token vehicle prompt; request is that prompt one token at a time (the
runtime path for prompts above 64 tokens) plus 64 new tokens. Variants of one
row were built and timed in one job, 7 or 15 interleaved rounds, median.
Quality is teacher-forced over 8307 text tokens after the image (26 held-out
COCO images, each with a question and a news or Wikipedia answer): KL(PyTorch
BF16 || engine) per token, top-1 agreement and perplexity (PyTorch BF16 11.20).

| Engine | Plan | Decode B200 | Decode B300 | KL mean / p99 | Top-1 | Perplexity |
|---|---|---|---|---|---|---|
| BF16 (default) | 56.9 GiB | 6.6 to 7.0 ms | 6.4 to 6.7 ms | 0.0049 to 0.0053 / 0.031 | 97.5 % | 11.19 to 11.22 |
| FP8 q/k/v | 56.4 GiB | 1.02x to 1.05x (3 runs) | 1.00x to 1.03x (4 runs) | 0.0073 / 0.059 | 96.5 % | 11.23 |
| FP8 routed experts | 29.9 GiB | 0.70x | 0.58x | 0.0079 to 0.0088 / 0.058 | 96.4 % | 11.23 to 11.25 |
| FP8 routed experts, builder level 3 | 29.9 GiB | 1.08x to 1.10x | 0.79x to 0.81x | 0.0087 / 0.065 | 96.1 % | 11.26 |
| FP8 experts + q/k/v, level 3 | 29.5 GiB | 1.12x | 0.82x | 0.0101 / 0.082 | 95.9 % | 11.26 |

Speed ratios are against BF16 at the same builder level (level 3 alone gives
BF16 1.00x to 1.03x). The request time follows decode within 1 %.

- FP8 q/k/v is 2 % to 5 % faster on B200 but 0 % to 1 % on B300 in three of four
  runs (request 1.00x to 1.01x), and the plan is only 0.8 % smaller: not a
  speedup on both GPUs.
- FP8 routed experts halve the plan, but on B300 they are slower. The selected
  experts are gathered per token before the GEMMs; the fused Gather of the FP8
  expert weights takes 79 us per layer on B300 and 32 us on B200 (the BF16
  Gather takes 34 us on both). A one-layer graph repeated 8 times shows the
  same: BF16 0.81 ms and FP8 0.45 ms on B200, BF16 0.79 ms and FP8 0.75 ms on
  B300. Also tried at level 3 (B200 / B300 vs BF16): weight-only FP8 (0.73x /
  0.59x), the checkpoint `[in, out]` layout (0.57x / 0.56x) and one gate/up
  stack per layer (1.09x / 0.82x).

NVFP4 (builder level 3 except the q/k/v-only row; "vs FP8" is the FP8 build of
the same projections in the same job):

| Layout | Plan | Decode vs FP8 B200 / B300 | KL mean (x FP8) | Top-1 | Perplexity |
|---|---|---|---|---|---|
| Experts W4A4, all 128 experts in one batched GEMM, then the selected rows | 18.1 GiB | 0.86x / 1.23x | 0.037 (4.3x) | 91.2 % | 11.53 |
| Experts W4A16, packed codes gathered as int32 words and decoded | 23.2 GiB | 0.43x (2-layer probe, B200) | 0.021 (2.4x) | 93.9 % | 11.38 |
| q/k/v W4A4, experts BF16 (family level) | 56.2 GiB | 0.96x / 0.98x | 0.025 (3.6x) | 93.1 % | 11.37 |
| q/k/v W4A4, experts FP8 | 29.3 GiB | 0.98x / 1.02x | 0.030 (3.4x) | 92.8 % | 11.36 |
| Every linear above W4A4 | 17.4 GiB | 0.86x / 1.21x (vs FP8 experts) | 0.057 (5.7x of FP8 experts + q/k/v) | 89.1 % | 11.66 |

Every NVFP4 layout is above 2.2x the FP8 loss, and most are not faster than
FP8. An FP4 constant can only feed a Dequantize, so FP4 GEMMs on the selected
experts alone are not possible; the batched layout reads all 128 experts per
token.

Other results:

- No NaN or Inf in any variant: teacher-forced tokens, the prefill profile,
  128-token runs of random, the highest and one repeated token ids, and greedy
  runs on black, white, noise and 1600x1200 images.
- TensorRT 11.1 (B200): FP8 q/k/v builds and runs, decode 1.05x, KL 0.0069
  (BF16 0.0041); FP8 experts at level 3 1.09x, KL 0.0083.
- The native E2E path was checked with an FP8 q/k/v bundle on B200 and B300:
  48 greedy tokens of an image description are each within 0.5 of the largest
  PyTorch BF16 logit; activation scales x0.1 give 27.5.
- Finding, not changed here: in the multirow prefill profile each token gathers
  its own experts, so a 64-token chunk takes about 175 ms (BF16, both GPUs). The
  batched all-expert NVFP4 layout above takes 8 to 9 ms for the same chunk.
