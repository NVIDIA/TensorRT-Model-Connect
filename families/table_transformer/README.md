# Table Transformer

This family implements `microsoft/table-transformer-structure-recognition` at
revision `f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343`. It constructs a
ResNet-18 and pre-normalized encoder/decoder directly with TensorRT's strongly
typed Network API. The checkpoint format is `model.safetensors`.

The semantic Task is `image_to_boxes`, exposed by the CLI's `detect` command
and the public C and C++ SDKs. Results contain confidence scores, numeric class
IDs, and XYXY coordinates in the original image's pixels. The six class IDs
are table (0), column (1), row (2), column header (3), projected row header (4),
and spanning cell (5). The family filters scores at 0.5 and preserves overlapping
structures without NMS. OCR, cell text, and HTML reconstruction are separate
downstream operations.

## Build and run

Use a prepared development environment as described in
[Build from Source](../../website/docs/getting-started/source-build.md), and
install this family's optional dependencies:

```bash
python -m pip install -r families/table_transformer/requirements.txt
export PYTHONPATH=core/builder:apps/benchmark:.
python -m tensorrt_model_connect build \
  microsoft/table-transformer-structure-recognition \
  --revision f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343 \
  --precision fp16 --image-height 1000 --image-width 1000 \
  --output table-transformer-structure-recognition.bundle
```

A local directory containing that revision's `config.json`,
`preprocessor_config.json`, and `model.safetensors` can replace the model ID.
The builder resolves ownership from checkpoint metadata.

Build the native consumers in a configured native build directory:

```bash
cmake --build build --parallel 8 --target \
  trtmc trtmc_backend_trt trtmc_model_table_transformer \
  test_table_transformer_task_contract test_table_transformer_image_preprocess \
  test_table_transformer_sdk_c test_table_transformer_sdk_cpp
build/trtmc detect table-transformer-structure-recognition.bundle \
  --runtime-root "$PWD/build" \
  --image families/table_transformer/tests/data/structure-table.jpeg
```

FP32 and FP16 are implemented for one image on one GPU. In FP16 builds,
convolutions and input projection use FP16; the encoder, decoder, attention,
and output heads retain FP32 arithmetic because narrow structure boxes amplify
half-precision perturbations. TF32 is disabled for the reference-sensitive path. The processor preserves
aspect ratio, targets an 800-pixel shortest edge, caps the longest edge at
1000 pixels, and applies the checkpoint's ImageNet normalization. RGB floats in [0, 1]
are restored to bytes before the official PIL-compatible bilinear resize. Engine input
dimensions are dynamic; the default capacity is 1000 by 1000. The runtime
rejects images whose resized dimensions exceed a caller's smaller build bound.
Batching, distributed execution, quantization, BF16, and other backends are
explicitly rejected.

## Validate

The two manifests pin the same checkpoint for FP32 and FP16. The direct E2E
builds a bundle, runs the CLI and public SDK consumers, and compares against
the official `TableTransformerForObjectDetection` and `DetrImageProcessor`.
The official reference runs on CPU; the native TensorRT path runs on CUDA.
Landscape, portrait, and square images exercise dynamic geometry. Each case
requires the same detection count and labels, matched IoU of at least 0.95,
and score error of at most 0.01. The test writes a `parity.json` receipt beside
the produced bundle.

```bash
export TRTMC_BINARY="$PWD/build/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build"
export TRTMC_NATIVE_BUILD_DIR="$PWD/build"
export TRTMC_TABLE_TRANSFORMER_MODEL_DIR=/path/to/pinned/checkpoint
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/table_transformer/tests --e2e-model table_transformer -q
ctest --test-dir build -R '^table_transformer_' --output-on-failure
```

The fixture is a project-created scientific table with invented values.
These parity checks establish the declared test cases; they do not measure
PubTables-1M dataset accuracy or qualify other checkpoints or GPU targets.
