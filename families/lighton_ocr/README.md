# LightOnOCR

`lighton_ocr` owns `lightonai/LightOnOCR-2-1B`, including the checkpoint's
legacy `mistral3` metadata and `LightOnOCRForConditionalGeneration`
architecture. The pinned checkpoint revision is
`be7c41313c30e940fe5571de59e5fabc779e00c8`.

The initial path is image-only OCR through `images_text_to_text`: one user
message, one RGB image, and empty text. It builds the complete 24-layer Pixtral
encoder, spatial merger and projector, and the complete 28-layer Qwen3 decoder
with tied embeddings. The original BF16 checkpoint values are expanded to
FP32; TensorRT owns GPU lowering and tactic selection. Native inference uses
the family DSO and the public Task/backend contracts without Python inference.

## Build

Prepare a matching native development environment using the repository's
[source-build instructions](../../website/docs/getting-started/source-build.md).
Install this family's optional dependencies in the selected build/reference
environment:

```bash
python -m pip install -r families/lighton_ocr/requirements.txt
```

From the checkout root:

```bash
PYTHONPATH=core/builder:. python -m tensorrt_model_connect build \
  lightonai/LightOnOCR-2-1B \
  --revision be7c41313c30e940fe5571de59e5fabc779e00c8 \
  --precision fp32 --max-sequence-length 4096 \
  --output /tmp/lightonocr-2-1b-fp32.bundle

cmake --build "$BUILD" --parallel --target \
  trtmc trtmc_backend_trt trtmc_model_lighton_ocr
```

The default sequence capacity is 4,096 tokens, shared by the expanded image
prompt and output. The bundle stores `vision.plan`, `decoder.plan`, the FP32
embedding table, tokenizer assets, and family-owned runtime metadata.
The decoder maintains compact key/value caches during greedy generation.
The default output budget is 1,024 tokens, including EOS. The runtime checks
that each image prompt and requested output budget fit the bundle capacity.

## Native OCR

The CLI passes one image and an explicitly empty text part to the public
`trtmc::ImagesTextToText` Task:

```bash
"$BUILD/trtmc" run /tmp/lightonocr-2-1b-fp32.bundle \
  --runtime-root "$BUILD" --image document.png --prompt '' \
  --max-new-tokens 1024
```

Only deterministic greedy generation (`temperature=0`) is accepted. Each
request starts with fresh decoder cache state. Multiple images, nonempty text,
tools, other message roles, and conversation history are rejected.

The official reference is Transformers 5.2.0
`LightOnOcrForConditionalGeneration` and `LightOnOcrProcessor`, explicitly
loaded with `use_fast=False`. This selects the Pillow-based
`PixtralImageProcessor`, even though the original metadata names a fast
processor. Native preprocessing implements that selected RGB bicubic byte
resize, patch-aligned dimensions, rescaling, normalization, and axial image
positions. The checkpoint maximum image edge is 1,540 pixels; patches are
14 × 14 with a spatial merge factor of two. Odd patch grids use the
checkpoint's floor-crop merge behavior.

## Validation

The [selected manifest](tests/manifests/lightonocr-2-1b-fp32.json) declares a
672 × 448 synthetic invoice, the original image-only prompt, FP32 eager
attention with caching, and a 1,024-token greedy budget. Its
[fixed thresholds](tests/thresholds/lightonocr-2-1b-fp32-invoice.json) require
native/reference preprocessing and image-feature parity, two complete
teacher-forced vocabulary-logit comparisons, and exact nonempty generated
text/token IDs ending in EOS. The independent reference imports no family
graph implementation.

Use separate reference and build interpreters when their dependencies differ.
The reference needs Transformers 5.2.0 and compatible PyTorch/torchvision,
which the documented development environment provides. Run the declared
checkpoint E2E with the matching native targets and a fresh artifact directory:

```bash
PYTHONPATH=core/builder:apps/benchmark:. \
TRTMC_LIGHTON_OCR_MODEL_DIR=/absolute/pinned-checkpoint \
TRTMC_LIGHTON_OCR_REFERENCE_PYTHON=/absolute/reference-environment/bin/python \
TRTMC_BINARY="$BUILD/trtmc" \
TRTMC_RUNTIME_ROOT="$BUILD" \
TRTMC_NATIVE_BUILD_DIR="$BUILD" \
TRTMC_E2E_ARTIFACT_DIR=/absolute/new-evidence-directory \
python -m pytest families/lighton_ocr/tests --e2e-model lighton_ocr -q

PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/lighton_ocr/tests -m 'not gpu and not trt and not e2e' -q
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
git diff --check
```

Qualification applies to the executed checkpoint, fixture, FP32 profile,
single GPU, batch one, TP1/CP1, and declared token capacity. Other documents
and languages, larger contexts, FP16/BF16, quantization, batching, distributed
execution, streaming, and corpus-level OCR accuracy require their own evidence.
PDF rendering is outside the native Task; provide an RGB page image.
