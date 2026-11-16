# PP-DocLayoutV3

This family builds `PaddlePaddle/PP-DocLayoutV3_safetensors` as one native
TensorRT engine for `document_layout`, with batch one, FP32, and an 800 × 800
network input. The HGNetV2 backbone, hybrid encoder, mask-enhanced references,
deformable decoder, mask head, and pairwise reading-order head belong to this
family. Quantization and other precisions are rejected.

The native `IDocumentLayout::analyze_layout` input is HWC RGB floats in [0, 1].
The result contains regions in logical reading order, each with a class ID,
score, box, and contour polygon in original-image pixels. Region labels are
the checkpoint's `id2label` mapping. The runtime performs bicubic
resize and mask contour extraction; it does not execute Python.

The CI and development Docker images include the native OpenCV core and imgproc
development packages. For a host build, install them before CMake configuration,
for example `libopencv-core-dev libopencv-imgproc-dev` on Ubuntu.
The family target is omitted when these optional dependencies are absent.
The CI image also installs `opencv-python-headless` inside its Python virtual
environment for the Transformers reference postprocessor. The native development
packages alone do not provide that environment's `cv2` module.

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  PaddlePaddle/PP-DocLayoutV3_safetensors --precision fp32 \
  --output pp-doclayout-v3.bundle
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_pp_doclayout_v3
build/trtmc layout pp-doclayout-v3.bundle --runtime-root build --image document.png
```

The direct E2E test rebuilds the bundle and compares both the supplied document
and its horizontal mirror with Transformers 5.2.0. It checks class and reading
order, detection counts, score error, box IoU, and rasterized polygon IoU.
Install `tests/requirements.txt`, then set `TRTMC_BINARY` and `TRTMC_RUNTIME_ROOT`
before running `pytest families/pp_doclayout_v3/tests --e2e-model pp_doclayout_v3`.
Provision the exact checkpoint revision pinned in
`tests/manifests/pp-doclayout-v3.json` into the Hugging Face Hub cache first;
the test resolves that snapshot with `local_files_only=True`. Alternatively,
set `TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR` to an existing local checkpoint directory
containing `config.json`.

The manifest's `external_files` declares the official model-card demonstration
document for CI's online asset-provisioning stage. Internal CI supplies that
asset directory through `TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR`; the test keeps it
separate from the manifest-pinned Hub checkpoint. The image is absent from the
Hub snapshot and is downloaded at setup time, not redistributed in this repo:
`https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg`.
The original file's SHA-256 is
`785b7d19f158dcb636342dd3378ed3a4cddb7333d2d71688f0baa5c25a88ad51`.
Default image resolution uses the manifest's declared path in CI's external-file
directory, the local checkpoint directory, and `/opt/trtmc-ci`, checking the
selected image with Pillow before use. For a local run,
`TRTMC_PP_DOCLAYOUT_V3_TEST_IMAGE` can select another image
path. No network download occurs during E2E. Invalid explicit paths and corrupt
default images fail without silently selecting another input.
