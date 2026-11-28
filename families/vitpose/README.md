# ViTPose

This family builds `VitPoseForPoseEstimation` checkpoints with the simple decoder head, validated on
[`usyd-community/vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple) (ViT-B, 85.9 M
float32 parameters, the 17 COCO body keypoints), and serves them through the `image_text_to_points` Task
(`IModel` + `IImageTextToPoints`). The CLI command is `detect` with `--prompt person`.

## Task contract

- **Input**: an RGB image (8-bit, or float32 in [0, 1], which is rounded to 8 bits) and the query `person`; an
  empty query means the same. ViTPose locates the body keypoints of one person, so any other query is rejected.
- **Person box**: the optional config field `person_box` is `[x_min, y_min, x_max, y_max]` in original-image pixels,
  inside the image and at least one pixel wide and tall (CLI: `--set 'person_box=[x0, y0, x1, y1]'`). Without it the
  whole image is the person's box, the single-person crop. Several people need one request per box, for example
  from a person detector.
- **Output**: one point per located keypoint, in keypoint order, labelled with the checkpoint's `id2label` name
  (`Nose`, `L_Eye`, ... `R_Ankle`), in original-image pixels with pixel centres at integer coordinates. The
  confidence is the keypoint's heatmap maximum, the reference processor's `scores` (not a calibrated probability). A
  keypoint whose heatmap has no positive value is left out; the reference marks it -1. `raw_response` is empty.
  Heatmaps, the skeleton and multi-person grouping are not part of this Task.

## Pipeline

- **Crop** (native, `runtime/image_ops.cpp`): the reference processor's unbiased box crop. The box is grown to the
  model's 3:4 aspect and padded by 1.25, the 256 x 192 crop is sampled bilinearly from the 8-bit image with zeros
  outside it and rounded to 8 bits, then normalized with the ImageNet statistics. It reproduces
  `VitPoseImageProcessorFast` (which warps with `scipy.ndimage.affine_transform`) bit for bit.
- **Engine** (`model.py`): one strongly typed float32 engine, TF32 off, from pixel values to 17 heatmaps of
  64 x 48. The patch convolution has padding 2; there is no class token, and row 0 of the position table is added to
  every patch; 12 pre-norm layers with the exact GELU; the final layer norm; ReLU, bilinear upsampling by 4 and a
  3 x 3 convolution. The checkpoint is float32 and the family builds `--precision fp32` only.
- **Keypoints** (native): the first maximum of each heatmap, refined by DARK (Gaussian smoothing with sigma 0.8 and
  11 taps, then one Newton step on the log heatmap) and mapped back through the crop, as
  `post_process_pose_estimation` does in float32 and float64. A singular Hessian, where the reference raises, keeps
  the integer peak.

## Validation

- `tests/test_image_ops.py` (CPU): the native crop against the reference processor on ten image and box shapes
  (bit for bit), exact half-way samples, and the keypoint decoding against `post_process_pose_estimation`.
- `tests/cpp/test_task_contract.cpp`: the Task binding, query, person box, image and engine-output checks, and the
  strict reading of `runtime.json`.
- `tests/test_model.py` (GPU): the engine's heatmaps against `VitPoseForPoseEstimation` in float32, within 1e-4
  (measured 1.6e-6 on a B300; a tanh GELU or a missing position row moves them by 5.6e-4 or more).
- `tests/test_e2e.py` (GPU, case `vitpose-base-simple`): the CLI and the C and C++ SDK consumers against the
  Transformers pipeline on `tests/data/person.png`, for the whole image and for a person box: keypoints within
  0.005 px and scores within 1e-4 (measured 6.2e-5 px and 6.6e-7 on a B300; single injected defects move them by
  0.0126 px or more), and the keypoints must fit the shown scene (a person standing and facing the camera).

`tests/data/person.png` is a project-created image (see `ASSET_LICENSES.md`).
