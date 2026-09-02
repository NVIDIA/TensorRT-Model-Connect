# FoundationPose preprocessed refinement

This native example loads the pinned official FoundationPose refiner and scorer
from one TensorRT Model Connect bundle, refines three object-to-camera pose
hypotheses twice, ranks them, and writes row-major 4x4 transforms.

The integration boundary starts after segmentation and CAD rendering. Each
network input is NHWC `[N,160,160,6]` float32: RGB in `[0,1]`, followed by XYZ
relative to the current candidate translation and divided by half the mesh
diameter. Invalid/background XYZ samples are zero. A production crop callback
must render again from the current poses for every refinement iteration. The
deterministic example reuses static preprocessed crops solely for reproducible
neural inference qualification.

## Pinned artifacts

- Source: `NVlabs/FoundationPose` commit
  `a1b694b83e633c2cb6115b9063d940a687759392`.
- NGC model: `nvidia/isaac/foundationpose:1.0.1_onnx`.
- `refine_model.onnx`: SHA-256
  `dcc695a19c4bcfe5e1d909a22d8f652d8ec8bab1e19bd1544c6b45f2d3595cf7`.
- `score_model.onnx`: SHA-256
  `0bf1026c0db7320ebf9a548ecf0d3c810c8dbd377948630bd3e5af1d49440503`.

Place both ONNX files in one directory and build the FP32 bundle:

```bash
trtmc build /path/to/foundationpose-onnx \
  --precision fp32 -o /tmp/foundationpose.bundle
python3 examples/models/foundationpose/preprocessed_refinement/prepare_synthetic_inputs.py \
  /tmp/foundationpose-inputs
cmake -S examples/models/foundationpose/preprocessed_refinement \
  -B /tmp/foundationpose-example
cmake --build /tmp/foundationpose-example \
  --target trtmc_foundationpose_preprocessed -j
/tmp/foundationpose-example/trtmc_foundationpose_preprocessed \
  /tmp/foundationpose.bundle /tmp/foundationpose-inputs /tmp/refined-poses.f32
```

The initial bundle supports FP32, 1-42 hypotheses per refiner invocation,
up to 252 hypotheses overall (refinement is chunked), 1-10 refinement
iterations, and joint scoring of up to 252 hypotheses. `reset()` clears the
tracked best pose while preserving both TensorRT execution contexts.

## Qualification and limitations

The E2E test compares all refined poses, score logits, and the complete stable
hypothesis ordering with ONNX Runtime, checks rigid-transform invariants, rejects bad
shapes/non-finite values/non-rigid inputs, exercises dynamic batches and reset,
and gates single-hypothesis tracking at at least 10 Hz. The recorded GB300,
TensorRT 11.1 qualification is under `qualification/`.

This example does not perform object segmentation, mesh loading, CAD rendering,
camera calibration, collision checking, motion planning, or actuator control.
Its pose output is not a physical-robot safety guarantee. Applications must
validate calibration, units, coordinate frames, workspace limits, and failure
handling before consuming a pose in a control system.
