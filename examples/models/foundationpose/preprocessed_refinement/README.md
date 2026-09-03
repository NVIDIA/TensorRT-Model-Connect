# FoundationPose preprocessed refinement

This example refines and ranks three object-to-camera pose hypotheses with the
FoundationPose refiner and scorer in one TensorRT Model Connect bundle. It takes
preprocessed RGB+XYZ crops and writes row-major 4x4 transforms.

## Build and run

Run these commands from the repository root. The `trtmc` command must be
installed and available on `PATH`.

Set `MODEL_DIR` to the pinned `nvidia/isaac/foundationpose:1.0.1_onnx` weight
directory. It contains `refine_model.onnx` and `score_model.onnx`; their exact
digests are in the family
[`MODEL.toml`](../../../../python/tensorrt_model_connect/families/foundationpose/MODEL.toml).
The native builder reads only their initializer tensors—it does not parse or
execute either ONNX graph. A different weight format is not currently supported.

```bash
MODEL_DIR=/path/to/foundationpose-weights
BUNDLE=/tmp/foundationpose.bundle
INPUTS=/tmp/foundationpose-inputs
EXAMPLE_BUILD=/tmp/foundationpose-example

trtmc build "$MODEL_DIR" --precision fp32 -o "$BUNDLE"
python3 examples/models/foundationpose/preprocessed_refinement/prepare_synthetic_inputs.py "$INPUTS"

cmake -S examples/models/foundationpose/preprocessed_refinement -B "$EXAMPLE_BUILD"
cmake --build "$EXAMPLE_BUILD" --target trtmc_foundationpose_preprocessed -j

"$EXAMPLE_BUILD"/trtmc_foundationpose_preprocessed \
  "$BUNDLE" "$INPUTS" /tmp/refined-poses.f32
```

The executable prints the best hypothesis, its score, and whether every output
pose is rigid. All refined poses are written to `/tmp/refined-poses.f32` as
FP32 matrices.

## Input contract

Inputs are FP32 NHWC `[N,160,160,6]`: RGB in `[0,1]`, followed by XYZ relative
to the candidate translation and normalized by half the mesh diameter.
Invalid/background XYZ values are zero. The bundle supports 1-252 hypotheses,
1-10 refinement iterations, and FP32 only.

## Scope

The example uses fixed synthetic crops for reproducibility. A production
application must regenerate rendered crops from the current pose after every
refinement iteration. Segmentation, mesh loading, CAD rendering, calibration,
collision checking, motion planning, and robot-safety validation are outside
this example.
