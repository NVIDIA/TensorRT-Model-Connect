# FoundationPose preprocessed refinement

This example refines and ranks three object-to-camera pose hypotheses with the
FoundationPose refiner and scorer in one TensorRT Model Connect bundle. It takes
preprocessed RGB+XYZ crops and writes row-major 4x4 transforms.

## Build and run

Run these commands from the repository root. The `trtmc` command must be
installed and available on `PATH`.

Set `MODEL_DIR` to the `nvidia/isaac/foundationpose:1.0.1_onnx` weight
directory. It must contain `refine_model.onnx` and `score_model.onnx`.
The builder validates their graph I/O, nodes, operations, tensor names, and
shapes, then authors the TensorRT layers directly. It does not use the TensorRT
ONNX parser, and the native runtime does not execute ONNX.

```bash
MODEL_DIR=/path/to/foundationpose-weights
BUNDLE=/tmp/foundationpose.bundle
INPUTS=/tmp/foundationpose-inputs
EXAMPLE_BUILD=/tmp/foundationpose-example
RUNTIME_ROOT=/path/to/tensorrt_model_connect/bin

python -m tensorrt_model_connect build "$MODEL_DIR" \
  --task pose_hypothesis_refinement --precision fp16 --output "$BUNDLE"
python3 examples/models/foundationpose/preprocessed_refinement/prepare_synthetic_inputs.py "$INPUTS"

cmake -S examples/models/foundationpose/preprocessed_refinement -B "$EXAMPLE_BUILD"
cmake --build "$EXAMPLE_BUILD" --target trtmc_foundationpose_preprocessed -j

"$EXAMPLE_BUILD"/trtmc_foundationpose_preprocessed \
  "$BUNDLE" "$RUNTIME_ROOT" "$INPUTS" /tmp/refined-poses.f32
```

The executable prints the best hypothesis, its score, and whether every output
pose is rigid. All refined poses are written to `/tmp/refined-poses.f32` as
FP32 matrices.

## Input contract

Inputs are FP32 NHWC `[N,160,160,6]`: RGB in `[0,1]`, followed by XYZ relative
to the candidate translation and normalized by half the mesh diameter.
Invalid/background XYZ values are zero. The bundle supports 1-252 hypotheses
and 1-10 refinement iterations. The mixed-FP16 build keeps the
numerically sensitive scorer cross-attention in FP32. Pass `--precision fp32`
to build the entire model in FP32 instead.

## Scope

The example uses fixed synthetic crops for reproducibility. A production
application must regenerate rendered crops from the current pose after every
refinement iteration. Segmentation, mesh loading, CAD rendering, calibration,
collision checking, motion planning, and robot-safety validation are outside
this example.
