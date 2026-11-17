# BEVFusion BEV map segmentation

This family implements the BEV feature path selected by upstream
[`fusion-bev256d2-lss.yaml`](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/seg/fusion-bev256d2-lss.yaml):
ConvFuser, SECOND, SECONDFPN, the BEV grid transform, and the map head.
TensorRT builds the graph directly and chooses its GPU execution. No upstream
OpenMMLab installation, ONNX export, or custom GPU operators are needed to build it.

The public `spatial_features_to_class_scores` Task accepts two named, contiguous
host FP32 CHW feature maps, one example per call:

| Input | Shape | Coordinate scope |
| --- | --- | --- |
| `camera_bev` | `[80,128,128]` | source camera BEV features |
| `lidar_bev` | `[256,128,128]` | source LiDAR BEV features |

The engine adds a batch dimension of one. Input BEV scope is
`[-51.2,51.2)` meters on both axes, at 0.8-meter resolution. The result contains
FP32 sigmoid probabilities `[6,200,200]`, with a 0.5-meter grid over
`[-50,50)` meters. Axis order follows the upstream grid transform. Class order is
`drivable_area`, `ped_crossing`, `walkway`, `stop_line`, `carpark_area`, `divider`.
Classes may overlap; scores retain independent sigmoid probabilities.

The path starts at BEV feature tensors. Raw camera encoding/view transformation,
LiDAR voxelization/sparse encoding, and dataset-level nuScenes evaluation are
outside this family. FP32 and FP16 are supported; inputs/output and BatchNorm
statistics remain FP32. BF16, quantization, other configurations, batching and
parallel sharding are unsupported.

## Checkpoint and build

Use the official segmentation checkpoint linked from the
[upstream model zoo](https://github.com/mit-han-lab/bevfusion/tree/326653dc06e0938edf1aae7d01efcd158ba83de5#pretrained-models).
Its tested SHA-256 is
`4c049b596c23f81227ae139cac09914cc3a45474f4861134817233236d37c9ac`.
Save it as `artifacts/bevfusion/model/bevfusion-seg.pth`. The builder accepts
compatible checkpoint weights and records architecture/precision metadata in `runtime.json`;
the retained E2E cases qualify this exact official checkpoint.

From the repository root, in a working TRT MC development environment:

```bash
mkdir -p artifacts/bevfusion/model
curl -fL 'https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1' \
  -o artifacts/bevfusion/model/bevfusion-seg.pth
PYTHONPATH=core/builder:. python - <<'PY'
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build
for precision in ("fp32", "fp16"):
    build(BuildRequest(
        model_dir=Path("artifacts/bevfusion/model"),
        output_path=Path(f"artifacts/bevfusion/bevfusion-bev-{precision}.bundle"),
        family="bevfusion", task="spatial_features_to_class_scores", precision=precision,
    ))
PY
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTRTMC_BUILD_TESTS=ON
cmake --build build --target trtmc trtmc_backend_trt test_bevfusion_pipeline -j8
```

The public C++ call takes existing host feature buffers:

```cpp
#include <trtmc/features.hpp>
auto model = trtmc::Model::load("bevfusion.bundle", load_options);
trtmc::SpatialFeaturesToClassScoresRequest request{{
    {"camera_bev", {camera.data(), camera.size()}, 80, 128, 128},
    {"lidar_bev", {lidar.data(), lidar.size()}, 256, 128, 128},
}};
auto result = model.task<trtmc::SpatialFeaturesToClassScores>().run(request);
auto scores = result.view(); // Owned result; scores.values is contiguous [6,200,200].
```

## Validate and benchmark

Prepare the declared reference files at their recorded revision. The reference adapter
executes the upstream class bodies, replacing only registry and dependency
plumbing with the corresponding PyTorch constructors. It loads every BEV-stage
weight strictly. The test compares all output probabilities for seeds 0, 1 and 2
against FP32 PyTorch with TF32 disabled: FP32 atol/rtol 0.0002; FP16 0.02.
The native helper exercises bundle loading and the public Task, including
reversed input order, repeated calls, class identity and result ownership.

```bash
PYTHONPATH=core/builder:. python -m families.bevfusion.tests.reference \
  --prepare artifacts/bevfusion/model
TRTMC_BEVFUSION_MODEL_DIR="$PWD/artifacts/bevfusion/model" \
TRTMC_RUNTIME_ROOT="$PWD/build" TRTMC_NATIVE_BUILD_DIR="$PWD/build" PYTHONPATH=core/builder:. \
python -m pytest families/bevfusion/tests --e2e-model bevfusion -q
```

For a single TensorRT engine forward pass, the same qualification helper has an
internal `--engine-only` option. It reads the exact qualified bundle, uploads the
two feature tensors once, binds a resident GPU output and captures exactly one
TensorRT `enqueueV3`. CUDA events measure graph replay after 20 warmups, in three
rounds of 100 calls. Setup, input/output transfers, public Task validation and
output ownership/copying are outside this boundary. The JSON records median,
p95, each round's median, versions and the timing boundary. Output probabilities
are copied back after timing; the retained E2E requires them to match the public
Task bit for bit.

```bash
PYTHONPATH=core/builder:. python -m families.bevfusion.tests.inputs \
  --output artifacts/bevfusion/inputs --seed 0
build/test_bevfusion_sdk_cpp \
  artifacts/bevfusion/bevfusion-bev-fp16.bundle build \
  artifacts/bevfusion/inputs/bevfusion/tests/data/camera.f32 \
  artifacts/bevfusion/inputs/bevfusion/tests/data/lidar.f32 \
  artifacts/bevfusion/engine-fp16.json --engine-only
```

Read `median_ms` and `p95_ms` from the output JSON. Use the FP32 bundle instead
to measure FP32 execution with the same inputs.

The existing source-derived benchmark runner uses randomly initialized weights and a transposed 1x1 first
FPN branch. This family follows the selected upstream config's ordinary 1x1
convolution and official weights. Its timings are a validated execution baseline;
they do not establish cross-checkpoint or cross-hardware performance parity.
