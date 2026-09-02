# LeRobot ACT recorded-control example

This example runs a native TensorRT Action Chunking Transformer (ACT) policy on an immutable recorded ALOHA observation and emits its complete 100-action chunk at 50 Hz. It is a reproducible software qualification, not a physical-robot safety qualification.

## Qualified target

- Hardware: one NVIDIA GB300 GPU (283136 MiB reported device memory); no physical robot or live sensor was attached
- Software: driver `580.105.08`, CUDA toolkit `13.3.73`, TensorRT `11.1.0.106`, container `trtmc-dev-gb300:manylinux_2_39`
- Policy: `lerobot/act_aloha_sim_transfer_cube_human`
- Policy revision: `ba73b2766f1371cdc133ca4efb97eb090d744625`
- `model.safetensors` SHA-256: `42772891cb6eba1e7bc36ad8e12c0fa0723c61f036fa235c725ce6026e6e81df`
- Training implementation: LeRobot revision `3c0a209f9fac4d2a57617e686a7f2a2309144ba2`
- Recorded dataset: `lerobot/aloha_sim_transfer_cube_human` revision `6a43d500f101255823a9d2b9dc244eeb01a2cd31`, v3, episode 0 frame 0
- Simulator/task: `gym-aloha==0.1.1`, `AlohaTransferCube-v0`
- Precision: FP32; TensorFloat-32 is disabled for the qualified graph

The qualified sensor input is recorded data only: one RGB top-camera frame in HWC `[480,640,3]`, represented as finite floats in `[0,1]`, plus a finite 14-value joint state. Camera intrinsics, extrinsics, exposure, synchronization, and calibration are inherited from the pinned dataset and are not generalized by this qualification. The output is an unnormalized `[100,14]` action chunk. The 14 values are, in order, left waist, shoulder, elbow, forearm roll, wrist angle, wrist rotate, gripper, followed by the same seven right-arm values.

The graph owns image/state mean-standard-deviation normalization and action unnormalization. It uses one observation step, has no temporal ensemble, and queues all 100 predicted steps. At 50 Hz, one chunk covers a two-second control horizon. Outputs are reported against the dataset's per-joint training extrema; they are never silently clipped.

## Build and launch

Build the bundle with the pinned policy revision:

```bash
trtmc build lerobot/act_aloha_sim_transfer_cube_human \
  --model-revision ba73b2766f1371cdc133ca4efb97eb090d744625 \
  --precision fp32 \
  -o /tmp/act-aloha-sim-transfer-cube.bundle
```

Materialize the qualified recorded observation. The helper verifies the pinned Parquet artifact and records the downloaded video digest:

```bash
python tests/e2e/models/lerobot_act/prepare_recorded_observation.py \
  --output /tmp/lerobot-act-replay --episode-index 0 --frame-index 0
```

The preparation command verifies and copies the repository's qualified replay fixture, whose
source dataset revision and source-file SHA-256 digests are recorded alongside it. It therefore
works without network access and fails closed if the packaged image or state digest changes.

The main CLI runs parity-ready inference, ten timed chunks, and a real-time 50 Hz emission loop. Its JSON includes startup time, chunk p50/p95 latency and throughput, peak resident memory, effective control rate, p99 interval jitter, deadline misses, and the training-bound result:

```bash
trtmc act /tmp/act-aloha-sim-transfer-cube.bundle \
  --image /tmp/lerobot-act-replay/observation.images.top.png \
  --state /tmp/lerobot-act-replay/observation.state.f32 \
  --output /tmp/lerobot-act-actions.f32 \
  --control-hz 50 --warmup 2 --benchmark 10
```

To compile the direct C++ integration example:

```bash
cmake -S examples/models/lerobot_act/recorded_control \
  -B /tmp/lerobot-act-example -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/lerobot-act-example \
  --target trtmc_lerobot_act_recorded_control -j
```

Then run it with the model-plugin directory produced by that build:

```bash
/tmp/lerobot-act-example/trtmc_lerobot_act_recorded_control \
  /tmp/act-aloha-sim-transfer-cube.bundle \
  --image /tmp/lerobot-act-replay/observation.images.top.png \
  --state /tmp/lerobot-act-replay/observation.state.f32 \
  --backend-dir /tmp/lerobot-act-example/trtmc \
  --model-plugin-dir /tmp/lerobot-act-example/models/lerobot_act \
  --control-hz 50
```

The full E2E test additionally compares every unnormalized action against the exact pinned LeRobot PyTorch implementation:

```bash
pytest -q tests/e2e/models/lerobot_act/test_lerobot_act_e2e.py \
  --trtmc-binary /path/to/trtmc \
  --engine-dir /path/to/bundles \
  --model-plugin-dir /path/to/model/plugins
```

The checked-in [GB300 qualification record](qualification/gb300-trt11.1-fp32.json) captures the exact software/hardware fingerprint, build/startup/memory costs, numerical parity, chunk throughput, and measured control-loop jitter for this contract.

## Reset, failures, and limits

Call `IPipeline::reset()` at every environment or episode reset. Reset discards the queued chunk and resets the TensorRT execution context; the next `act()` call starts a new chunk. Missing observations, wrong shapes, non-finite state values, and image samples outside `[0,1]` fail closed with an input error. Non-finite actions or actions outside the recorded training extrema are surfaced to the caller and are not clipped.

This qualification covers one simulation-trained ACT checkpoint, one top camera, the exact ALOHA joint ordering above, batch size one, FP32, recorded replay, and `AlohaTransferCube-v0`. It does not cover other sensors, camera calibration, robot geometries, checkpoints, mixed precision, temporal ensembling, dropped observations, networked control, actuator communication, collision avoidance, force/torque limits, emergency stopping, or recovery after physical faults.

No physical robot safety has been established. Do not connect these actions directly to hardware. A deployment owner must independently implement and validate workspace limits, velocity/acceleration/effort limits, collision avoidance, watchdogs, interlocks, emergency stops, human exclusion zones, sensor-health checks, and a safe fallback controller.
