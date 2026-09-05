# LeRobot ACT recorded-control example

This example runs a native TensorRT Action Chunking Transformer (ACT) policy on an immutable recorded ALOHA observation and emits its complete 100-action chunk at 50 Hz. It is a reproducible software qualification, not a physical-robot safety qualification.

## Qualified target

- Hardware: one NVIDIA GB300 GPU (283136 MiB reported device memory); no physical robot or live sensor was attached
- Software: driver `580.105.08`, CUDA toolkit `13.3.73`, TensorRT `11.1.0.106`, container `trtmc-dev-gb300:manylinux_2_39`
- Policy: `lerobot/act_aloha_sim_transfer_cube_human`
- Policy revision: `ba73b2766f1371cdc133ca4efb97eb090d744625`
- Training implementation: LeRobot revision `3c0a209f9fac4d2a57617e686a7f2a2309144ba2`
- Recorded dataset: `lerobot/aloha_sim_transfer_cube_human` revision `6a43d500f101255823a9d2b9dc244eeb01a2cd31`, v3, episode 0 frame 0
- Simulator/task: `gym-aloha==0.1.1`, `AlohaTransferCube-v0`
- Precision: FP32; TensorFloat-32 is disabled for the qualified graph

The qualified sensor input is recorded data only: one RGB top-camera frame in HWC `[480,640,3]`, represented as finite floats in `[0,1]`, plus a finite 14-value joint state. Camera intrinsics, extrinsics, exposure, synchronization, and calibration are inherited from the declared dataset revision and are not generalized by this qualification. The output is an unnormalized `[100,14]` action chunk. The 14 values are, in order, left waist, shoulder, elbow, forearm roll, wrist angle, wrist rotate, gripper, followed by the same seven right-arm values.

The graph owns image/state mean-standard-deviation normalization and action unnormalization. It uses one observation step, has no temporal ensemble, and queues all 100 predicted steps. At 50 Hz, one chunk covers a two-second control horizon. Outputs are reported against the dataset's per-joint training extrema; they are never silently clipped.

## Build and launch

Build the bundle with the policy revision:

```bash
python -m tensorrt_model_connect build lerobot/act_aloha_sim_transfer_cube_human \
  --revision ba73b2766f1371cdc133ca4efb97eb090d744625 \
  --precision fp32 \
  -o /tmp/act-aloha-sim-transfer-cube.bundle
```

Materialize the qualified recorded observation from the packaged fixture or exact dataset revision:

```bash
python families/lerobot_act/tests/prepare_recorded_observation.py \
  --output /tmp/lerobot-act-replay --episode-index 0 --frame-index 0
```

The preparation command validates and copies the repository's recorded replay fixture. Its exact
dataset revision, episode, frame, and tensor shapes are recorded alongside it, so it works without
network access.

The main CLI runs one action-chunk inference and writes the complete float32 action tensor. Its
JSON includes the action shape, training-bound result, and family-measured inference time:

```bash
trtmc control /tmp/act-aloha-sim-transfer-cube.bundle \
  --runtime-root /path/to/runtime \
  --image /tmp/lerobot-act-replay/observation.images.top.png \
  --state /tmp/lerobot-act-replay/observation.state.f32 \
  --output /tmp/lerobot-act-actions.f32
```

To compile the direct C++ integration example:

```bash
cmake -S examples/models/lerobot_act/recorded_control \
  -B /tmp/lerobot-act-example -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/lerobot-act-example \
  --target trtmc_lerobot_act_recorded_control -j
```

Then run it with the exact runtime directory produced by that build:

```bash
/tmp/lerobot-act-example/trtmc_lerobot_act_recorded_control \
  /tmp/act-aloha-sim-transfer-cube.bundle \
  --image /tmp/lerobot-act-replay/observation.images.top.png \
  --state /tmp/lerobot-act-replay/observation.state.f32 \
  --runtime-root /tmp/lerobot-act-example/trtmc \
  --control-hz 50
```

The full E2E test additionally compares every unnormalized action against the exact-revision LeRobot PyTorch implementation:

```bash
TRTMC_BINARY=/path/to/trtmc \
TRTMC_RUNTIME_ROOT=/path/to/runtime \
TRTMC_LEROBOT_SOURCE_DIR=/path/to/lerobot \
pytest -q families/lerobot_act/tests/test_e2e.py \
  --e2e-testcase act-aloha-sim-transfer-cube-recorded-episode-0-frame-0
```

The checked-in [GB300 qualification record](qualification/gb300-trt11.1-fp32.json) captures the software/hardware context, build/startup/memory costs, numerical parity, chunk throughput, and measured control-loop jitter for this contract.

## Reset, failures, and limits

Call `IRobotControl::reset()` at every environment or episode reset. Reset discards the queued chunk and resets the TensorRT execution context; the next `act()` call starts a new chunk. Missing observations, wrong shapes, non-finite state values, and image samples outside `[0,1]` fail closed with an input error. Non-finite actions or actions outside the recorded training extrema are surfaced to the caller and are not clipped.

This qualification covers one simulation-trained ACT checkpoint, one top camera, the exact ALOHA joint ordering above, batch size one, FP32, recorded replay, and `AlohaTransferCube-v0`. It does not cover other sensors, camera calibration, robot geometries, checkpoints, mixed precision, temporal ensembling, dropped observations, networked control, actuator communication, collision avoidance, force/torque limits, emergency stopping, or recovery after physical faults.

No physical robot safety has been established. Do not connect these actions directly to hardware. A deployment owner must independently implement and validate workspace limits, velocity/acceleration/effort limits, collision avoidance, watchdogs, interlocks, emergency stops, human exclusion zones, sensor-health checks, and a safe fallback controller.
