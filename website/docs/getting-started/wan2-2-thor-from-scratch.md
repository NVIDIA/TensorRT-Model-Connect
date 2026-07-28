---
title: Wan2.2 On A Fresh Jetson Thor
---

This guide starts with a Jetson AGX Thor that has a supported BSP and NVIDIA
driver, but does not have Docker, Model Connect, a Hugging Face cache, or a
Wan2.2 bundle.

The customer container is built directly from NVIDIA's official
`nvcr.io/nvidia/tensorrt:26.07-py3` image and checks that TensorRT is exactly
11.1.0.106 before installing Model Connect. It does not use an internal
TensorRT image, wheel, package registry, or SDK archive.

## Requirements

- Jetson AGX Thor with 128 GB unified memory and a driver compatible with
  CUDA 13.3 containers.
- At least 150 GB of free disk space; 200 GB is recommended when retaining
  the image, checkpoint cache, bundle, and build cache.
- Read access to the TensorRT-Model-Connect GitHub repository.
- Internet access to GitHub, NVIDIA NGC, the PyTorch wheel index, and
  Hugging Face.

## 1. Prepare The Thor Host

If the board has no operating system, start with the official
[Jetson AGX Thor Quick Start Guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html).
The host must provide a working NVIDIA driver:

```bash
nvidia-smi
```

Install Docker and NVIDIA Container Toolkit if they are not already present:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container curl ffmpeg file git

curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
if ! command -v docker >/dev/null 2>&1; then
  sudo sh /tmp/get-docker.sh
fi
sudo systemctl --now enable docker

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the Docker group, then verify GPU access:

```bash
docker run --rm --runtime=nvidia --gpus all \
  nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi
```

Use `sudo docker` instead if local policy does not permit Docker-group
membership.

## 2. Clone Model Connect And Build The Customer Image

Clone with the GitHub authentication method approved by your organization:

```bash
git clone git@github.com:NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect
git checkout main
git pull --ff-only
```

Build the model-owned Thor image:

```bash
REVISION="$(git rev-parse HEAD)"
IMAGE="trtmc-wan22-thor:${REVISION:0:12}"
WAN22_DOCKERFILE="python/tensorrt_model_connect/families/wan2_2_ti2v/Dockerfile.thor"

docker build --pull \
  --build-arg "TRTMC_SOURCE_REVISION=$REVISION" \
  --file "$WAN22_DOCKERFILE" \
  --tag "$IMAGE" \
  .
```

The Dockerfile pins the official ARM64 NGC image by digest. The build stops
unless the Python distribution, C++ headers, runtime library, and SM110
builder resource all match TensorRT 11.1.0.106. Confirm the final image:

```bash
docker image inspect "$IMAGE" \
  --format 'image={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} tensorrt={{index .Config.Labels "com.nvidia.tensorrt.version"}}'

docker run --rm "$IMAGE" version
```

## 3. Create Persistent Storage

```bash
DATA_ROOT="$HOME/trtmc-wan22"
mkdir -p "$DATA_ROOT"/{home,huggingface,work}
```

Reusing this directory avoids downloading the checkpoint again.

## 4. Build The Bundle From Hugging Face

```bash
time docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/data/home \
  --volume "$DATA_ROOT:/data" \
  "$IMAGE" \
  build Wan-AI/Wan2.2-TI2V-5B \
    --model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
    --fp8 \
    --output /data/work/wan22-thor.trtfb
```

This downloads the pinned checkpoint, loads Model Connect's packaged FP8
scales, builds target-specific TensorRT engines, and writes the bundle. It
does not require a local checkpoint, quantization JSON, plugin path, or
backend selector.

## 5. Generate The Full 720p Video

```bash
time docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/data/home \
  --volume "$DATA_ROOT:/data" \
  "$IMAGE" \
  generate-video /data/work/wan22-thor.trtfb \
    --set wan2_2_ti2v.easycache_enabled=true \
    --set wan2_2_ti2v.easycache_threshold=1.0 \
    --set wan2_2_ti2v.easycache_max_consecutive_reuse=4 \
    --set wan2_2_ti2v.late_cfg_enabled=true \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
    --output /data/work/wan22-frames \
    --seed 42
```

The bundle supplies the qualified 1280x704, 121-frame, 50-step, CFG 5, flow
shift 5, and 24 FPS profile. No Wan2.2 environment variable is required.

Verify that generation produced 121 PNG files:

```bash
find "$DATA_ROOT/work/wan22-frames" -maxdepth 1 -name 'frame_*.png' | wc -l
file \
  "$DATA_ROOT/work/wan22-frames/frame_0000.png" \
  "$DATA_ROOT/work/wan22-frames/frame_0060.png" \
  "$DATA_ROOT/work/wan22-frames/frame_0120.png"
cat "$DATA_ROOT/work/wan22-thor.effective_config.json"
```

The representative frames must all report `1280 x 704`. Create a 24 FPS MP4:

```bash
ffmpeg -y -framerate 24 \
  -i "$DATA_ROOT/work/wan22-frames/frame_%04d.png" \
  -c:v libx264 -pix_fmt yuv420p \
  "$DATA_ROOT/work/wan22-thor.mp4"

ffprobe -v error \
  -show_entries stream=codec_name,width,height,nb_frames,avg_frame_rate \
  -of default=noprint_wrappers=1 \
  "$DATA_ROOT/work/wan22-thor.mp4"
```

The expected MP4 is H.264, 1280x704, 121 frames, and 24 FPS.

## Scope

This Dockerfile intentionally targets Thor/SM110. The Wan2.2 implementation is
not Thor-only: Model Connect can build a separate target-specific bundle on
other supported GPUs. TensorRT plans must always be built on the target GPU.
