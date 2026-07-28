---
title: Wan2.2 On A Fresh Jetson Thor
---

This guide starts with a Jetson AGX Thor that has a supported BSP and NVIDIA
driver, but does not have Docker, TensorRT-Model-Connect, a Hugging Face cache,
or a Wan2.2 bundle.

The container installs the CUDA 13.3 user-space stack, TensorRT 11.2.0.113,
Python 3.12, PyTorch 2.12/cu130, the Wan2.2 builder, and the native Model
Connect runtime. The host remains responsible for the BSP, GPU driver, Docker,
NVIDIA Container Toolkit, and credentials.

## Requirements

- Jetson AGX Thor with 128 GB unified memory and a driver compatible with
  CUDA 13.3 containers.
- At least 150 GB of free disk space; 200 GB is recommended when retaining
  the image, 32 GB checkpoint cache, 20.6 GB bundle, and build cache.
- Read access to the private TensorRT-Model-Connect GitHub repository.
- Read access to the repository-linked
  `ghcr.io/nvidia/tensorrt-model-connect/tensorrt-sdk` package.
- Internet access to GitHub, GHCR, the PyTorch wheel index, and Hugging Face.

The Docker container cannot install or replace the Thor BSP or GPU driver.
Use a supported Thor software image before continuing.

## 0. Install The Thor BSP

If the board has no operating system, use the official
[Jetson AGX Thor Quick Start Guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html)
to install the current supported Jetson BSP from the Jetson ISO, then finish
the first-boot user and network setup. The alternative SDK Manager and flash
script paths are covered by the
[Thor BSP setup guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_bsp.html).

Log in to the board and confirm the host driver is working:

```bash
nvidia-smi
```

Continue only after the command identifies `NVIDIA Thor`. The remaining steps
start from this host boundary and install all Model Connect user-space
dependencies inside Docker.

## 1. Install Docker And NVIDIA Container Toolkit

Follow the
[Jetson AGX Thor Docker setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html).
On a BSP installed with the flash script or SDK Manager, the host setup is:

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

Log out and back in after adding the Docker group. If local policy does not
permit Docker-group membership, prefix every Docker command below with
`sudo`. Membership in the Docker group grants root-equivalent host access, so
enterprise environments may prefer the `sudo docker` path.

The Jetson ISO installation may already provide Docker and the NVIDIA
Container Toolkit. The guarded installer above leaves an existing Docker
installation in place; still run the runtime configuration and GPU smoke test.

Verify the host/container boundary before building Model Connect:

```bash
docker run --rm --runtime=nvidia --gpus all \
  nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi
```

The output must identify `NVIDIA Thor`. If this fails, fix the host driver,
Docker, or NVIDIA Container Toolkit before continuing. The general toolkit
configuration is documented in the
[NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## 2. Clone The Source And Authenticate GHCR

Clone with the GitHub credential mechanism approved by your organization. This
example uses an SSH key:

```bash
git clone git@github.com:NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect
git checkout main
git pull --ff-only
test -z "$(git status --porcelain)"
```

The TensorRT 11.2 SDK image is access-controlled. Create a GitHub token with
read access to the linked package, then log Docker in without placing the
token in the image or command history:

```bash
read -rsp "GitHub package token: " GITHUB_PACKAGE_TOKEN
echo
printf '%s' "$GITHUB_PACKAGE_TOKEN" | \
  docker login ghcr.io --username YOUR_GITHUB_USERNAME --password-stdin
unset GITHUB_PACKAGE_TOKEN
```

Repository read permission and GHCR package permission are separate. An
`unauthorized` error while resolving the TensorRT SDK image means the account
does not yet have the required package access.

Without a Docker credential helper, `docker login` stores the credential in
the current user's `~/.docker/config.json`. Use the credential helper approved
by your organization, or run `docker logout ghcr.io` after the image is built.
If the workflow uses `sudo docker`, the login and logout commands must use
`sudo docker` as well.

## 3. Build The Thor Image

The dedicated Dockerfile builds only the native CLI, TensorRT backend, and
Wan2.2 runtime DSO for SM110. It does not build the GB300 CI profiles or other
model DSOs.

```bash
REVISION="$(git rev-parse HEAD)"
IMAGE="trtmc-wan22-thor:${REVISION:0:12}"

docker build --pull \
  --build-arg "TRTMC_SOURCE_REVISION=$REVISION" \
  --file Dockerfile.wan22-thor \
  --tag "$IMAGE" \
  .

docker image inspect "$IMAGE" \
  --format 'image={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
```

The final build checks TensorRT, PyTorch, Transformers, `trtmc version`, the
TRT backend DSO, and the Wan2.2 model DSO.

## 4. Create Persistent Storage

The checkpoint cache and generated bundle must survive the short-lived build
and generation containers:

```bash
DATA_ROOT="$HOME/trtmc-wan22"
mkdir -p "$DATA_ROOT"/{home,huggingface,work}
```

For a cold-download qualification, use a new empty `DATA_ROOT`. Reusing it on
later builds avoids downloading the 32 GB checkpoint again.

## 5. Build The Bundle From The Hugging Face Model ID

This command downloads the pinned official checkpoint, verifies its contents,
loads the packaged FP8 scales, builds target-specific TensorRT engines, and
writes the bundle under `work/`:

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

No local checkpoint path, quantization JSON, plugin path, or backend selector
is required.

Inspect the result before generation:

```bash
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --env HOME=/data/home \
  --volume "$DATA_ROOT:/data" \
  "$IMAGE" \
  inspect /data/work/wan22-thor.trtfb
```

## 6. Generate The Full 720p Video

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
shift 5, and 24 FPS profile. The runtime schema supplies the 7/2 exact-step
windows. No Wan2.2 environment variable is used.

Verify the output:

```bash
find "$DATA_ROOT/work/wan22-frames" -maxdepth 1 -name 'frame_*.png' | wc -l
file \
  "$DATA_ROOT/work/wan22-frames/frame_0000.png" \
  "$DATA_ROOT/work/wan22-frames/frame_0060.png" \
  "$DATA_ROOT/work/wan22-frames/frame_0120.png"
cat "$DATA_ROOT/work/wan22-thor.effective_config.json"
```

Expected signals are 121 PNG files, `1280 x 704` for every representative
frame, and `session_request` as the source of the four explicit
`wan2_2_ti2v.*` settings.

Create a customer-viewable 24 FPS MP4 from the complete frame sequence:

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

The expected MP4 metadata is H.264, 1280x704, 121 frames, and 24 FPS.

## Optional: Match The Qualified Performance Setup

Functional deployment does not require fixed clocks. To compare against the
qualified approximately 205-second generation, place the board in MAXN and
lock clocks before running:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo nvpmodel -q
sudo jetson_clocks --show
```

Clock commands affect the host and require the power and cooling configuration
recommended for the Thor developer kit.

## Authentication And Reproducibility Boundaries

- GitHub source access does not automatically grant GHCR package access.
- The public Wan2.2 checkpoint normally downloads without a Hugging Face
  token. Supply account credentials only if your network or account policy
  requires them; do not bake tokens into the image.
- Docker makes the user-space stack reproducible. It does not prove every
  custom or future BSP/driver combination.
- The first cold run includes checkpoint download time. Generation timing
  begins after the bundle already exists.
