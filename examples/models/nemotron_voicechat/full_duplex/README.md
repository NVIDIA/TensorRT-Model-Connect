<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Nemotron VoiceChat full-duplex microphone example

This example runs one local, in-process `ISpeechSession`. It continuously
captures a Linux ALSA microphone while playing the agent's audio, so a user can
interrupt the agent without waiting for playback to finish. It does not start a
server, open a network port, or use WebSocket or Python at runtime.

On top of the repository-pinned TensorRT base, the application layer adds only
the example executable, the core and explicit runtime loader, the TensorRT
backend, the Nemotron VoiceChat model DSO, and ALSA runtime packages. It does
not contain the checkpoint or a bundle.

## Requirements

- Linux with a current Docker Engine using BuildKit, NVIDIA Container Toolkit,
  and a compatible NVIDIA driver;
- a local ALSA capture and playback device under `/dev/snd`;
- one GPU with enough memory for the bundle (the repository qualification uses
  at least 90,000 MiB of free GPU memory); and
- a prebuilt `nemotron_voicechat` bundle for the same GPU architecture and
  TensorRT 11.1 runtime used by this image.

The qualified FP32 VoiceChat bundle is about 46.5 GB. Keep it outside the image
and mount it read-only at runtime.

## Build the image once

Run the build from the repository root. The default Docker base and CUDA target
are the repository-pinned aarch64 TensorRT 26.07 image and GB300 SM 10.3:

```bash
docker build \
  --platform linux/arm64 \
  --file examples/models/nemotron_voicechat/full_duplex/Dockerfile \
  --tag trtmc-voicechat-full-duplex:local \
  .
```

For a native x86_64 build, override both the pinned architecture-specific base
digest and the CUDA architecture. Derive the latter from the GPU that the
bundle targets; this example shows a B200 (`sm_100`):

```bash
docker build \
  --platform linux/amd64 \
  --file examples/models/nemotron_voicechat/full_duplex/Dockerfile \
  --build-arg TENSORRT_IMAGE='nvcr.io/nvidia/tensorrt:26.07-py3@sha256:b82db1abc23750ab0069abc99bbe4ea29138dbdc23ea39861199e2346638b48a' \
  --build-arg TRTMC_CUDA_ARCHITECTURES=100-real \
  --tag trtmc-voicechat-full-duplex:local \
  .
```

Build natively for the host architecture. Changing the application image's
CUDA architecture does not make an existing TensorRT bundle portable: rebuild
the bundle for the target GPU as well.

## Start a conversation

After the one-time image build, each conversation starts with one `docker run`:

```bash
VOICECHAT_BUNDLE="$(realpath nemotron-voicechat-11b.bundle)"

docker run --rm --interactive --tty \
  --network none \
  --gpus 'device=0' \
  --device /dev/snd:/dev/snd \
  --mount "type=bind,src=${VOICECHAT_BUNDLE},dst=/models/model.bundle,readonly" \
  trtmc-voicechat-full-duplex:local
```

The image's default command opens `/models/model.bundle`, the ALSA `default`
capture device, and the ALSA `default` playback device. Press Ctrl-C to stop.
No `--privileged`, host IPC, network access, checkpoint mount, or credentials
are required.

List ALSA devices without loading a bundle:

```bash
docker run --rm --interactive --tty \
  --network none \
  --device /dev/snd:/dev/snd \
  trtmc-voicechat-full-duplex:local \
  --list-devices
```

Select devices explicitly when `default` is not the desired hardware endpoint:

```bash
VOICECHAT_BUNDLE="$(realpath nemotron-voicechat-11b.bundle)"

docker run --rm --interactive --tty \
  --network none \
  --gpus 'device=0' \
  --device /dev/snd:/dev/snd \
  --mount "type=bind,src=${VOICECHAT_BUNDLE},dst=/models/model.bundle,readonly" \
  trtmc-voicechat-full-duplex:local \
  --capture-device 'plughw:CARD=Headset,DEV=0' \
  --playback-device 'plughw:CARD=Headset,DEV=0' \
  /models/model.bundle
```

Run `docker run --rm trtmc-voicechat-full-duplex:local --help` for the complete
CLI surface.

## Audio and container boundaries

- Use a headset or hardware acoustic echo cancellation. ALSA and this example
  do not provide AEC; speaker output captured by the microphone can otherwise
  look like a user interruption.
- The minimal container contract passes physical ALSA devices directly through
  `/dev/snd`. A desktop's PipeWire or PulseAudio `default` device is not
  automatically forwarded into the container. Use an explicit `plughw` device,
  or configure desktop audio socket forwarding separately.
- The host audio device must support simultaneous capture and playback. USB
  headsets are generally easier to isolate than a shared desktop sound card.
- Docker Desktop on macOS or Windows, remote GPU machines without `/dev/snd`,
  and rootless Docker configurations that cannot pass GPU/audio devices do not
  satisfy this direct-device contract.
- The runtime is deliberately offline. Build or obtain the model bundle before
  starting the container; do not pass Hugging Face tokens or model credentials
  to the application image.
- A Dockerfile in the repository supports a one-time local image build followed
  by one-command runs. A literal pull-and-run experience on a fresh machine
  additionally requires publishing this image to a container registry.
