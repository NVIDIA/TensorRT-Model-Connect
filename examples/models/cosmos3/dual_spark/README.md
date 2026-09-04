<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 dual-Spark video generation example

This example generates one native 1280x720 Cosmos3-Nano video with context
parallelism across two one-GPU DGX Sparks. A host-side Python launcher prepares
the hardware-specific TensorRT bundle, synchronizes the image and bundle
to the peer, and starts rank 0 on the primary Spark and rank 1 on the peer.
It does not start a server, open an application port, or require a browser.

On top of the repository-pinned TensorRT base, the image adds the TRTMC CLI,
the core and explicit runtime loader, the TensorRT backend, the Cosmos3 model
DSO, the Python builder environment, FFmpeg, and RDMA runtime packages. It does not contain a model
checkpoint, TensorRT bundle, generated video, SSH key, or Hugging Face token.

## Requirements

- two networked, one-GPU GB10 DGX Sparks with matching GPU architecture and
  NVIDIA driver versions;
- Docker Engine using BuildKit and NVIDIA Container Toolkit on both Sparks;
- passwordless SSH from the primary to the peer, with the peer host key already
  present in `known_hosts` and Docker available without an interactive prompt;
- a direct, active RoCE link between the Sparks; and
- enough free disk, unified memory, and swap for the checkpoint and TensorRT
  build.

Run all commands below on the primary Spark from the repository root.

## Build the image once

The default base is the repository-pinned aarch64 TensorRT 26.07 image and the
default CUDA target is the GB10 GPU's SM 12.1:

```bash
docker build \
  --platform linux/arm64 \
  --file examples/models/cosmos3/dual_spark/Dockerfile \
  --tag trtmc-cosmos3-dual-spark:local \
  .
```

Build natively on the primary Spark. TensorRT bundles are specific to the model
revision, precision, context-parallel topology, TensorRT build, and GPU
architecture.

## Generate one video

After the one-time image build, select one of the built-in scenes:

```bash
python3 examples/models/cosmos3/dual_spark/run_dual_spark.py all \
  --peer-host <SECOND_SPARK> \
  --image trtmc-cosmos3-dual-spark:local \
  --scene showcase-high-speed-racing
```

The first run downloads the public `nvidia/Cosmos3-Nano` checkpoint, builds a
CP=2 TensorRT bundle on the primary Spark, and copies the image and bundle
to the peer. That preparation can take hours. Later runs reuse the checkpoint
cache, image, and hardware-specific bundle.

Each showcase preset produces a 189-frame, 7.875-second H.264 MP4 at 24 FPS.
The available presets correspond to the selected showcase videos:

- `showcase-high-speed-racing`;
- `showcase-mars-robots`;
- `showcase-delivery-robot`;
- `showcase-apple-to-plate`;
- `showcase-humanoid-sprint`; and
- `showcase-cake-cutting`.

Use `--scene all` to run the six presets sequentially. To separate the
expensive preparation from generation, use the `prepare` and `run` actions:

```bash
python3 examples/models/cosmos3/dual_spark/run_dual_spark.py prepare \
  --peer-host <SECOND_SPARK> \
  --image trtmc-cosmos3-dual-spark:local

python3 examples/models/cosmos3/dual_spark/run_dual_spark.py run \
  --peer-host <SECOND_SPARK> \
  --image trtmc-cosmos3-dual-spark:local \
  --scene showcase-delivery-robot
```

Use `--dry-run` to print the mutation-free JSON execution plan. Run
`python3 examples/models/cosmos3/dual_spark/run_dual_spark.py --help` for the
complete CLI surface.

## Network, cache, and output boundaries

- SSH is non-interactive and strict host-key checking remains enabled. Use
  `--ssh-key` and `--known-hosts` for non-default SSH files.
- The default RoCE settings are HCA `rocep1s0f0:1`, network interface
  `enp1s0f0np0`, and GID index `3`. Change them only when both Sparks use the
  same alternative configuration.
- The launcher requires NCCL RoCE (`NET/IB`) rather than socket fallback and
  records cgroup-v2 peak memory for both ranks.
- Reusable assets and run records live under
  `~/.cache/trtmc/cosmos3-physics` by default. Each run directory contains its
  MP4 files, rank logs, container records, and `run.json`. Use `--work-root`
  to select another primary location; the peer defaults to
  `/var/tmp/cosmos3-physics-dual-spark`.
- The public checkpoint is downloaded at preparation time into the reusable
  model directory under the work root. No credentials are required. Generated
  motion can still contain physical or visual errors; review outputs before
  sharing them.
