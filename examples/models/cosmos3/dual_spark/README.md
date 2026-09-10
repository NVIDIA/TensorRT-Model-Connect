<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 dual-Spark video generation example

This example generates one native 1280x720 Cosmos3-Nano video with context
parallelism (CP=2) across two one-GPU DGX Sparks. Run one launcher command on
the primary Spark. It prepares and synchronizes the image and TensorRT bundle,
then invokes one `mpirun` command that starts an MPI worker on each Spark. Each
worker owns one local Docker container. You do not log in to the peer to start
rank 1.

On top of the repository-pinned TensorRT base, the image adds the TRTMC CLI,
the core and explicit runtime loader, the TensorRT backend, the Cosmos3 model
DSO, the Python builder environment, FFmpeg, and RDMA runtime packages. It does
not contain a model checkpoint, TensorRT bundle, generated video, SSH key, or
Hugging Face token. The example does not start a server, open an application
port, or require a browser.

## Requirements

- two networked, one-GPU GB10 DGX Sparks with matching GPU architecture and
  NVIDIA driver versions;
- Docker Engine using BuildKit and NVIDIA Container Toolkit on both Sparks;
- matching Open MPI and `mpi4py` installations on both Sparks;
- passwordless SSH from the primary to the peer, with the peer host key already
  present in `known_hosts` and Docker available without an interactive prompt;
- a direct, active RoCE link between the Sparks; and
- enough free disk, unified memory, and swap for the checkpoint and TensorRT
  build.

Install the MPI prerequisites once on both Sparks:

```bash
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev python3-mpi4py
mpirun --version
python3 -c 'from mpi4py import MPI; print(MPI.get_vendor())'
```

The Open MPI and `mpi4py` vendor versions reported by the two Sparks must
match. Run the remaining commands on the primary Spark from the repository
root.

## How launch and rendezvous work

The launcher invokes `mpirun` on the primary with two hosts and one process per
host. Rank 0 starts its local container, which creates and publishes the live
NCCL unique ID. The MPI worker broadcasts the 128-byte ID with `MPI_Bcast`.
Rank 1 receives it and starts the peer container with the ID in its environment.
Both containers then communicate directly over NCCL/RoCE. MPI is the launch and
bootstrap control plane; it is not the model data plane.

The MPI workers remain alive and wait for their containers. This lets `mpirun`
track the complete distributed job even though model execution stays isolated
inside Docker. No NCCL rendezvous file is copied to the peer.

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

After the one-time image build, run the complete prepare-and-generate workflow:

```bash
python3 examples/models/cosmos3/dual_spark/run_dual_spark.py all \
  --peer-host <SECOND_SPARK> \
  --image trtmc-cosmos3-dual-spark:local \
  --scene showcase-high-speed-racing
```

The first run downloads the public `nvidia/Cosmos3-Nano` checkpoint, builds a
CP=2 TensorRT bundle on the primary Spark, and copies the image and bundle
to the peer with strict SSH. That preparation can take hours. Later runs reuse
the checkpoint cache, image, and hardware-specific bundle.

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

Add `--peer-user <USER>` when the peer username differs from the primary
username. Add `--ssh-key <KEY>` and `--known-hosts <KNOWN_HOSTS>` when SSH does
not use the defaults. The same strict SSH configuration is passed to Open MPI.

Use `--dry-run` to print the mutation-free JSON execution plan. Run
`python3 examples/models/cosmos3/dual_spark/run_dual_spark.py --help` for the
complete CLI surface.

By default, the launcher prints concise workflow milestones and actionable
errors. Set `COSMOS3_LOG_COMMANDS=1` only while troubleshooting to also print
sanitized local and peer commands; encoded SSH transport payloads and inline
scripts are never printed.

## Operational boundaries and output locations

- SSH is non-interactive and strict host-key checking remains enabled. Use
  `--ssh-key` and `--known-hosts` for non-default SSH files.
- The default RoCE settings are HCA `rocep1s0f0:1`, network interface
  `enp1s0f0np0`, and GID index `3`. Change them only when both Sparks use the
  same alternative configuration.
  The launcher pins Open MPI TCP traffic to the same direct-link network
  interface, avoiding accidental use of another host interface.
- The launcher requires NCCL RoCE (`NET/IB`) rather than socket fallback and
  records cgroup-v2 peak memory for both ranks. The MPI stdout and stderr logs
  are saved with each scene.
- Reusable assets and run records live under
  `~/.cache/trtmc/cosmos3-physics` by default. Each run directory contains its
  MP4 files, rank logs, container records, and `run.json`. Use `--work-root`
  to select another primary location; the peer defaults to
  `/var/tmp/cosmos3-physics-dual-spark`.
- The public checkpoint is downloaded at preparation time into the reusable
  model directory under the work root. No credentials are required. Generated
  motion can still contain physical or visual errors; review outputs before
  sharing them.

## Common failures

- `mpirun` or `mpi4py` missing: install the MPI prerequisites on both Sparks.
- MPI version mismatch: install the same Open MPI and `mpi4py` packages on both
  hosts.
- SSH launch failure: verify the same key and `known_hosts` file with a
  non-interactive `ssh` command from the primary.
- Bundle `Permission denied` during `scp`: new bundles are built with the
  primary user's UID and GID. The launcher also repairs ownership of a reusable
  bundle created by an older root-running example container.
