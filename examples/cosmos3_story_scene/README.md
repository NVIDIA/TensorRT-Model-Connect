<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos Story Scene

Turn one creative brief into a 7.9-second story scene. Pick a pattern such
as Impossible ASMR, Pocket Universe, Product Metamorphosis, Plot Twist, or
Nature Glitch; direct the subject, hook, reveal, camera, and lighting; then
download both the clean landscape master and a derived vertical social cut.

Cosmos Story Scene is a dependency-free Python web app around TensorRT Model
Connect (TRTMC) and its native TensorRT backend. It runs entirely on your GPU
server. Cosmos3-Nano is public, so the default launch needs no Hugging Face
credential; if you supply one for authenticated Hub access, the browser never
receives it.

## Before you start

This sample targets Linux **x86_64** and **aarch64** and requires:

- Docker Engine, Docker Compose v2 with `gpus: all` support, and the NVIDIA
  Container Toolkit configured for Docker;
- an NVIDIA driver compatible with the pinned TensorRT 26.07 / CUDA container;
- network access to the public
  [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano)
  checkpoint for the first download;
- plan for at least one **80GB-class NVIDIA GPU**. Context-parallel sizes 2, 4,
  and 8 require that many visible, same-architecture GPUs. The recovered #763
  runtime was previously exercised on A100-SXM4-80GB, but its exact minimum
  memory was not established. This rebased app has not yet been rerun on GPU;
  the image includes Ada, Hopper, and Blackwell cubins for later qualification;
- substantial host resources: budget at least 150 GB of free disk for the NGC
  image, checkpoint cache, temporary build data, TensorRT bundle, and clips.

The first start downloads the pinned checkpoint revision and compiles a bundle
for the visible GPU architecture. That may take hours. Later starts reuse the
named Hugging Face cache and TensorRT bundle volumes.

Verify GPU passthrough before spending time on the image build:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

If your environment requires NGC authentication, log in to `nvcr.io` before
building the image.

## Launch with one command

No Hugging Face token is required for the public checkpoint. Launch everything:

```bash
docker compose up --build
```

On an aarch64 DGX Spark, build the same image with the repository-pinned ARM
TensorRT base and compile only the native GB10 cubin:

```bash
docker compose build \
  --build-arg TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.07-py3@sha256:f794a79e8b996d16dbc2e5884e19d8e2269a51c960106c9b49b0061a6926c541 \
  --build-arg TRTMC_CUDA_ARCHITECTURES=121-real
docker compose up
```

The image derives the TensorRT library/include triplet from the target compiler,
so the same Dockerfile does not hard-code x86 library paths. A GB10 engine must
still be built and qualified on the target Spark before treating it as portable.

Open <http://localhost:8080>. The service stays in `starting` health state
while the first bundle is built. Follow truthful build and job progress with:

```bash
docker compose logs -f story-scene
```

The image never copies an optional credential, checkpoint cache, model bundle,
or generated clips. Compose uses an empty secret by default. If authenticated
Hub access is required, set `HF_TOKEN_FILE` to an absolute, mode-0600 token
file; the entrypoint exposes it only to the first `trtmc build` child and
removes it from the web app environment afterward. An `HF_TOKEN` environment
fallback exists for non-Compose launches and emits a security warning.

## Multi-GPU context parallelism

The default is one GPU. Select a supported context-parallel topology at launch:

```bash
COSMOS3_CP_SIZE=2 docker compose up --build
```

Valid values are `1`, `2`, `4`, and `8`. The container includes Open MPI and
the app launches one TRTMC rank per device for values greater than one. The
entrypoint rejects too few GPUs and mixed GPU architectures. Changing the CP
size or GPU architecture causes an atomic rebuild of `/models/cosmos3.trtfb`;
an interrupted rebuild leaves the previously complete bundle intact.

TensorRT bundles are hardware-, topology-, TensorRT-, precision-, and model-
revision-specific. Do not copy this volume to a different GPU class and assume
it is portable. The sample always builds:

```text
nvidia/Cosmos3-Nano
revision 411f42a8fdfb8c5b2583cb8786e0938f49796eaa
BF16, CP 1/2/4/8
```

## How it fits together

```text
Browser creative brief
        |
        v
stdlib Python HTTP app  -- one serialized generation worker
        |
        +--> trtmc (mpirun for CP > 1)
        |      |
        |      +--> native Cosmos3 model plugin + TensorRT backend
        |      +--> persisted, hardware-specific .trtfb bundle
        |
        +--> ffmpeg --> clean 16:9 master + derived 9:16 social cut
                            |
                            v
                     /outputs volume
```

The three named volumes survive `docker compose down`:

| Volume | Container path | Purpose |
| --- | --- | --- |
| `hf-cache` | `/root/.cache/huggingface` | Pinned public checkpoint cache |
| `cosmos3-models` | `/models` | Hardware-specific TRTMC bundle and build marker |
| `generated-clips` | `/outputs` | Clean and social MP4 results |

## Honest limitations

- **Text-to-video only.** This app does not accept an image or video input.
- **Fixed model profile.** Cosmos3-Nano runs 1280x720, 189 frames, 24 FPS,
  35 denoising steps, guidance 6.0, and flow shift 10.0 (about 7.9 seconds).
- **No safety checker.** Review prompts and outputs, add policy controls for
  your deployment, and do not expose this demo directly to the public internet.
- **Derived vertical format.** The 9:16 result is an ffmpeg reframing of the
  landscape source, not native vertical generation.
- **One worker.** Jobs are deliberately serialized to prevent concurrent model
  loads from exhausting GPU memory; this is a creative demo, not a scaled API.
- **Single node.** CP 2/4/8 uses multiple GPUs in one host, not multi-node MPI.

## Operations and cleanup

Copy all server-side results out while the service container still exists:

```bash
docker compose cp story-scene:/outputs ./cosmos-outputs
```

Then stop containers while retaining the model, cache, and clips:

```bash
docker compose down
```

To remove the checkpoint cache, compiled bundle, and every generated clip,
stop the stack and delete its volumes. **This is irreversible unless you copied
the outputs first:**

```bash
docker compose down --volumes
```

If you supplied an optional token file, remove it separately when it is no
longer needed.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Hugging Face returns 401/403 | Confirm the public model URL is reachable from the host. If site policy requires authenticated Hub access, set `HF_TOKEN_FILE` to a read-token file; never paste the token into logs. |
| `nvidia-smi` is missing or no GPU is visible | Install/configure NVIDIA Container Toolkit, restart Docker, and rerun the GPU passthrough check above. |
| NGC image pull is denied | Authenticate to `nvcr.io` and confirm access to the TensorRT container. |
| Health remains `starting` | On first run this is expected for hours; inspect `docker compose logs -f story-scene` for download and TensorRT build progress. |
| TensorRT build runs out of memory | Use an 80GB-class or larger build GPU, stop other GPU workloads, and confirm adequate host RAM/swap and disk. CP does not distribute the one-process engine build. |
| CP launch reports world-size or architecture errors | Expose at least `COSMOS3_CP_SIZE` contiguous, same-architecture GPUs and use only 1, 2, 4, or 8. |
| Port 8080 is busy | Launch with `STORY_SCENE_PORT=8090 docker compose up --build`, then open `http://localhost:8090`. |
| A source/runtime change still reuses an old bundle | Engine compatibility is intentionally conservative; run `docker compose down --volumes` to force a clean checkpoint and bundle build (copy outputs first). |
