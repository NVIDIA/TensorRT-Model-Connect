---
title: Intermediate Tutorial - Diffusion, Vision, and Time-Series Pipelines
---

import Diagram from '@site/src/components/Diagram';

## Learning objectives

- build and invoke image, video, forecasting, and segmentation Tasks;
- recognize that each family owns its component graph and bundle sections;
- use the family manifest rather than infer support from a shared CLI command.

<Diagram
  src="/img/diagrams/tutorials/intermediate/task-pipeline-patterns.svg"
  alt="Typed input reaches family-owned preprocessing, TensorRT engines, postprocessing, and an abstract Task result"
  caption="The user-facing pattern is shared; every concrete processing loop stays in its family."
/>

Set the runtime installation used by all examples:

```bash
export TRTMC_RUNTIME_ROOT=/opt/trtmc/lib
```

## FLUX image generation

```bash
python -m pip install -r families/flux/requirements.txt
python -m tensorrt_model_connect build black-forest-labs/FLUX.1-schnell \
  --output flux-schnell.bundle \
  --precision fp32 \
  --image-height 384 \
  --image-width 384

trtmc generate-image flux-schnell.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "A photo of a cat sitting on a windowsill at sunset" \
  --output flux.png \
  --height 384 \
  --width 384 \
  --num-steps 20 \
  --seed 42
```

<Diagram
  src="/img/diagrams/tutorials/intermediate/flux-denoising-pipeline.svg"
  alt="Prompt conditioning and latent noise pass repeatedly through the FLUX denoiser and scheduler before VAE decoding"
  caption="FLUX owns all component engines, scheduler behavior, sections, and image-generation evidence."
/>

The shared API exposes `IImageGeneration`; it does not prescribe FLUX's text
encoders, denoising loop, scheduler, or VAE.

### Advanced FLUX context parallelism

Use the exact CP4 contract in
`families/flux/tests/manifests/flux-schnell-l0-cp4.json`. Build with
`--context-parallel-size 4` and follow the
[multi-device lab](../advanced/multi-device-inference.md). This documentation
does not claim multi-device execution on the current machine.

## PixArt-Sigma image generation

```bash
python -m pip install -r families/pixart/requirements.txt
python -m tensorrt_model_connect build PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
  --output pixart-sigma.bundle \
  --precision fp16 \
  --image-height 1024 \
  --image-width 1024

trtmc generate-image pixart-sigma.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "A small cabin beneath the northern lights" \
  --output pixart.png \
  --height 1024 \
  --width 1024
```

Check `families/pixart/tests/manifests/` for the exact step count, seed, and
oracle used by a qualified case.

## Wan video generation

```bash
python -m pip install -r families/wan_t2v/requirements.txt
python -m tensorrt_model_connect build Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --output wan21.bundle \
  --precision fp16

trtmc generate-video wan21.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "Ocean waves at sunrise" \
  --output wan-frames \
  --seed 42
```

The CLI writes the returned frames. The Wan family owns temporal shape,
component engines, preprocessing, denoising, and output semantics.

### Advanced recipe: Wan2.2 TI2V

Wan2.2 TI2V is a separate family under `families/wan2_2_ti2v/`. Use its exact
manifest and install its own requirements. Do not reuse Wan2.1 paths or infer a
Jetson/Thor support claim without target-hardware evidence.

## Chronos-Bolt time-series forecasting

### Build the official tiny model

```bash
python -m pip install -r families/chronos_bolt/requirements.txt
python -m tensorrt_model_connect build amazon/chronos-bolt-tiny \
  --output chronos-bolt-tiny.bundle \
  --precision fp32
```

### Forecast from float32 input

The native CLI accepts raw float32 values. Create the example input with the
Python standard library:

```bash
python -c 'from array import array; array("f", [100.1,100.15,100.18,100.22,100.21,100.27,100.31,100.35,100.37,100.4,100.44,100.5]).tofile(open("history.f32", "wb"))'

trtmc forecast chronos-bolt-tiny.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input history.f32
```

The canonical input and expected shape are owned by
`families/chronos_bolt/tests/manifests/chronos-bolt-tiny-official.json`.

## Segmentation mental model

<Diagram
  src="/img/diagrams/tutorials/intermediate/segmentation-detection-pipeline.svg"
  alt="Image pixels pass through family-owned preprocessing and a vision engine before a typed segmentation result"
  caption="The current shared Task surface includes segmentation; it does not advertise a generic object-detection Task."
/>

SegFormer is one concrete segmentation family:

```bash
python -m tensorrt_model_connect build nvidia/segformer-b0-finetuned-ade-512-512 \
  --output segformer.bundle \
  --precision fp16

trtmc segment segformer.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image families/segformer/tests/data/test_img.jpeg
```

The family implements `ISegmentation` and returns a typed mask. The current
CLI has no generic `detect` command, so the retained diagram and this lab make
no object-detection support claim.

## What you should understand now

- A Task interface describes user behavior, not a shared model pipeline.
- Each family owns its graph, sections, dependencies, runtime, and oracle.
- Similar modalities do not justify cross-family imports.
- Build success, output quality, target support, and performance are distinct
  evidence levels.

## Self-check

1. Why do FLUX and PixArt not share a denoising implementation in core?
2. Which manifest owns the Chronos input contract?
3. Why must a Wan2.2 target claim name its tested hardware?
4. What does `ISegmentation` guarantee, and what stays model-owned?
