---
title: Intermediate Tutorial - Diffusion, Vision, and Time-Series Pipelines
---

import Diagram from '@site/src/components/Diagram';

This tutorial covers pipelines that do not return generated text.

## Learning objectives

By the end of this lab, you should be able to distinguish token decoding from
denoising and forecasting loops, select the correct typed task method, and use
an exact E2E manifest to identify canonical non-text inputs and outputs.

These pipelines still use `IPipeline`, but their task methods and internal loops differ.

The commands below assume that you completed
[Installation](../../getting-started/installation.md) and are in a
TensorRT/CUDA environment with a supported NVIDIA GPU. Model builds download
checkpoint data unless it is already cached, so they also need network access
or a populated local cache.

<Diagram
  src="/img/diagrams/tutorials/intermediate/task-pipeline-patterns.svg"
  alt="Shared task pipeline showing stable typed input and result boundaries around task-specific TensorRT execution patterns"
  caption="Text, diffusion, audio, vision, and forecasting use the same public pipeline pattern, but the work inside the model-owned execution stage is task specific."
/>

## FLUX image generation

```bash
trtmc build black-forest-labs/FLUX.2-dev \
  -o /tmp/flux2.bundle \
  --precision fp16 \
  --image-height 1024 \
  --image-width 1024 \
  --num-inference-steps 28

trtmc generate-video /tmp/flux2.bundle \
  --prompt "A photo of a cat sitting on a windowsill at sunset" \
  --output /tmp/flux2-frames \
  --num-steps 28
```

The command is named `generate-video` because the C++ API returns an `ImageResult` with `num_frames`; single-frame image generation is the same surface as video generation.

<Diagram
  src="/img/diagrams/tutorials/intermediate/flux-denoising-pipeline.svg"
  alt="FLUX diffusion pipeline combining prompt conditioning and latent noise in a denoiser and scheduler loop before VAE decoding"
  caption="Diffusion repeatedly updates latent tensors for a configured number of steps, then the VAE decoder converts the final latents into ImageResult pixels or frames."
/>

Diffusion inference is iterative like text generation, but the loop is over denoising steps rather than generated tokens.

| Text generation | Diffusion |
| --- | --- |
| Generates one token per decode step. | Updates a latent tensor each denoising step. |
| Uses KV cache to avoid recomputing prompt attention. | Uses a scheduler to control the noise trajectory. |
| Samples from logits. | Decodes final latents into pixels. |
| Stops at EOS or token budget. | Stops after configured denoising steps. |

### Advanced FLUX context parallelism

The build CLI also exposes `--context-parallel-size` (or `--cp-size`) for
families that implement context parallelism. The current checked multi-device
contract is `tests/e2e/models/flux/manifests/flux-schnell-l0-cp4.json`:
it uses FLUX.1 Schnell, four ranks, `mpirun`, NCCL rendezvous, and four GPUs.
The FLUX builder stores one shared rank-dynamic Ulysses denoiser plan while
replicating weights and sharding sequence activations. Tensor-parallel and
context-parallel build options cannot be combined.

Treat the manifest and its retained E2E result as the runnable qualification
contract. Parser acceptance or a successful single-rank build does not prove
the four-rank runtime path.

## PixArt-Sigma image generation

This recipe mirrors the repository's
`tests/e2e/models/pixart/manifests/pixart-sigma-1024-l0.json` contract: the
same checkpoint, FP16 precision, FP32 text-encoder selector, 256-token cache,
512-by-512 spatial profile, prompt, and 20 denoising steps.

```bash
trtmc build PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
  -o /tmp/pixart-sigma-512.bundle \
  --precision fp16 \
  --fp32-layers 0 \
  --max-cache-length 256 \
  --image-height 512 \
  --image-width 512 \
  --num-inference-steps 20

trtmc generate-video /tmp/pixart-sigma-512.bundle \
  --prompt "A photo of a cat sitting on a windowsill at sunset" \
  --output /tmp/pixart-sigma-512-frames \
  --num-steps 20
```

Success creates a single image frame in
`/tmp/pixart-sigma-512-frames`. The bundle should inspect as family `pixart`
with runtime strategy `diffusion_pixart`. The E2E manifest classifies that
runtime under the `diffusion_media_generation` task strategy; the regular
bundle inspector does not print the E2E task-strategy field.

This direct smoke test uses the L0 manifest's public inputs, but it is not by
itself the repository's parity proof. The E2E harness controls the initial
latents and runs its configured framework comparison so both implementations
start from equivalent data.

## Wan video generation

```bash
trtmc build Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  -o /tmp/wan21.bundle \
  --precision fp16 \
  --video-height 480 \
  --video-width 832 \
  --video-num-frames 81

trtmc generate-video /tmp/wan21.bundle \
  --prompt "A cinematic shot of clouds moving over mountains" \
  --output /tmp/wan21-frames
```

Diffusion bundles often have multiple engine sections: text encoder, denoiser, and VAE decoder.

Video generation adds temporal dimensions. The output is still `ImageResult`, but `num_frames` is greater than one and the latent tensor includes time.

## Advanced recipe: Jetson Thor Wan2.2 720p

This is an advanced target-specific example, not an environment smoke test.
Use a release wheel with its `wan` build extra, then build and run the packaged
profile on the target:

```bash
trtmc build Wan-AI/Wan2.2-TI2V-5B \
  --model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --fp8 \
  --output wan22-thor.bundle

trtmc generate-video wan22-thor.bundle \
  --set wan2_2_ti2v.easycache_enabled=true \
  --set wan2_2_ti2v.easycache_threshold=1.0 \
  --set wan2_2_ti2v.easycache_max_consecutive_reuse=4 \
  --set wan2_2_ti2v.late_cfg_enabled=true \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
  --output wan22-frames \
  --seed 42
```

The public source tree does not retain the earlier internal-SDK performance
receipt as qualification for the official TensorRT release path. A latency,
quality, or release-readiness claim requires a fresh target-hardware run with
the exact bundle, software cohort, prompt, seed, and retained artifacts.

## Chronos-Bolt time-series forecasting

Chronos-Bolt uses the neural-operator task surface: a numeric history enters as
the branch input, and `solve()` returns the forecast vector. The repository's
official contract is
`tests/e2e/models/chronos_bolt/manifests/chronos-bolt-tiny-official.json`.
The following example uses that manifest's real model ID, precision, bundle
name, and input values.

### Build the official tiny model

```bash
trtmc build amazon/chronos-bolt-tiny \
  -o /tmp/chronos-bolt-tiny-official.bundle \
  --precision fp32
```

Keep `fp32` for this qualified path. The official manifest remains FP32 because
the FP16 attention path does not meet its framework-reference accuracy
contract.

The build requires the same TensorRT/CUDA GPU environment as the other engine
builds on this page. Chronos-Bolt also declares a build-time Python profile
named `chronos`. On first use, the CLI materializes the pinned dependencies
from
`python/tensorrt_model_connect/families/chronos_bolt/python_profile_requirements/chronos.lock.txt`,
including `chronos-forecasting==2.2.2`. That step needs package access or a
pre-populated profile/cache. A successful build exits with status 0 and creates
`/tmp/chronos-bolt-tiny-official.bundle`.

### Forecast from the manifest input

```bash
trtmc solve /tmp/chronos-bolt-tiny-official.bundle \
  --branch-input "100.1,100.15,100.18,100.22,100.21,100.27,100.31,100.35,100.37,100.4,100.44,100.5"
```

`--branch-input` is a comma-separated, univariate history. Here it contains the
12 ordered observations from the official E2E manifest. Chronos-Bolt consumes
that history directly, so this model contract does not use `--trunk-input`.

The native `chronos_bolt_trt` runtime flattens the model's quantile forecast
into the public result vector. Success is an exit status of 0 and one line in
this form:

```text
Output [N]: <N floating-point values>
```

`N` is the bundle's output dimension. The values after the colon are the
forecast output, not another input to copy. Once the bundle exists, `solve`
runs through the native C++/TensorRT runtime; it does not invoke the
build/reference Python profile. The runtime machine still needs a compatible
NVIDIA GPU, TensorRT, CUDA, and the model runtime DSO.

## Segmentation and detection mental model

Segmentation and detection are covered by feature and E2E docs, but the runtime shape is similar to other vision pipelines:

<Diagram
  src="/img/diagrams/tutorials/intermediate/segmentation-detection-pipeline.svg"
  alt="Vision task pipeline from image preprocessing through engine execution and model-owned mask or box decoding"
  caption="This is the shared API shape. Current checked-in model qualification covers the named segmentation owners below, not a model-owned object-detection strategy."
/>

Current segmentation owners use qualified strategies: SegFormer emits
`segformer_segmentation`, while SAM and SAM3 emit
`sam_prompted_segmentation` and `sam3_prompted_segmentation`. The public API
and CLI also expose `detect()`, but the current model descriptors do not claim
an object-detection runtime strategy. Do not present that API surface as a
supported model until a model-owned descriptor and E2E manifest provide the
evidence.

## What you should understand now

- Not every model returns text.
- `IPipeline` provides task-specific methods so users do not manipulate raw engine tensors.
- Iterative inference can mean token decode, diffusion denoising, streaming audio chunks, or another task-specific loop.
- Time-series models can use `solve()` to map a numeric history to a forecast vector without a text or image interface.
- E2E manifests are the safest way to find canonical inputs for non-text models.

## Self-check

1. What replaces token-by-token decode in a diffusion pipeline?
2. Why is the number printed by `Output [N]:` not another input parameter?
3. Does the presence of `detect()` prove a detector model is supported?

<details>
<summary>Check your answers</summary>

1. A scheduler-controlled denoising loop repeatedly updates latent tensors and
   then decodes them into pixels or frames.
2. `N` is the returned forecast vector dimension packaged by the model
   contract; the following values are outputs.
3. No. Support requires a model-owned detection runtime strategy, descriptor,
   exact manifest, and appropriate passing evidence.

</details>

{/* Collaborative review anchor: batch 2. */}
