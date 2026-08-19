---
title: Model Recipes
---

This page is an optional task index after the
[Quick Start](quick-start.md), not a second Getting Started path. Each model
can add its own dependencies, memory requirements, and qualified hardware
boundary.

If the command itself is unavailable, return to
[System Requirements](environment-and-repro.md) and
[Installation](installation.md). If you have not yet built and run the Qwen
first-inference bundle, complete the Quick Start before choosing a recipe here.

## Text generation

The canonical build-inspect-run sequence lives only in the
[Quick Start](quick-start.md). Continue with the
[Text Generation tutorial](../tutorials/beginner/text-generation.md) to learn
sampling, deterministic decoding, chat templates, and other request-time
controls without repeating setup.

## Vision-language generation

```bash
trtmc build Qwen/Qwen2.5-VL-3B-Instruct \
  -o /tmp/qwen25vl.bundle \
  --precision fp16 \
  --max-cache-length 384

trtmc run /tmp/qwen25vl.bundle \
  --prompt "Describe this image." \
  --image tests/assets/test_image.jpg \
  --max-new-tokens 48
```

These length controls have different scopes. `--max-cache-length 384` fixes
the bundle's KV-cache capacity. The Qwen-VL build field
`qwen_vl_decoder.max_prefill_length` defaults to `0`, which uses that cache
length as the prefill-profile maximum; an explicit value is clamped to the
cache length, and `opt_prefill_length` is clamped to the resulting prefill
maximum. `--max-new-tokens 48` is only the request-time decode-loop limit: it
does not resize either build-time profile. After the Qwen-VL cache fills, its
runtime advances with a sliding cache, so earlier context rows are evicted.
See [Configuration and Backends](../features/config-and-backends.md) for the
exact family fields and defaults.

This Qwen-VL bundle routes through the model-owned
`runtime_strategy="qwen_vl_vision_language"`. Other vision-language families
use their own strategy keys and DSOs even when they implement the same public
`generate(prompt, image, ...)` task shape.

## Speech and audio

```bash
trtmc build openai/whisper-large-v3-turbo -o /tmp/whisper.bundle --precision fp16

trtmc transcribe /tmp/whisper.bundle \
  --audio python/tensorrt_model_connect/models/whisper/tests/data/Recording.wav \
  --max-new-tokens 224
```

```bash
trtmc build nvidia/magpie_tts_multilingual_357m -o /tmp/magpie.bundle --precision fp16

trtmc generate-audio /tmp/magpie.bundle \
  --prompt "A clear short test sentence." \
  --output /tmp/magpie.wav
```

Streaming paths are exposed through `trtmc transcribe --stream` for cache-aware ASR and `trtmc serve-audio` for prompt-driven audio serving. Add `--hf-python /opt/venv/bin/python` only for runtime strategies that still need helper Python code.

## Diffusion and video

Follow [Diffusion, Vision, and Time-Series Pipelines](../tutorials/intermediate/diffusion-and-time-series.md)
for FLUX, PixArt-Sigma, Wan, and the hardware-qualified Jetson Thor Wan2.2
recipe. Those workloads are intentionally kept out of the first-inference
path: they have larger artifacts, longer builds, and model-specific profiles.

## Segmentation

This example follows the real
`python/tensorrt_model_connect/models/segformer/tests/manifests/segformer-b0-ade.json` manifest from model
ID through inference:

```bash
trtmc build nvidia/segformer-b0-finetuned-ade-512-512 \
  -o /tmp/segformer-b0-ade.bundle \
  --precision fp16

trtmc segment /tmp/segformer-b0-ade.bundle \
  --image python/tensorrt_model_connect/models/segformer/tests/data/test_img.jpeg \
  --output /tmp/segformer-b0-ade-mask.png
```

The JPEG is a checked-in E2E input, so the path works when the command is run
from the repository root. `segment` loads it as normalized HWC pixels and
writes a grayscale PNG whose pixel values are class indices. Success means the
command exits with status 0, the output PNG exists, and the CLI prints a line
like:

```text
Segmentation saved: /tmp/segformer-b0-ade-mask.png (<width>x<height>)
```

Building the bundle needs the supported TensorRT/CUDA GPU environment and
network access or a cached copy of the NVIDIA checkpoint. Running it needs a
compatible NVIDIA GPU and the `segformer_segmentation` runtime DSO.

The public API and CLI reserve `IPipeline::detect()` and `trtmc detect`, but
the current model manifests and E2E catalog do not include an object-detection
owner. There is therefore no supported detector bundle to run in this guide.
Treat the command as an API contract for a future model implementation, not as
current support evidence.

## Chronos-Bolt time-series forecasting

This build-to-solve example follows
`python/tensorrt_model_connect/models/chronos_bolt/tests/manifests/chronos-bolt-tiny-official.json`:

```bash
trtmc build amazon/chronos-bolt-tiny \
  -o /tmp/chronos-bolt-tiny-official.bundle \
  --precision fp32

trtmc solve /tmp/chronos-bolt-tiny-official.bundle \
  --branch-input "100.1,100.15,100.18,100.22,100.21,100.27,100.31,100.35,100.37,100.4,100.44,100.5"
```

The branch input is the manifest's 12-value, ordered univariate history.
Chronos-Bolt forecasts directly from this context, so its current model
contract does not take `--trunk-input`. Keep FP32 for the officially qualified
path; the manifest notes that the FP16 attention path does not satisfy its
framework-reference accuracy contract.

On its first build, the CLI may materialize the family-owned `chronos` Python
profile, which pins `chronos-forecasting==2.2.2`. The build therefore needs the
TensorRT/CUDA GPU environment plus access to the checkpoint and Python
packages, or populated caches. The resulting bundle runs through the native
`chronos_bolt_trt` C++/TensorRT strategy and does not invoke that Python profile
during `solve`.

Success means both commands exit with status 0, the named bundle exists, and
`solve` prints one line in the form `Output [N]:` followed by `N`
floating-point forecast values. See
[Diffusion, Vision, and Time-Series Pipelines](../tutorials/intermediate/diffusion-and-time-series.md)
for the input/output mental model and dependency boundaries.

{/* Collaborative review anchor: batch 2. */}
