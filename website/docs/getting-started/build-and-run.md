---
title: Build and Run
---

All examples use `$TRTMC` for the unified CLI. Set it to `trtmc` when using a
release wheel, or `./build/trtmc` when using a source build inside the dev
container.

```bash
TRTMC=trtmc
# Source build alternative:
# TRTMC=./build/trtmc
```

Run [Installation](installation.md) or
[Environment and First Repro](environment-and-repro.md) first if either command
fails.

## Text generation

```bash
$TRTMC build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --precision fp16

$TRTMC run /tmp/qwen3.trtfb \
  --prompt "Write one sentence about TensorRT." \
  --max-new-tokens 32 \
  --temperature 0.7 \
  --top-p 0.9
```

Useful runtime knobs include `--greedy`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--chat-template`, and `--no-thinking`. Use `--greedy` for deterministic smoke tests before trying sampling. Qwen text generation uses the native C++ BPE tokenizer from the bundle, so `--hf-python` is not needed for this path.

## Vision-language generation

```bash
$TRTMC build Qwen/Qwen2.5-VL-3B-Instruct \
  -o /tmp/qwen25vl.trtfb \
  --precision fp16 \
  --max-cache-length 384

$TRTMC run /tmp/qwen25vl.trtfb \
  --prompt "Describe this image." \
  --image tests/assets/test_image.jpg \
  --max-new-tokens 48
```

This Qwen-VL bundle routes through the model-owned
`runtime_strategy="qwen_vl_vision_language"`. Other vision-language families
use their own strategy keys and DSOs even when they implement the same public
`generate(prompt, image, ...)` task shape.

## Speech and audio

```bash
$TRTMC build openai/whisper-large-v3-turbo -o /tmp/whisper.trtfb --precision fp16

$TRTMC transcribe /tmp/whisper.trtfb \
  --audio tests/e2e/models/whisper/data/Recording.wav \
  --max-new-tokens 224
```

```bash
$TRTMC build nvidia/magpie_tts_multilingual_357m -o /tmp/magpie.trtfb --precision fp16

$TRTMC generate-audio /tmp/magpie.trtfb \
  --prompt "A clear short test sentence." \
  --output /tmp/magpie.wav
```

Streaming paths are exposed through `trtmc transcribe --stream` for cache-aware ASR and `trtmc serve-audio` for prompt-driven audio serving. Add `--hf-python /opt/venv/bin/python` only for runtime strategies that still need helper Python code.

## Diffusion and video

```bash
$TRTMC build black-forest-labs/FLUX.2-dev \
  -o /tmp/flux2.trtfb \
  --precision fp16 \
  --image-height 1024 \
  --image-width 1024 \
  --num-inference-steps 28

$TRTMC generate-video /tmp/flux2.trtfb \
  --prompt "A photo of a cat sitting on a windowsill at sunset" \
  --output /tmp/flux2-frames \
  --num-steps 28
```

Image diffusion and video diffusion use separate runtime strategies, but both use the same bundle and C++ runtime entrypoint family.

## Segmentation

This example follows the real
`tests/e2e/models/segformer/manifests/segformer-b0-ade.json` manifest from model
ID through inference:

```bash
$TRTMC build nvidia/segformer-b0-finetuned-ade-512-512 \
  -o /tmp/segformer-b0-ade.trtfb \
  --precision fp16

$TRTMC segment /tmp/segformer-b0-ade.trtfb \
  --image tests/e2e/models/segformer/data/test_img.jpeg \
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
`tests/e2e/models/chronos_bolt/manifests/chronos-bolt-tiny-official.json`:

```bash
$TRTMC build amazon/chronos-bolt-tiny \
  -o /tmp/chronos-bolt-tiny-official.trtfb \
  --precision fp32

$TRTMC solve /tmp/chronos-bolt-tiny-official.trtfb \
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
