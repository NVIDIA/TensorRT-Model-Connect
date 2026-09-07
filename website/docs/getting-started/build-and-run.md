---
title: Model Recipes
---

This page is an optional task index after the [Quick Start](quick-start.md),
not a second setup path. Each recipe follows the same boundary:

1. install the selected family's `requirements.txt` when it has one;
2. build with `python -m tensorrt_model_connect build`;
3. inspect with native `trtmc inspect`; and
4. execute the matching Task command with an explicit `--runtime-root`.

Set the runtime directory once for the examples below:

```bash
TRTMC_RUNTIME_ROOT=/opt/trtmc/lib
```

## Text generation

The canonical text build and run sequence lives in
[Quick Start](quick-start.md). Continue with the
[Text Generation guide](../user-guides/text-generation.md) for deterministic
and sampled request controls.

## Vision-language generation

This recipe follows the model-owned Qwen-VL manifest and checked-in image:

```bash
python -m tensorrt_model_connect build Qwen/Qwen2.5-VL-3B-Instruct \
  --precision fp32 \
  --max-sequence-length 384 \
  --output /tmp/qwen25vl.bundle

trtmc inspect /tmp/qwen25vl.bundle

trtmc run /tmp/qwen25vl.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "What color is the vehicle in this image? Answer in one word." \
  --image families/qwen_vl/tests/data/test_img.jpeg \
  --max-new-tokens 10 \
  --temperature 1 \
  --top-k 1
```

The family owns image preprocessing, engine layout, cache behavior, and output
decoding. Request limits do not resize an engine built for a smaller profile.

## Speech recognition

```bash
python -m tensorrt_model_connect build openai/whisper-tiny \
  --precision fp16 \
  --max-sequence-length 128 \
  --fp32-layer 0 \
  --output /tmp/whisper.bundle

trtmc transcribe /tmp/whisper.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input families/whisper/tests/data/librispeech-test-clean-6930-75918-0003.wav \
  --max-output-tokens 120
```

Use `transcribe-batch` or `transcribe-streaming` only when the selected family
implements the corresponding Task interface.

## Audio generation

```bash
python -m pip install -r families/magpie_tts/requirements.txt

python -m tensorrt_model_connect build nvidia/magpie_tts_multilingual_357m \
  --precision fp16 \
  --output /tmp/magpie.bundle

trtmc generate-audio /tmp/magpie.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "A clear short test sentence." \
  --output /tmp/magpie.wav
```

The requirements file is build/reference setup owned by Magpie. Native bundle
execution does not read that file or start Python.

## Diffusion and video

Image and video families expose `generate-image`, `generate-image-batch`, or
`generate-video` according to their declared task. Build dimensions and frame
count are selected with common build arguments such as `--image-height`,
`--image-width`, and `--video-num-frames`; denoising controls belong to the
request. See [Image & Video Generation](../user-guides/image-video-generation.md).

## Segmentation

```bash
python -m tensorrt_model_connect build nvidia/segformer-b0-finetuned-ade-512-512 \
  --precision fp16 \
  --max-sequence-length 1 \
  --output /tmp/segformer-b0-ade.bundle

trtmc segment /tmp/segformer-b0-ade.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image families/segformer/tests/data/test_img.jpeg \
  > /tmp/segformer-result.json
```

The JSON result contains the mask and its dimensions. Compare it with the
family-owned thresholds before making an accuracy claim.

## Chronos-Bolt forecasting

`forecast` reads little-endian float32 input files. Prepare the same values as
the model-owned smoke case, then build and run:

```bash
python -c 'from array import array; array("f", [100.1,100.15,100.18,100.22,100.21,100.27,100.31,100.35,100.37,100.4,100.44,100.5]).tofile(open("/tmp/chronos-values.f32", "wb"))'

python -m tensorrt_model_connect build amazon/chronos-bolt-tiny \
  --precision fp32 \
  --output /tmp/chronos-bolt.bundle

trtmc forecast /tmp/chronos-bolt.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input /tmp/chronos-values.f32 \
  --frequency 0
```

The exact input layout and output comparison belong to
`families/chronos_bolt/tests/`.

## Where examples fit

Repository examples and `apps/` are one-way consumers of public ModelConnect
APIs. They can compose a family bundle into a larger workflow, but core and
families do not import application code.
