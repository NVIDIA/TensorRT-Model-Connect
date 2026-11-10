# MiniMax-H3 reference-to-video (experimental)

This explicit build turns reference images and a prompt into video with stereo
audio through the original `MiniMaxAI/MiniMax-H3` image-reference path (Ref2VA),
124 frames per request.

It is experimental and not qualified end to end with the complete trained
model. Automatic discovery in `support.py` does not select it; do not present it
as production support or quality-equivalent inference. It needs the original
`transformer_ref` checkpoint partition; a text-only checkpoint does not replace
it. Initial-frame editing is not supported.

## Build from Python

With `core/builder` and the repository root on `PYTHONPATH`:

```python
import os
from pathlib import Path

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.minimax_h3.reference_build import build

request = BuildRequest(
    model_dir=Path(os.environ["H3_CHECKPOINT"]),
    output_path=Path(os.environ["H3_OUTPUT"]),
    family="minimax_h3",
    task="references_text_to_audio_video",
    precision="bf16",
    backend="trt",
    image_height=1344,
    image_width=768,
    video_num_frames=124,
)
writer = BundleWriter(request.output_path)
writer.record_build(request)
try:
    build(
        request, writer,
        memory_preset=os.environ.get("H3_MEMORY_PRESET", "minimal"),
        reference_capacity=os.environ.get("H3_REFERENCE_CAPACITY", "standard"),
    )
    writer.finish()
except BaseException:
    writer.abort()
    raise
```

The family-local entry point exposes the presets below without adding shared
CLI options or changing discovery. It validates the original configs,
tokenizer, key partitions and normalization statistics, and loads only the
requested tensors from the indexed components and the single-file audio VAE.
Supported settings are BF16, batch 1, TP/CP 1 and no quantization; other
overrides fail. The writer stages every plan once in a single
file with a 64-KiB header and publishes that file atomically; a replaced
destination keeps its space until publication.

## Residency presets

| Preset | Module residency | Video tile batch |
| --- | --- | --- |
| `minimal` (default) | Staged; completed phases released | 1 |
| `original` | Resident modules and caches | 28 |

Both keep the trained partitions, reference preprocessing, output geometry,
precision boundaries and denoiser forwards; neither reduces resolution, frames,
steps or depth. A different tile batch can change TensorRT tactics, so the two
are not proven numerically identical. Plans load one at a time, the largest
still through a transient host buffer, and the FP32 video result at the default
geometry alone takes 1,535,901,696 bytes.

## Reference capacity

`reference_capacity` sets the language and DiT operand profiles at build time:

| Capacity | Text rows, including visual tokens | DiT condition-video rows (min/opt/max) |
| --- | --- | --- |
| `standard` (default) | 32,768 | 4,096 / 5,504 / 16,384 |
| `six_portrait` | 65,536 | 4,096 / 43,776 / 49,152 |

Original preparation maps a 9:16 image to 3648x2048, which adds 7,296 rows to
each conditioning stream, so six distinct portraits need `six_portrait`; it also
fits six images up to 2:1, within the separate text budget. A request that
overflows either budget fails before engines load; references are never resized
or dropped to fit. Editing an old bundle's metadata does not enlarge its
engines: build again with the capacity you need.

## Bundle composition

The `minimax_h3.image_ref2va/v1` schema identifies these eight plans:

| Plan | Role |
| --- | --- |
| `vision.plan` | Qwen image features |
| `text_encoder.plan` | Reference-aware language conditioning |
| `vae_encoder.plan` | Image-reference VAE moments |
| `vae_reference_posterior.plan` | Weightless reference sampling and normalization |
| `adaln.plan` | Timestep-dependent conditioning |
| `denoiser.plan` | Reference-conditioned joint audio/video denoising |
| `vae.plan` | Video decoding |
| `audio_vae.plan` | Audio decoding |

`tokenizer.json` and `runtime.json` complete the bundle. The family factory
recognizes the explicit reference task header and validates the metadata before
loading engines. The text-only bundle path is separate.

## Request and output

The family exposes `IReferencesTextToAudioVideo` v1.0. Reference images are
contiguous RGB uint8; audio and video references are not supported. Capacity is
checked after original image preparation. The request seed is nonnegative,
default 42; reference posterior sampling keeps the original seed 42 per image.

The original recipe uses 50 schedule points (49 denoiser forwards) and produces
124 frames at 24 fps with stereo 32-kHz audio. The default portrait output is
1344 high by 768 wide (4:7); an explicit 1536-by-864 build gives 9:16, untested
with trained output. Audio and video keep their native endpoints;
neither is stretched to match the other.

## Tests

CPU tests and ownership checks:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/minimax_h3/tests --ignore=families/minimax_h3/tests/test_e2e.py \
  -m 'not gpu and not trt' -q
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
```

The GPU tests compare TensorRT components and native execution with the
upstream implementation. They read `TRTMC_MINIMAX_H3_MODEL_DIR`, a local snapshot
of the original checkpoint, and `TRTMC_RUNTIME_ROOT`, the native build
directory. Most use synthetic weights and do not qualify trained output.
`test_trained_video_decoder.py` compares all 124 FP32 frames of the trained
video decoder with the upstream model at `rtol=1e-3`, `atol=1e-5` for both
presets, cold and warm.

Known gap: the trained 256x256 one-frame FP32 encoder comparison in
`test_vae_encoder.py` fails its unchanged `rtol=1.3e-6`, `atol=1e-5` on 54 of
12,288 moments. The trained reference encoder, Qwen conditioning, DiT
trajectory and audio/video composition all still need verification before
either path is advertised.
