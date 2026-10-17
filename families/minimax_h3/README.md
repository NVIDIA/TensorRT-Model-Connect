# MiniMax-H3 image Ref2VA: experimental integration

The image-reference path is implemented but **not qualified end to end** with
the complete trained model. Do not advertise it as production support,
quality-equivalent inference, or real-time generation. Automatic discovery in
`support.py` remains unchanged. A passing factory/host test does not execute or
qualify the eight inference components.

This path requires the original `transformer_ref` checkpoint partition. A
text-only checkpoint, including a text-only FastH3 checkpoint, is not a
replacement. It is image-reference conditioning, not initial-frame/video
continuation, and does not enable four-step reference inference.

## Explicit experimental build

Use a complete local original checkpoint. The trained component oracles pin
`MiniMaxAI/MiniMax-H3` revision
`48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc`; no download is implicit in the
example. Provide an existing output directory and enough disk for both staged
sections and the atomic final bundle. During packaging, peak disk can approach
twice the total serialized plan size. This is separate from GPU residency.

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
    build(request, writer, memory_preset=os.environ.get("H3_MEMORY_PRESET", "minimal"))
    writer.finish()
except BaseException:
    writer.abort()
    raise
```

The family-local entry point is intentional: it exposes the experimental
memory preset without inventing a shared CLI option or changing discovery.
It validates original configs, tokenizer, key partitions and normalization
statistics. It does not establish model correctness merely by producing a
bundle. Supported build settings are batch 1, TP/CP 1 and no quantization;
unsupported overrides fail explicitly.

## Residency presets

| Preset | Module residency | Video tile batch |
| --- | --- | --- |
| `minimal` (default) | Staged; release completed phases | 1 |
| `original` | Resident modules/cache | 28 |

Both preserve original trained partitions, reference preprocessing, output
geometry, precision boundaries and 49 denoiser forwards. No resolution, frame
count, step count or model depth is reduced to save memory. However, neither
end-to-end preset equivalence nor cross-preset numerical identity is currently
qualified; changing batch size can change TensorRT tactics and numerical
results. Do not interpret the presets as a proven quality/speed tradeoff.

Plans are loaded one at a time from the bundle. The largest plan still incurs
a transient CPU read buffer. Minimal residency also does not remove the full
FP32 video result: default geometry alone needs 1,535,901,696 bytes for that
buffer. There is no qualified minimum RAM/VRAM figure for the complete path.

## Bundle composition

The `minimax_h3.image_ref2va/v1` metadata schema identifies these eight plans:

| Plan | Role |
| --- | --- |
| `vision.plan` | Qwen image features |
| `text_encoder.plan` | Reference-aware language conditioning |
| `vae_encoder.plan` | Image-reference VAE moments |
| `vae_reference_posterior.plan` | Weightless reference sampling/normalization |
| `adaln.plan` | Timestep-dependent conditioning |
| `denoiser.plan` | Original reference-conditioned joint A/V denoising |
| `vae.plan` | Video decoding |
| `audio_vae.plan` | Audio decoding |

`tokenizer.json` and `runtime.json` complete the bundle. The real family
factory recognizes its explicit reference Task header and validates metadata
before loading engines. The legacy text-only bundle path is separate.

## Native request/output contract

The owner exposes `IReferencesTextToAudioVideo` v1.0. Reference images must be
contiguous RGB uint8; audio/video references and initial-frame semantics are
not supported by this implementation. Capacity is checked after original
image preparation; oversized inputs fail rather than being silently resized
to fit the engine profile. Request seed is nonnegative, default 42; reference
posterior sampling separately keeps the original seed 42 per image.

The fixed recipe uses 50 schedule points / 49 denoiser forwards, producing
124 video frames at 24 fps and stereo 32-kHz audio. Default portrait output is
1344 high by 768 wide (4:7, not 9:16). An explicit 1536-by-864 build gives
9:16 geometry, but trained output at that geometry remains unqualified.
The original audio/video endpoint difference is not silently time-stretched.

## Validation boundaries

Family CPU tests and ownership checks:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/minimax_h3/tests --ignore=families/minimax_h3/tests/test_e2e.py \
  -m 'not gpu and not trt' -q
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
```

The owning full trained-video gate is
`tests/test_trained_video_decoder.py`. It compares every pixel of all 124
FP32 output frames against the pinned original reference at unchanged
`rtol=1e-3`, `atol=1e-5`, and checks cold/warm behavior for both presets.
Prefix diagnostics, synthetic trajectory tests and guarded factory fixtures
cannot replace that gate or qualify full reference-conditioned generation.
The complete trained reference encoder, Qwen conditioning, DiT trajectory,
video/audio composition and actual native reference request must all be
verified before advertising end-to-end support.
