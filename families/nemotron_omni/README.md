# Nemotron Omni

This family owns `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`,
pinned at `16993199e436da4ba75ddc410855f87e0d996ee6`. The official Edge-LLM
0.11.0 support row lists this prepared checkpoint, whose language projections
use both NVFP4 and FP8. The adapter preserves their packed values and scales;
it does not recalibrate or claim additional checkpoint formats.

## Build and installation

```sh
python -m tensorrt_model_connect build /path/to/Nemotron-Omni \
  --family nemotron_omni --task vision_language_generation \
  --precision fp16 --quantization nvfp4 --max-sequence-length 4096 \
  -o nemotron-omni.bundle
```

Build on the executing GPU; there is no cross-compilation. Optional Edge-LLM
0.11.0 uses the existing CMake SDK installation manifest discovered through
`CMAKE_PREFIX_PATH`. Compatible wheel-installed Python builder tools are used
when available; otherwise the installed SDK's Python tools are used. The C++
SDK and plugin must match the executing TensorRT/CUDA installation. A Python
wheel alone does not replace the native SDK.

Without a compatible SDK, the family builds native TensorRT graphs. Failed Edge
preparation retains its diagnostic log, warns, and retries the unchanged native
request once. Bundle-publication errors and inference errors do not silently
switch backends. All selection is family-owned.

The native bundle contains language, image-patch, temporal-video-patch, visual
encoder and acoustic encoder engines. Graph construction uses TensorRT Python;
inference and preprocessing use the Model Connect C++ runtime, without ONNX or
custom GPU kernels. The Edge bundle contains the official language/visual
engines and their external checkpoint weights, so the original checkpoint
directory is not needed after publication. At runtime, Edge extracts these files
into a temporary directory and removes it after the model is destroyed. Set
`TMPDIR` to a disk-backed directory with space for the bundled engines and
checkpoint (about 24 GB for the exercised bundle), in addition to the bundle
itself. The adapter checks available space before extraction; avoid a small
RAM-backed `/tmp`.

## Modalities and execution

- Native: text, image, audio, real timestamped video, image/audio and audio/video
  inputs through the corresponding public Model Connect task interfaces.
- Model Connect Edge 0.11.0 offload is limited to text/image tasks. Audio and
  typed-video task builds select the complete native implementation directly,
  before attempting Edge preparation; they do not wait for a command failure.
- `--task audio_video_text_to_text` selects a native bundle with all six native
  task bindings. FP32 computation also selects native; FP16 is the exercised
  full-model profile. Source prepared quantization remains unchanged.
- One request and one device; tensor/context parallelism and speculative modes
  are not advertised. No family GPU allowlist is imposed.
- Native PCM is finite interleaved floating-point audio. The declared default
  sample rate is 16 kHz; explicit rates are resampled and channels averaged.
  Acoustic capacity is 1600 feature frames; oversized requests are rejected,
  never truncated.
- Native video takes actual frames with explicit, ordered, nonnegative
  timestamps. Temporal patches consume frame pairs; odd last frames repeat.
  Checkpoint-defined spatial processing and video pruning are preserved.
- Aligned audio/video currently requires an explicit common zero origin.
  Arbitrary audio offsets and missing timing are rejected rather than silently
  discarded. This is not a claim of general synchronization support.

## Validation and known limitations

Prepared native projections still require TensorRT support for their source
quantization operations. TensorRT 11.1 rejects even a small FP8 Q/DQ projection
on an A30; the NVFP4 path contains FP8 block scales and reaches the same guard.
This is an observed TensorRT datatype/hardware limitation, not a family GPU
allowlist, an Edge-only defect, or an out-of-memory result. On an RTX 4000 Ada,
both the same FP8 projection and the NVFP4 projection built and produced the
exact independent expected output. TensorRT warned that the hardware does not
accelerate FP4 compute, but executed that graph successfully. These small
projections neither qualify the full model nor prove its memory requirements
on either GPU; they do rule out a blanket Blackwell-only native restriction.

The native full-checkpoint build succeeds and all six public task modes execute.
The original free-form description cases failed the independent normalized
edit-distance gate of 0.15 for image, video and image/audio. With the corrected
Mamba gating precision, their distances were approximately 0.445, 0.825 and
0.517. Those accuracy-only failures remain runnable and recorded; they are not
reclassified as passing descriptions.

The owning E2E now uses human-reviewed factual questions for those three cases:
vase colors in image order, road direction, and vase colors plus verbatim speech.
Their golden answers were regenerated with the same pinned independent vLLM
reference before executing Model Connect, and checked against the unchanged
fixtures. Text, audio and audio/video questions and goldens are unchanged, as
are all six modality routes, decoding settings and the 0.15 threshold. A replay
using the corrected native decoder and retained media plans passes all six:
text/image/audio/video match exactly, image/audio has distance 0.0137, and
audio/video 0.0566. This is scoped grounding/transcription evidence, not proof of
free-form caption equivalence or a fresh full-builder run of every source change.
The declared premerge case still performs the full public build and inference.

The owning component/configuration tests pass on the repository's existing
Transformers 5.2 dependency. The public Model Connect Edge build also succeeds,
and both text and image tasks executed using an actual Edge-backed bundle.
Text matched the original reference exactly; the original free-form image case
exceeded the same 0.15 gate (approximately 0.735) and remains runnable. The revised
factual image question has not been rerun through Edge. A separate prompt-only
diagnostic matched all six original native prompt token sequences against the
saved independent reference, including media placeholder counts. That rules out
prompt formation on those inputs, not numerical media-feature or decoder
differences; it is not a generation test.

There is a concrete upstream visual-position contract discrepancy: official
Edge 0.11 and checkpoint RADIO use bilinear `align_corners=True`, which native
preserves, while vLLM 0.20 uses `False`. A private same-FP16 component diagnostic
reduced image/video relative feature error from about 8.3%/11.3% to 0.35%/0.48%
when only the reference position interpolation was aligned. That localizes a
major visual-feature difference, not the sole cause of generated-text failures.
The interpolation semantics and numerical gates remain unchanged pending
model-owner confirmation of the authoritative contract. Original free-form
outputs and their failures remain recorded alongside the revised factual tests.

The direct official Edge builder completed language, visual and audio engines.
Against the independent provider-documented vLLM 0.20.0 reference, text matched
exactly. The image command completed but exceeded the unchanged normalized
edit-distance limit of 0.15. That accuracy-only failure remains runnable.
Audio execution failed with TensorRT CASK6 initialization; the same engine
reproduced the failure directly in TensorRT without Model Connect or Edge
runtime orchestration. Typed video failed Edge's placeholder-count check.
The independent reference completed all six input modes. These are scoped
observations, not catalog-wide or all-platform qualification.

The owning full-checkpoint tests retain exact independent outputs, pinned input
revisions, lossless real frames and reference parameters in `tests/data/reference.json`. Local E2E requires explicit selection; the native full-model profile is the
declared premerge case, alongside the small native component tests:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/nemotron_omni/tests --e2e-model nemotron_omni \
  --e2e-testcase nemotron-omni-nvfp4-native -q
```

Set `TRTMC_RUNTIME_ROOT` to the compiled native runtime containing the owning
`nemotron_omni_inference_test` target. `TRTMC_NEMOTRON_OMNI_MODEL_DIR` and
`TRTMC_NEMOTRON_OMNI_ASSET_DIR` optionally reuse the pinned local checkpoint and
public fixture assets. Accuracy failures remain test failures, not relaxed
thresholds or expected passes. The six-request audio/video fixture tests both
modalities with separate zero-origin clips, not physical synchronization.

To independently replay the saved oracle in a separate vLLM 0.20.0 environment:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m \
  families.nemotron_omni.tests.record_reference \
  --model-dir /path/to/Nemotron-Omni --output /path/to/new-reference.json
```

The replay command writes a result only after every request succeeds, refuses
to overwrite previous results, and never updates the checked-in oracle
automatically. It records the supplied local checkpoint path as unverified and
reports the expected model/revision separately; a directory name is not proof
of checkpoint identity. Running it is not necessary for every test.
