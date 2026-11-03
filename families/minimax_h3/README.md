# MiniMax-H3 reference-to-video (experimental)

These explicit builds turn reference images and a prompt into video with stereo
audio:

- the original `MiniMaxAI/MiniMax-H3` image-reference path (Ref2VA), as one
  124-frame window or two windows for 226 frames;
- the `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` student (OmniRef): eight
  denoiser forwards per window with 128-token video sparse attention (VSA).

Both are experimental and not qualified end to end with the complete trained
model. Automatic discovery in `support.py` does not select them; do not present
them as production support, quality-equivalent inference or real-time
generation. They need the original `transformer_ref` checkpoint partition; a
text-only checkpoint, including a text-only FastH3 checkpoint, does not replace
it. Initial-frame editing is not supported.

## Build the OmniRef bundle

`build-omniref` builds the original two-window bundle once, then the student,
which rebuilds its AdaLN and DiT plans and copies the other six:

```sh
python -m tensorrt_model_connect minimax_h3 build-omniref \
  --output h3-omniref.bundle --scratch h3-scratch \
  --bridge build/libtrtmc_byok_tvm_ffi.so \
  --kernel-library build/libtrtmc_minimax_h3_omniref_vsa.so
```

It downloads the pinned revisions of both checkpoints, or takes local snapshots
through `--base` and `--student`, and builds the `minimal` preset with the
`six_portrait` capacity and two windows. Each bundle is about 130 GB: plan for
about 250 GB of disk and 100 GB of host memory while it builds. The original
bundle stays in `--scratch` for the next build.

Configure with `-DTRTMC_ENABLE_BYOK=ON -DTRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA=ON`
for the kernel library, and add `-DTRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA_SM100A=ON`
for the Blackwell cubins (see `kernels/README.md`). A server registers the
kernel before loading the bundle:

```sh
trtmc-server --model video=h3-omniref.bundle \
  --byok-library build/libtrtmc_minimax_h3_omniref_vsa.so \
  --byok-function run --byok-name minimax_h3.omniref_vsa128
```

## Build the original path from Python

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
overrides fail. `continuation=True` selects the two-window profile
(`video_num_frames=226` or unset). The writer stages every plan once in a single
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

The original recipe uses 50 schedule points (49 denoiser forwards) per window
and produces 124 frames at 24 fps with stereo 32-kHz audio. The default portrait
output is 1344 high by 768 wide (4:7); an explicit 1536-by-864 build gives 9:16,
untested with trained output. Audio and video keep their native endpoints;
neither is stretched to match the other.

## Two-window continuation

The 226-frame profile follows the
[fixed vLLM-Omni reference](https://github.com/vllm-project/vllm-omni/blob/cbe4cc4b7c1126a903553f8e79e44ddfab782cf3/vllm_omni/diffusion/models/minimax_h3/continuation.py).
Static image and text positions stay fixed, and the guide and target share one
global media origin. Video guide rows get the original 0.999 noise augmentation;
audio guide rows stay clean at time 1.0. Only target rows are integrated, and a
change in clean-audio presence invalidates the AdaLN cache before the next DiT
load. Image-only requests keep their RNG draw order and outputs.

The window planner uses the 17n+5 frame grid, global rounded audio boundaries
and fractional media rotary offsets. Two 124-frame windows with a 22-frame hidden
overlap give 226 video frames, 67 video latent frames and 377 audio latents per
channel. Window one admits no audio guide rows and window two admits 74; the
video condition maximum adds seven target-grid latent frames to the static image
budget (56,208 condition-video rows and 159,528 packed rows at 1344x768 with
`six_portrait`).

`ImageReferenceDenoiser::run_windows` runs the two windows in sequence with the
request seed for each, keeps every static reference and appends only the
generated latent suffix. The runtime decodes all 67 video latent frames with the
seven-latent tile engine and builds a 377-latent audio decoder for 301,600
samples per stereo channel. Both residency presets apply. Keeping the latent
prefix does not guarantee an unchanged decoded prefix or a seamless visible
boundary.

Only this profile declares the optional runtime string `continuation_prompt`.
Empty or identical text reuses the first window's conditioning; distinct text is
encoded separately with every static reference. Both windows use the longest
complete Qwen text prefix as their media-time origin, as upstream does, so a
longer second prompt also shifts the first window's rotary coordinates. Both
prompts and their operand caps are checked before the first plan loads.

`reference_continuation_build.build(request, writer, source_bundle=...)` builds
this profile from an existing single-window bundle. It rebuilds `denoiser.plan`
and `audio_vae.plan` and copies the six other plans in bounded chunks. Before
loading new weights it checks the source's bundle header and complete section
layout, that its `runtime.json` equals the original profile for this request,
that its provenance names the same pinned checkpoint and an unquantized BF16
build, and that its tokenizer bytes match. The source must remain the same file
while plans are copied, and `minimax_h3.reused_plans.json` lists the reused and
rebuilt sections. Use the source's memory preset and reference capacity; its
plans must suit the selected backend, and an incompatible source fails without
touching it.

## OmniRef student

`reference_pdd.py` owns the sampling contract and build-time output-head math of
`FastVideo/FastVideo-FastH3-Omni-8-Step-V1` at revision
`53e0d26a1691aa473b6262d8196a8feb2c297df0`. The student partitions a
32-interval base grid into eight blocks with separate video and audio shifts.
Each block normalizes and combines its trained output heads in FP32, rounds the
fused parameters to BF16, and keeps H3's separate FP32 score and scheduler
clocks. Target noise starts at 0.999; fixed conditioning rows stay unchanged.

`reference_pdd_builder.py` adds a strongly typed output-head selector: it stores
the eight precomposed banks in BF16, selects one bank per forward, accumulates
the linear operation and bias in FP32 and rounds the velocity to BF16 before
widening it for the scheduler. `reference_vsa.py` defines the 128-token segment
transport, separate reference-video regions, the zero-valid partner tile and the
TVM-FFI graph boundary; `kernels/omniref_vsa.cu` implements routing, sparse
attention and the learned compression branch.

`reference_pdd_dit_builder.py` composes the student DiT: dense text refinement,
one VSA binding per transformer block, target-only output selection and the
eight fused heads. The native denoiser selects the trained head banks, scales
only target noise by the initial sigma, prepares matching AdaLN times and keeps
fixed image, video and audio conditioning. The bundle schemas are
`minimax_h3.omniref_pdd/v1` and `minimax_h3.omniref_pdd_continuation/v1`; the
latter keeps two eight-forward windows with the original overlap and decoding
geometry. The factory requires the released checkpoint and sampling contract and
an explicit VSA tile capacity. Original-H3 schemas keep their 49-forward recipe.

`reference_pdd_build.build` checks both released contracts, the widened-head
transformer config and the complete 688-tensor index before loading student
weights, and rebuilds AdaLN and DiT from the student. It copies the six common
plans from an original-H3 bundle after the same checks as the continuation
build. Only base revisions `9bfb6693f2cf6de171db46d1aa586f67d773a1da` and
`48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc` are admitted; their text encoder,
tokenizer and video/audio VAE files are identical. The source must match the
requested window count and memory preset: a 377-latent audio decoder from a
two-window bundle cannot serve a single window.

The build writes the serialized AdaLN and DiT plans straight into the bundle and
streams the common sections in bounded chunks, avoiding a second plan-sized copy
in Python and another full bundle copy. `denoiser_workspace_bytes` caps only the
DiT's TensorRT tactic scratch memory (default 8 GiB, as for AdaLN); large VSA
fusions can need more. The reuse receipt records both rebuilt plans' limits.

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
directory; the native window tests also read `TRTMC_H3_PACKED_SEQUENCE_SOURCE`,
the upstream vLLM-Omni packed-sequence source they execute as their reference.
Most use synthetic weights and do not qualify trained output.
`test_trained_video_decoder.py` compares all 124 FP32 frames of the trained
video decoder with the upstream model at `rtol=1e-3`, `atol=1e-5` for both
presets, cold and warm.

Known gap: the trained 256x256 one-frame FP32 encoder comparison in
`test_vae_encoder.py` fails its unchanged `rtol=1.3e-6`, `atol=1e-5` on 54 of
12,288 moments. The trained reference encoder, Qwen conditioning, DiT
trajectory and audio/video composition all still need verification before
either path is advertised.
