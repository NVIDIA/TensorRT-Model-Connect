# MiniMax-H3 Turbo LoRA (experimental)

This opt-in variant builds native TensorRT-RTX graphs with an unmerged,
strength-1 Turbo LoRA. Generation runs in ModelConnect C++; Python is used only
for offline checkpoint preparation and engine construction. The original
non-Turbo workflows are unchanged.

This change is based on the audiovisual runtime in PR #1241. It is not yet
migrated to the newer `main` architecture. Full-size packed-bundle deployment,
clean-machine installation, model parity and comparative performance remain
unqualified. No speedup or quality-equivalence claim is made.

## Build

Follow the [MiniMax-H3 setup guide](../../website/docs/models-recipes/minimax-h3.md)
for the native runtime, TensorRT-RTX SDK and auxiliary checkpoint. Use separate
output paths for each base precision; existing bundles do not gain new engines
from a runtime update.

```powershell
python -m tensorrt_model_connect build '<auxiliary-checkpoint>' `
  --backend trt_rtx --precision bf16 --output '<turbo-bf16.bundle>' `
  --set minimax_h3.turbo=true
```

This builds T2VA and FL2VA plans. To select the full INT8 ConvRot base, add
`--set minimax_h3.turbo_base_precision=int8` and use a different output path.
The outer `--precision bf16` remains unchanged: adapter factors and graph
activations remain BF16. The base defaults to BF16.

The builder resolves the following public sources. File overrides are
`minimax_h3.turbo_transformer`, `minimax_h3.turbo_text_encoder`, and
`minimax_h3.turbo_lora`.

| Component | Repository / revision | File |
|---|---|---|
| BF16 base | `Comfy-Org/MiniMax-H3` / `7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1` | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` |
| INT8 base | `Comfy-Org/MiniMax-H3` / `4cc1d817b6184899b41293954329f576cb5ae86b` | `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` |
| BF16 text encoder | Same repository, BF16 revision above | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| Turbo adapter | `larryvrh/MiniMax-H3-Turbo-Lora` / `43a74557ac3f6539db8e0f2a959d03feb7a81480` | `minimax_h3_turbo_v4_step600_ema.safetensors` |

For experimental REF2VA, additionally supply
`--set minimax_h3.turbo_ref_transformer='<reference-checkpoint.safetensors>'`.
Use the distinct full REF2VA base from the same repository and matching
precision/revision above: `diffusion_models/minimax_h3_ref2va_bf16.safetensors`
or `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors`. This optional
override is an existing local file, not an automatic reference-model download.
It adds genuine reference conditioning and REF denoiser engines; it does not
substitute the FL2VA model. Applying this adapter to REF2VA is experimental and
is not certified by the adapter author.

## Run

Use eight steps, guidance 1 and no FirstBlockCache. The same C++ entrypoint
selects the mode from the supplied inputs:

```powershell
$Prompt = Get-Content '<prompt.txt>' -Raw -Encoding utf8

# T2VA
trtmc.exe generate-video '<turbo.bundle>' --runtime-root '<runtime-bin>' `
  --prompt $Prompt --num-frames 124 --width 1344 --height 768 `
  --num-steps 8 --guidance-scale 1 --seed 42 `
  --runtime-cache '<turbo.rtxcache>' --output '<text-video.mp4>'

# FL2VA: one endpoint or both
trtmc.exe generate-video '<turbo.bundle>' --runtime-root '<runtime-bin>' `
  --prompt $Prompt --first-frame '<first.png>' --last-frame '<last.png>' `
  --num-frames 124 --width 1344 --height 768 --num-steps 8 `
  --guidance-scale 1 --seed 42 --output '<keyframe-video.mp4>'

# REF2VA: requires the optional reference-model engines
trtmc.exe generate-video '<turbo-ref.bundle>' --runtime-root '<runtime-bin>' `
  --prompt $Prompt --reference-image '<reference.png>' `
  --num-frames 124 --width 1344 --height 768 --num-steps 8 `
  --guidance-scale 1 --seed 42 --output '<reference-video.mp4>'
```

## Contract and limits

- Turbo uses dual-clock Euler sampling: eight forwards, nine sigma points,
  video shift 12, audio shift 3, and independent seeds `seed` / `seed + 1`.
  There is no separate CFG branch or Turbo super-resolution path.
- LoRA operates on the original activation and is never merged into the base.
  INT8 base projections use native quantized layers; the BF16 adapter branch
  does not consume ConvRot's rotated input.
- Prompts and video shapes remain dynamic. T2VA/FL2VA share a 2641-row text
  capacity, including image tokens. REF uses separate multimodal profiles;
  actual supported inputs remain bounded by bundle profiles and available memory.
- Native frame alignment is preserved. At 24 fps, 124 / 345 frames are
  5.167 / 14.375 seconds. Turbo also accepts 362 frames and a 1280x736 canvas;
  362 frames are 15.083 seconds. No silent crop, interpolation or frame duplication
  is performed to force an exact duration.
- Text weights come from the BF16 checkpoint; language computation and output
  bindings are FP32 through layer 50, with final conditioning rounded to BF16
  in C++. No chat template, final norm or output head is applied.
- Partitioned text, AdaLN and denoiser engines retain all model layers.
  REF text continuation engines use the same workspace policy as the first
  segment. Reserve storage for source checkpoints, staged plans and the bundle.
- Equal seeds do not imply bitwise equality with the author's runtime: noise
  rounding, decoder execution and conditioning precision can differ. CPU/mock
  tests establish contracts, not full-model accuracy or deployment qualification.

Rebuild the matching runtime and bundle together. Keep this variant experimental
until fresh packed-bundle generation and representative quality checks complete.
