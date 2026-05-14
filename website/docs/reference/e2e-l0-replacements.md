# E2E L0 Replacements

## Selection Guideline

PR L0 may replace a large or slow model only when the replacement preserves the
same E2E contract:

- the source model is slow enough to matter, normally at least about 10 minutes
  in nightly E2E
- the source model mainly adds scale coverage, such as long build time, memory
  pressure, large weights, image or video size, frame count, denoising steps, or
  decode length
- the replacement keeps the same family, runtime strategy, plugin/model path,
  precision or quantization mode, runner, comparator, and artifact contract
- if no smaller checkpoint preserves the path, the L0 representative may keep
  the same checkpoint and reduce only workload knobs
- source manifests stay `nightly_only`; L0-only representatives use
  `ci_tier: "l0_only"`

Nightly remains the scale validation lane. PR L0 applies the configured
replacement for broad changes and for direct changes to a nightly-only model's
manifest or model-specific E2E data.

## L0 Replacement Set

| Nightly-only model | PR L0 replacement | L0 change |
| --- | --- | --- |
| `bark-large` | `bark-small` | Smaller Bark checkpoint. |
| `deepseek-ocr` | `deepseek-ocr-l0` | Same checkpoint; shorter OCR decode. |
| `deepseek-v2-lite` | `deepseek-v2-tiny` | Tiny DeepSeek-V2 checkpoint. |
| `flux-2-dev` | `flux-2-dev-l0` | Same checkpoint; 384px, 20-step image run. |
| `flux-2-dev-fp8` | `flux-2-dev-fp8-l0` | Same checkpoint and FP8 scales; 384px, 20-step image run. |
| `flux-schnell` | `flux-schnell-l0` | Same checkpoint; 384px, 20-step image run. |
| `glm-4-9b` | `glm-4-9b-l0` | Same checkpoint; shorter decode. |
| `gpt-oss-20b` | `gpt-oss-20b-l0` | Same checkpoint; shorter decode. |
| `internvl3-8b` | `internvl3-2b` | Smaller InternVL3 checkpoint. |
| `minitron-4b-width` | `minitron-4b-width-l0` | Same checkpoint; shorter decode. |
| `mistral-7b` | `mistral-7b-l0` | Same checkpoint; shorter decode. |
| `personaplex-7b` | `personaplex-7b-l0` | Same checkpoint; shorter speech generation. |
| `phi-moe` | `phi-moe-l0` | Same checkpoint; shorter decode. |
| `pixart-sigma-1024` | `pixart-sigma-1024-l0` | Same checkpoint; 512px image run. |
| `qwen3-moe-30b-a3b` | `qwen3-moe-tiny-random` | Tiny Qwen3-MoE checkpoint; comparison skipped. |
| `wan21-t2v-1.3b` | `wan21-t2v-1.3b-l0` | Same checkpoint; 384x672, 5-frame, 15-step video run. |
| `z-image-turbo` | `z-image-turbo-l0` | Same checkpoint; 512px image run. |
