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
| `bark-large` | `bark-small-fp32-l0` | Smaller full-precision Bark checkpoint. |
| `bark-large-tp4` | `bark-small-tp4` | Smaller Bark checkpoint while retaining the tensor-parallel path. |
| `deepseek-ocr` | `deepseek-ocr-l0` | Same checkpoint; shorter OCR decode. |
| `deepseek-v2-lite` | `deepseek-v2-tiny` | Tiny DeepSeek-V2 checkpoint. |
| `flux-2-dev` | `flux-2-dev-l0` | Same checkpoint; 384px, 20-step image run. |
| `flux-2-dev-fp8` | `flux-2-dev-fp8-l0` | Same checkpoint and FP8 scales; 384px, 20-step image run. |
| `flux-schnell` | `flux-schnell-l0` | Same checkpoint; 384px, 20-step image run. |
| `glm-4-9b` | `glm-4-9b-l0` | Same checkpoint; shorter decode. |
| `gpt-oss-20b` | `gpt-oss-20b-l0` | Same checkpoint; shorter decode. |
| `internvl3-8b` | `internvl3-2b` | Smaller InternVL3 checkpoint. |
| `internvl3-8b-tp4` | `internvl3-2b-tp2` | Smaller InternVL3 checkpoint while retaining a tensor-parallel VL path supported by its KV-head count. |
| `minitron-4b-width` | `minitron-4b-width-l0` | Same checkpoint; shorter decode. |
| `mistral-7b` | `mistral-7b-l0` | Same checkpoint; shorter decode. |
| `nemotron-labs-diffusion-8b` | `nemotron-labs-diffusion-8b-l0` | Same checkpoint and runtime; reduced generation-mode coverage for PR L0. |
| `personaplex-7b` | `personaplex-7b-l0` | Same checkpoint; shorter speech generation. |
| `phi-moe` | `phi-moe-l0` | Same checkpoint; shorter decode. |
| `pixart-sigma-1024` | `pixart-sigma-1024-l0` | Same checkpoint; 512px image run. |
| `qwen-image` | `qwen-image-l0` | Same checkpoint and paired image contract at reduced spatial scale. |
| `qwen-image-2512` | `qwen-image-l0` | Shared Qwen-Image build/runtime path at reduced spatial scale. |
| `qwen-image-edit-2511` | `qwen-image-l0` | Shared Qwen-Image build/runtime path with the minimum paired-image proof. |
| `qwen3-moe-30b-a3b` | `qwen3-moe-tiny-random` | Tiny Qwen3-MoE checkpoint; comparison skipped. |
| `wan21-t2v-1.3b` | `wan21-t2v-1.3b-l0` | Same checkpoint; 384x672, 5-frame, 15-step video run. |
| `wan22-ti2v-5b` | `wan22-ti2v-5b-l0` | Same native video path; reduced resolution, frame count, and denoising steps. |
| `z-image-turbo` | `z-image-turbo-l0` | Same checkpoint; 512px image run. |

This table is a checked snapshot of non-self replacements declared by
`testcases[*].l0_replacement`. The manifests remain authoritative; run
`PYTHONPATH=python:. python3 tools/test_impact.py --validate` after changing a
replacement.
