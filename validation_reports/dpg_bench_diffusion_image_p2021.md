# DPG-Bench diffusion image validation on P2021

Date: 2026-07-02

Branch: `codex/dpg-geneval-diffusion-task-eval`

Validated source snapshot: `b78f2d55`
Task-eval pull request: `#385`

Runtime changes used by this validation are reviewed separately:

- Flux/Z-Image caller-provided initial latents: `#388`.
- PixArt Diffusers parity and caller-provided initial latents: `#389`.
- Z-Image Qwen3/DiT parity fix discovered by this suite: `#387` (`#386`).

## Dataset evidence

- DPG-Bench source: ELLA commit
  `3c228f1dc6c4d3cad0a47493816151a419f14db3` (Apache-2.0).
- Processed content: 1,065 prompts and 14,392 proposition questions.
- The first ten records are a fixed manual-review slice; all remaining records
  follow official source order and retain their original source indices.
- DPG JSON SHA-256:
  `cb2028c1c348e84d96ff8059300344bb8f9a756bb0e9710e9542a1b88a181aa3`.
- GenEval supplementary metadata SHA-256:
  `5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13`.
- NAS root:
  `http://dlswqa-nas.nvidia.com:8080/mobile/Safety/mcnnt/trtmc_benchmark_datasets/`.
- P2021 dataset:
  `/localhome/local-chaofengw/trtmc/data/DPG-Bench/dpg_bench.json`.

## Compare contract

- HF and TRTMC receive the same full prompt, seed, dimensions, inference steps,
  and serialized initial latent tensor.
- Prompt CLIPScore is diagnostic only because dense prompts can exceed CLIP's
  77-token limit.
- Every case records PSNR, SSIM, HF/TRT prompt CLIPScores, image-to-image CLIP
  cosine, exact image paths, the full prompt, and the DPG proposition checklist.
- Image-to-image CLIP cosine >= 0.85 is a mandatory semantic parity gate for
  all diffusion families.
- `visual_review.html` presents HF and TRTMC images side by side with the full
  prompt, proposition questions, and all metrics.

## P2021 results

| Model | Executed scope | Result | Evidence and manual review |
|---|---:|---|---|
| `pixart-sigma-1024` | 10 cases | 9/10 pass | Case 2 correctly fails: HF has four 2x2 macarons while TRT has about eight and a different layout. The other nine pairs preserve the subject, scene and composition; cases 7-10 are nearly pixel-identical. |
| `z-image-turbo` | 10 cases | 1/10 pass | Manual review agrees with the semantic gate. Only case 2 preserves the same 2x2 cowboy-hat emoji layout. The other cases change object count, subject, composition, or rendered text. |
| `flux-schnell-l0` | 1-case smoke | runtime blocked | TRT deserialization loads a 47,430,721,180-byte denoiser plan and then fails a 19,116,343,936-byte CUDA allocation on the 48 GB P2021 GPU. No TRT image exists to compare. |
| `flux-2-dev-l0` | build | build blocked | Existing P2021 build reaches about 201,314 MiB CPU use in TensorRT compiler backend and is killed by signal 9 (`rc=137`). No bundle was produced. |
| `flux-2-dev-fp8-l0` | build | build blocked | Same TensorRT build-memory failure as the non-FP8 L0 bundle; no bundle was produced. |
| `qwen-image` | build | build blocked | Full 60-block denoiser build is killed by signal 9 (`rc=137`); no bundle was produced. |
| `qwen-image-2512` | build | build blocked | Same full-denoiser build-memory failure; no bundle was produced. |

The final plan selects all seven models with `diffusion_prompt_json`; build or
runtime blockers are not converted into score failures and are reported
separately from HF/TRTMC parity.

## Evidence paths on P2021

- Final seven-model plan:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/artifacts/plan-b78f2d55.json`
- PixArt summary:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/artifacts/pixart-limit10-a4b25a04/dpg_bench_diffusion_image/eval_summary.json`
- PixArt visual review:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/artifacts/pixart-limit10-a4b25a04/dpg_bench_diffusion_image/pixart-sigma-1024/visual_review.html`
- Z-Image summary:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/artifacts/z-image-limit10-b78f2d55/dpg_bench_diffusion_image/eval_summary.json`
- Z-Image visual review:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/artifacts/z-image-limit10-b78f2d55/dpg_bench_diffusion_image/z-image-turbo/visual_review.html`
- FLUX Schnell CUDA OOM log:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/artifacts/flux-schnell-smoke-a4b25a04/dpg_bench_diffusion_image/flux-schnell-l0/trtfb_artifacts/dpg_bench_000004/end_to_end_stderr.log`
- Flux2 and Qwen build logs:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/partiprompts-diffusion-image/artifacts/all-models-limit10-8f44795a/partiprompts_diffusion_image/<model>/build.log`

## Build and test evidence

- P2021 isolated source:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/checkout-b78f2d55`
- P2021 runtime build:
  `/localhome/local-chaofengw/trtmc/ai-loop/diffusion-task-eval-2026-06-30/dpg-bench-diffusion-image/runtime-build-579b5f40`
- The source snapshots are read-only; runtime, bundles and artifacts use
  separate directories.
- Local validation:
  `pytest -q tests/tools tests/e2e/test_diffusion_image_parity_inputs.py tests/e2e/models/flux/test_flux_hf_diffusers_reference.py tests/e2e/models/pixart/test_pixart_hf_reference.py`
  passed: 1,276 tests after rebuilding the task-eval branch on current `main`.
