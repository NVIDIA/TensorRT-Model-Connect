# Nightly E2E Model Fix Plan

> **Implementation goal:** Implement this plan until the GitHub nightly E2E model matrix is in a good state. Use one pull request per model fix. Do not batch unrelated model fixes into the same PR. Keep iterating on each PR until its GitHub CI is green before starting the next model PR.

Ground truth run: <https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/25747071012>

Ground truth commit: `08468945c580d0654010252ccf4c07512eec301c`

Run date: `2026-05-12`

## Operating Rules

- Work from GitHub `main`.
- Use the GitHub remote named `github` for fetch, push, and PR work.
- Do not push directly to `main`.
- Create a short-lived branch for each PR.
- Submit one model fix per PR.
- Merge or abandon the current PR before starting the next model PR.
- Use `$write-git-messages` for every commit message, PR title, PR body,
  squash merge message, and rebase message. If the runtime skill list does not
  expose it, read
  `plugins/trtmc-agent-skills/skills/write-git-messages/SKILL.md` directly.
- If a PR fails CI, inspect the GitHub logs and artifacts, fix the same branch, and rerun until green.
- Do not remove a waive unless the replacement validation is real and the model passes without the waive.
- Do not convert real failures into weaker skips just to make CI green.
- Keep report-only or CI-harness changes separate from model fixes unless the harness change is required to make that exact model test truthful.

## Green Definition

A PR is green only when all of these are true:

- GitHub PR checks pass.
- The focused model E2E passes or has an explicitly justified skip that is visible in the report.
- No new hidden skip, xfail, or xpass is introduced.
- `e2e_artifacts/artifacts/<model>/result.json` exists for every model that ran the orchestrator.
- The HTML report and CI summary tell the same truth as JUnit and console logs.
- If the model produces media, the report contains the expected media artifact and semantic assessment when applicable.
- If a model is still intentionally unsupported, the PR documents that state in the manifest or waive reason and makes it visible in CI output.

## Standard PR Loop

For every PR:

1. Sync to GitHub main.

   ```bash
   git fetch github main
   git switch main
   git merge --ff-only github/main
   ```

2. Create one branch.

   ```bash
   git switch -c fix/<model-or-ci-gap>
   ```

3. Reproduce the issue locally when feasible.

   Use the agent-1 container and run the narrowest useful test first. For a model E2E:

   ```bash
   printf '%s\n' '<model-name>' > /tmp/trtmc-models.txt
   ./scripts/run_e2e_parallel.sh \
     --models-file /tmp/trtmc-models.txt \
     --engine-dir "$ENGINE_DIR" \
     --result-dir /tmp/trtmc-e2e-<model-name> \
     --trtmc-binary ./build/trtmc \
     --hf-python /opt/venv/bin/python \
     --workers-per-gpu 1 \
     --rebuild-engines
   ```

4. Inspect source, logs, and artifacts.

   Required surfaces:

   - `tests/e2e/models/<model>.json`
   - `tests/e2e/waives.txt`
   - `tests/e2e_harness/runners/`
   - `tests/e2e_harness/references/`
   - `tests/e2e_harness/plugins/`
   - `tests/e2e_harness/comparators/`
   - `e2e_artifacts/artifacts/<model>/result.json`
   - `e2e_artifacts/artifacts/<model>/e2e_run.log`
   - Worker console log and JUnit XML

5. Fix the root cause.

   Prefer fixing the model/runtime/reference contract over relaxing thresholds. If thresholds change, prove the new threshold is a meaningful contract and document why.

6. Run focused validation.

   At minimum run relevant unit/tool tests plus the model E2E or a local artifact-level validation when full TRT execution is not available.

7. Draft commit and PR text with `$write-git-messages`.

   Inspect the real diff first:

   ```bash
   git diff --stat
   git diff --name-only
   git diff
   ```

   Use `plugins/trtmc-agent-skills/skills/write-git-messages/SKILL.md` if the
   skill is not available in the active runtime. Commit and PR text must not
   contain `Claude`.

8. Push and open one PR.

   ```bash
   git push -u github HEAD
   gh pr create --base main --head "$(git branch --show-current)" --fill
   ```

9. Watch CI.

   ```bash
   gh pr checks --watch
   ```

10. If CI fails, download the new artifacts, compare against the previous attempt, fix the same branch, and repeat.

11. Merge only after GitHub CI is green. Then sync `main` again before the next PR.

## Required Baseline Audit

Before starting model PRs, download and keep the run `25747071012` artifacts available as the baseline. The original audit found:

- `92` tests scheduled.
- `88` per-model `result.json` files.
- `80` `pass`, `6` `fail`, `2` `skip` from `result.json`.
- One hard JUnit failure: `internvl3-8b`.
- Five xfailed model failures: `albert-base`, `bart-base`, `dpr-ctx-encoder`, `falcon-rw-1b`, `fnet-base`.
- Two skipped result models: `deepseek-ocr`, `nemotron-h-nano-9b`.
- Four scheduled tests skipped before result artifact creation: `gemma-2-2b`, `internlm2-1.8b`, `nllb-200-distilled-600m`, `phi4-multimodal`.
- Two xpass stale waives: `z-image-turbo`, `qwen2.5-0.5b-torchtrt`.
- Seven diffusion models had TRT/HF frame pairs but no `diffusion_vlm_assessment.json`.

## Tracking Checklist

- [ ] PR 0: CI truthfulness prerequisite
- [ ] PR 1: `internvl3-8b`
- [ ] PR 2: `deepseek-ocr`
- [ ] PR 3: `nemotron-h-nano-9b`
- [ ] PR 4: `bart-base`
- [ ] PR 5: `falcon-rw-1b`
- [ ] PR 6: `albert-base`
- [ ] PR 7: `dpr-ctx-encoder`
- [ ] PR 8: `fnet-base`
- [ ] PR 9: `flux-2-dev-fp8`
- [ ] PR 10: `z-image-turbo`
- [ ] PR 11: `qwen2.5-0.5b-torchtrt`
- [ ] PR 12: `gemma-2-2b`
- [ ] PR 13: `internlm2-1.8b`
- [ ] PR 14: `nllb-200-distilled-600m`
- [ ] PR 15: `phi4-multimodal`
- [ ] Final nightly closure check

For every checked item, add the merged PR link, the green CI run link, and the artifact/report evidence next to the checkbox.

## PR 0: CI Truthfulness Prerequisite

This PR is allowed before model PRs because it fixes the CI/report surface that model PRs rely on. Keep it small and do not include model behavior changes.

**Branch:** `fix/ci-e2e-truthfulness`

**Problems:**

- `run_full_e2e` exits before `run_diffusion_vlm_assessment` when any E2E worker fails.
- `generate_ci_summary.py` only reads `result.json`, so it omits tests skipped before the orchestrator writes artifacts.
- XPASS is visible in console logs but not clearly surfaced in the summary.

**Files to inspect:**

- `.github/workflows/nightly.yml`
- `.github/scripts/run-gha-stage.sh`
- `.github/scripts/run-trtmc-ci.sh`
- `scripts/run_e2e_parallel.sh`
- `scripts/generate_ci_summary.py`
- `scripts/generate_e2e_report.py`
- `tests/tools/test_generate_ci_summary.py`
- `tests/tools/test_generate_report.py`
- `tests/tools/test_diffusion_vlm_similarity_tool.py`

**Implementation requirements:**

- Preserve the original E2E exit code.
- Always attempt diffusion VLM assessment after E2E if diffusion frame pairs exist.
- Then return the original E2E failure code.
- Make the summary include scheduled-but-pre-orchestrator skips by reading JUnit or the schedule plus console/JUnit status.
- Add an XPASS/stale-waive section to the CI summary.
- Keep GitHub Pages untouched.

**Validation:**

```bash
python -m pytest \
  tests/tools/test_generate_ci_summary.py \
  tests/tools/test_generate_report.py \
  tests/tools/test_diffusion_vlm_similarity_tool.py \
  -q
```

Generate a local summary from the baseline artifacts and verify it reports `92` scheduled tests, not only `88` result files.

## Model PRs

Implement the following in order. Each item is one PR.

### PR 1: `internvl3-8b`

**Branch:** `fix/internvl3-8b-e2e`

**Current state:**

- Hard CI failure.
- TRT output: `White`.
- HF reference output: the prompt itself.
- Failure: `VL QA answer diverged: NED=1.000`.
- `vision_encode` passed.

**Likely root cause:**

The HF VL reference can decode prompt-only output as the reference answer. That makes the comparison untrustworthy.

**Files to inspect:**

- `tests/e2e/models/internvl3-8b.json`
- `tests/e2e_harness/references/hf_transformers.py`
- `tests/e2e_harness/plugins/vl_qa.py`
- `tests/e2e_harness/runners/vision_language.py`
- `tools/diff_vl.py`

**Fix requirements:**

- Make the HF VL reference reject empty or prompt-only generated text.
- If InternVL needs a model-specific chat template or generated-token slicing rule, implement it in the reference path, not in the comparator.
- Re-run `internvl3-8b`.
- If the corrected HF answer disagrees with TRT, fix the TRT prompt/chat-template/image path behavior.
- Remove no waives because this model is not waived.

**Green criteria:**

- `internvl3-8b` passes as a normal test.
- HF reference output is a real answer, not the prompt.
- The report shows both vision encode and full generation as validated.

### PR 2: `deepseek-ocr`

**Branch:** `fix/deepseek-ocr-e2e`

**Current state:**

- Top-level `skip`.
- `vision_encode` is reported passed even though `diff_vl.py` returned `rc=1`.
- Error: unknown `pad_center_chw` preprocessor, then reshape failure.
- Full generation outputs architecture text, including `Vision: SAM ViT-B + Qwen2 encoder (not supported in TRT yet, text-only)`.
- Reference is `invariant_only`, so OCR text is not validated.

**Files to inspect:**

- `tests/e2e/models/deepseek-ocr.json`
- `tests/e2e_harness/plugins/vl_qa.py`
- `tests/e2e_harness/runners/vision_language.py`
- `tensorrt_model_connect/tensorrt_model_connect/debug_runner.py`
- `tools/diff_vl.py`

**Fix requirements:**

- `vision_encode` must fail when the subprocess has nonzero return code.
- Implement or correctly route the `pad_center_chw` preprocessor.
- Replace invariant-only OCR validation with a real oracle:
  - HF reference if compatible, or
  - a small golden OCR text check against `tests/e2e/data/orc_test_img.jpeg`.
- Ensure TRT output is OCR text, not architecture-description text.
- Remove or update the waive only after real validation exists.

**Green criteria:**

- `deepseek-ocr` is no longer a hidden skip.
- `vision_encode` is truthful.
- OCR full generation is validated against a real expected text contract.

### PR 3: `nemotron-h-nano-9b`

**Branch:** `fix/nemotron-h-nano-9b-e2e`

**Current state:**

- Top-level `skip`.
- Waive says no HF reference because `mamba-ssm` is missing.
- TRT output for "What is the capital of France? Answer in one word." is `<think>\nOkay, the user is`.
- Command includes `--no-thinking`, but output still contains thinking text.

**Files to inspect:**

- `tests/e2e/models/nemotron-h-nano-9b.json`
- `tests/e2e/waives.txt`
- `tests/e2e_harness/plugins/causal_continuation.py`
- `tests/e2e_harness/references/hf_transformers.py`
- Runtime files for `hybrid_mamba_attention`

**Fix requirements:**

- Add a real validation path even if HF parity is unavailable.
- At minimum use a golden-answer contract for a deterministic prompt.
- Enforce `--no-thinking` behavior.
- The model should not pass merely because the binary returned `0`.

**Green criteria:**

- No hidden skip for this model.
- Output does not contain `<think>` when `--no-thinking` is used.
- The answer contract is validated.

### PR 4: `bart-base`

**Branch:** `fix/bart-base-e2e`

**Current state:**

- XFAIL.
- TRT continuation is empty.
- HF continuation is `The capital of France is`.
- Logs show TensorRT/Myelin/CUDA runtime errors around `enqueueV3` and `MyelinCheckException`.

**Files to inspect:**

- `tests/e2e/models/bart-base.json`
- `tests/e2e/waives.txt`
- Seq2seq builder/runtime files
- `tests/e2e_harness/plugins/causal_continuation.py`
- `tests/e2e_harness/references/hf_transformers.py`

**Fix requirements:**

- Root cause the Myelin/runtime error instead of changing the threshold.
- Ensure C++ return code and stderr are treated as failure when TensorRT reports runtime errors.
- Produce a non-empty TRT continuation.
- Remove the XFAIL after the model passes.

**Green criteria:**

- `bart-base` passes without XFAIL.
- No TensorRT runtime error is present in `e2e_run.log`.
- TRT continuation is meaningfully comparable to HF.

### PR 5: `falcon-rw-1b`

**Branch:** `fix/falcon-rw-1b-e2e`

**Current state:**

- XFAIL.
- TRT output repeats `time`.
- HF output is a normal continuation.
- Failure: continuation NED `0.818` with threshold `0.25`.
- Waive points at ALiBi attention numerical divergence.

**Files to inspect:**

- `tests/e2e/models/falcon-rw-1b.json`
- `tests/e2e/waives.txt`
- Falcon builder/runtime family files
- ALiBi attention implementation
- `tests/e2e_harness/plugins/causal_continuation.py`

**Fix requirements:**

- Fix ALiBi/logit parity.
- Add focused unit coverage for the ALiBi path if missing.
- Remove the XFAIL only after E2E passes.

**Green criteria:**

- `falcon-rw-1b` passes without XFAIL.
- Output is not repetitive degeneration.
- Logit or continuation parity is inside the contract threshold.

### PR 6: `albert-base`

**Branch:** `fix/albert-base-e2e`

**Current state:**

- XFAIL.
- Encoder cosine is `0.7065`; enforced floor is `0.8`.
- Existing manifest threshold override is below the minimum contract floor.

**Files to inspect:**

- `tests/e2e/models/albert-base.json`
- `tests/e2e/waives.txt`
- ALBERT builder/runtime family files
- `tests/e2e_harness/comparators/encoder_only.py`
- `tests/e2e_harness/plugins/encoder_features.py`

**Fix requirements:**

- Fix ALBERT representation parity or define a stronger ALBERT-specific validated contract.
- Do not lower the global floor to hide this issue.
- Remove the XFAIL after the model passes.

**Green criteria:**

- `albert-base` passes without XFAIL.
- The cosine metric is above the enforced floor or a reviewed model-specific contract exists.

### PR 7: `dpr-ctx-encoder`

**Branch:** `fix/dpr-ctx-encoder-e2e`

**Current state:**

- XFAIL.
- Encoder cosine is `0.6086`; enforced floor is `0.8`.
- Old negative threshold override is no longer accepted.

**Files to inspect:**

- `tests/e2e/models/dpr-ctx-encoder.json`
- `tests/e2e/waives.txt`
- DPR/BERT mapping code
- `tests/e2e_harness/references/hf_transformers.py`
- `tests/e2e_harness/comparators/encoder_only.py`

**Fix requirements:**

- Confirm the TRT model and HF reference compare the same DPR context encoder path.
- Fix weight mapping, pooling, tokenization, or reference extraction as needed.
- Remove the XFAIL after the model passes.

**Green criteria:**

- `dpr-ctx-encoder` passes without XFAIL.
- The comparator measures equivalent embeddings from TRT and HF.

### PR 8: `fnet-base`

**Branch:** `fix/fnet-base-e2e`

**Current state:**

- XFAIL.
- Encoder cosine is `0.6919`; enforced floor is `0.8`.
- Waive points at a DFT approximation gap.

**Files to inspect:**

- `tests/e2e/models/fnet-base.json`
- `tests/e2e/waives.txt`
- FNet builder/runtime files
- Fourier/DFT implementation
- `tests/e2e_harness/comparators/encoder_only.py`

**Fix requirements:**

- Fix the FNet transform/parity issue.
- Add focused tests for the DFT path if missing.
- Remove the XFAIL after E2E passes.

**Green criteria:**

- `fnet-base` passes without XFAIL.
- Encoder representation parity is above the contract floor.

### PR 9: `flux-2-dev-fp8`

**Branch:** `fix/flux-2-dev-fp8-e2e`

**Current state:**

- Top-level `pass`.
- `t5_encode` reports `error: Missing output paths for T5 comparison`.
- `dit_step` and `vae_decode` are skipped due missing comparison logic.
- `debug_diffusion_pipeline.py` appears mismatched to FLUX.2 dimensions.

**Files to inspect:**

- `tests/e2e/models/flux-2-dev-fp8.json`
- `tests/e2e_harness/runners/diffusion.py`
- `tests/e2e_harness/references/hf_diffusers.py`
- `tests/e2e_harness/plugins/diffusion.py`
- `tests/e2e_harness/comparators/diffusion.py`
- `tools/debug_diffusion_pipeline.py`

**Fix requirements:**

- Make `t5_encode` produce comparable TRT and HF outputs or explicitly mark the stage unsupported with a visible reason.
- Fix FLUX.2 support in the diffusion debug path or remove the misleading debug stage from this manifest.
- Ensure optional stage errors are visible in summary/report even when top-level status is pass.

**Green criteria:**

- No hidden `error` stage in a green `flux-2-dev-fp8` result.
- If substages remain optional, their status is accurately reported as diagnostic and not confused with pass.

### PR 10: `z-image-turbo`

**Branch:** `fix/z-image-turbo-waive-vlm`

**Current state:**

- XPASS.
- Waive says HF diffusion reference quality should be surfaced by the VLM gate.
- The VLM gate did not run in the ground truth run.

**Files to inspect:**

- `tests/e2e/models/z-image-turbo.json`
- `tests/e2e/waives.txt`
- `tools/evaluate_diffusion_vlm_similarity.py`
- `tests/tools/test_diffusion_vlm_similarity_tool.py`
- Diffusion report rendering in `scripts/generate_e2e_report.py`

**Fix requirements:**

- After PR 0 lands, confirm VLM assessment runs for this model.
- If VLM assessment passes, remove the stale XFAIL.
- If VLM assessment fails because the reference is poor, fix the prompt/reference setup or document and gate the reference-quality issue truthfully.

**Green criteria:**

- No XPASS for `z-image-turbo`.
- VLM semantic assessment is present in the report for this model.

### PR 11: `qwen2.5-0.5b-torchtrt`

**Branch:** `fix/qwen25-05b-torchtrt-waive`

**Current state:**

- XPASS.
- Waive says Torch-TensorRT bundle build depends on plugins not supported by the CUDA 13 CI image.
- Ground truth run passed with NED `0`.

**Files to inspect:**

- `tests/e2e/models/qwen2.5-0.5b-torchtrt.json`
- `tests/e2e/waives.txt`
- Torch-TensorRT builder path

**Fix requirements:**

- Confirm this pass is stable on current CI image.
- Remove stale XFAIL if the model is actually supported now.
- If support is partial, replace the stale waive with a precise gated condition.

**Green criteria:**

- No XPASS for this model.
- Either the model passes normally or the skip/xfail reason matches the current failure.

### PR 12: `gemma-2-2b`

**Branch:** `fix/gemma-2-2b-nightly-coverage`

**Current state:**

- Skipped before orchestrator result creation.
- Waive says gated model needs `HF_TOKEN`.
- Nightly workflow exports `HF_TOKEN`, so confirm whether the skip is stale or the model needs special gated handling.

**Files to inspect:**

- `tests/e2e/models/gemma-2-2b.json`
- `tests/e2e/waives.txt`
- HF cache warming code
- Manifest gated-model handling

**Fix requirements:**

- If nightly has access, remove the skip and make the model produce a result artifact.
- If access is not available, make the skip explicit in the report and summary through PR 0 mechanisms.
- Do not silently omit the model from the matrix.

**Green criteria:**

- The model appears in CI summary and report as pass or intentional skip.

### PR 13: `internlm2-1.8b`

**Branch:** `fix/internlm2-18b-e2e`

**Current state:**

- Skipped before orchestrator result creation.
- Waive says `DynamicCache.from_legacy_cache` was removed in transformers 5.x.

**Files to inspect:**

- `tests/e2e/models/internlm2-1.8b.json`
- `tests/e2e/waives.txt`
- InternLM builder/runtime/reference code
- Python dependency profile handling

**Fix requirements:**

- Update the reference/runtime integration for current transformers, or use a compatible profile if that is the intended contract.
- Remove skip after validation passes.

**Green criteria:**

- The model produces `result.json`.
- It passes normally or has a visible, current, justified skip.

### PR 14: `nllb-200-distilled-600m`

**Branch:** `fix/nllb-200-distilled-600m-e2e`

**Current state:**

- Skipped before orchestrator result creation.
- Skip reason says M2M-100 encoder-decoder weight mapping needs debugging, cosine around `0.30`.

**Files to inspect:**

- `tests/e2e/models/nllb-200-distilled-600m.json`
- NLLB/M2M encoder-decoder builder/runtime code
- Seq2seq reference and comparator paths

**Fix requirements:**

- Fix the encoder-decoder weight mapping or reference alignment.
- Add focused unit coverage for the mapping bug.
- Remove skip only after E2E passes.

**Green criteria:**

- The model produces `result.json`.
- It passes the seq2seq contract without hidden skip.

### PR 15: `phi4-multimodal`

**Branch:** `fix/phi4-multimodal-e2e`

**Current state:**

- Skipped before orchestrator result creation.
- Waive says remote code is incompatible with transformers 5.x.

**Files to inspect:**

- `tests/e2e/models/phi4-multimodal.json`
- `tests/e2e/waives.txt`
- VL reference profile code
- HF transformers profile handling

**Fix requirements:**

- Make the reference path compatible with the current transformers stack or run it under an explicit compatible profile.
- Confirm the TRT path is validating image and text behavior, not text-only fallback.
- Remove skip only after E2E passes.

**Green criteria:**

- The model produces `result.json`.
- It passes or is explicitly reported as unsupported with a current reason.

## Final Nightly Closure PR

After all model PRs are merged and `github/main` is synced, run or wait for a new nightly.

**Branch:** `fix/nightly-e2e-cleanup`

Use this PR only if the final nightly exposes report cleanup or stale documentation. Do not bundle new model fixes here.

Final acceptance:

- No hard E2E failures.
- No stale XPASS.
- All scheduled models are represented in the summary.
- Diffusion VLM assessment exists when diffusion frame pairs exist.
- The HTML report and CI summary agree with JUnit.
- Remaining skips, if any, are intentional, current, and visible.

## Evidence To Attach To Each PR

Every PR description must include:

- Model name.
- Ground truth failure from run `25747071012`.
- Root cause.
- Files changed.
- Local validation commands and results.
- GitHub CI run link for the green PR.
- Artifact/report evidence showing the model status.
- Confirmation that `$write-git-messages` was used for commit and PR text.

Use this template:

```markdown
## Root Cause

## Fix

## Validation

## CI

## Remaining Risk
```

## Do Not Do

- Do not delete waives in bulk.
- Do not mark xfails strict without first fixing or triaging the model.
- Do not weaken thresholds without evidence.
- Do not publish CI reports to GitHub Pages.
- Do not hide a model by skipping before artifact generation unless the skip is intentionally represented in summary/report.
- Do not start the next model PR while the current one is red.
