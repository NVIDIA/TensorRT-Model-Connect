---
title: Testing Reference
---

## Common commands

```bash
pytest tests/builder -q
pytest tests/tools -q
ctest --test-dir build --output-on-failure
pytest tests/test_e2e.py::test_e2e[<manifest-name>] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc
```

Add `--hf-python /opt/venv/bin/python` only for runtime strategies that still need helper Python code, such as speech-to-speech prompt handling or legacy fallback paths.

## When to use which test

| Change | Minimum useful validation |
| --- | --- |
| Python family plugin | Focused builder tests plus one E2E manifest. |
| Graph ops or decoder builder | Builder tests, parity tools, representative E2E. |
| Runtime plugin | C++ plugin/factory tests plus matching E2E. |
| Public API | C++ API tests and CLI smoke. |
| Tokenizer | Tokenizer unit tests and affected E2E. |
| Config schema | Cross-language schema tests and CLI config tests. |
| Report or diff tooling | `tests/tools` focused tests. |
| Quantization | Builder tests plus modality-specific parity/health tests. |

## E2E manifest fields

Common fields include:

- `name`
- `hf_id`
- `bundle`
- `family`
- `runtime_strategy`
- `precision`
- `max_cache_length`
- task input fields such as `prompt`, `test_prompt`, `audio`, `image`, or `inputs`
- oracle fields such as `reference_backend`, `reference_family`, `user_contract`, and thresholds

## SANA-WM E2E Runtime

The `sana-wm-bidirectional` case validates the model-card camera-control
contract for `Efficient-Large-Model/SANA-WM_bidirectional`:

```bash
python inference_video_scripts/inference_sana_wm.py \
  --image asset/sana_wm/demo_0.png \
  --prompt asset/sana_wm/demo_0.txt \
  --action "w-80,jw-40,w-40,lw-60,w-100" \
  --translation_speed 0.055 \
  --rotation_speed_deg 1.2 \
  --num_frames 321 \
  --output_dir results/demo
```

The HF reference uses an official SANA-WM runtime entrypoint as the oracle. If
the upstream checkpoint does not include `inference_video_scripts/inference_sana_wm.py`
or an action-capable Diffusers `model_index.json`, the case skips during
preflight with `precheck_fail` and writes `preflight_details.json`.

The TRTMC side is expected to run native TensorRT sections. Python fallback
execution is not available for SANA-WM bundles.
The C++ `generate-video` command accepts explicit SANA-WM intrinsics with
`--camera-intrinsics fx,fy,cx,cy`; when omitted, the bundle-level
`sana_wm_default_intrinsics` value is used because Pi3X estimation is not
implemented in C++.
Resolving `Efficient-Large-Model/SANA-WM_bidirectional` downloads only
`README.md` and `config.yaml` by default because the full checkpoint is large;
set `TRTMC_SANA_WM_DOWNLOAD_WEIGHTS=1` when you explicitly need the `dit/`,
`vae/`, `refiner/`, demo asset, and official script directories in the local
HF snapshot.
Set `SANA_WM_NATIVE_PLAN_DIR` to a directory containing the required
`trtmc_engines/*.plan` sections, or set `SANA_WM_MODEL_DIR` to a model
directory with a `trtmc_engines/` subdirectory. The minimum native set is
`text_encoder_0_plan`, `denoiser_plan`, and `sana_wm_vae_encoder_plan`, plus
either `vae_decoder_plan` or the complete refiner set:
`sana_wm_refiner_text_encoder_plan`, `sana_wm_refiner_denoiser_plan`, and
`sana_wm_refiner_vae_decoder_plan`.

To run true parity once the official runtime is available, point either
environment variable at that runtime before invoking the normal E2E command:

```bash
export SANA_REPO=/path/to/Sana
# or:
export SANA_WM_SCRIPT=/path/to/inference_video_scripts/inference_sana_wm.py

pytest tests/test_e2e.py::test_e2e[sana-wm-bidirectional] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc \
  --hf-python /opt/venv/bin/python \
  --e2e-artifacts-dir /workspace/users/yifeif/tensorrt-model-connect/test-result
```
