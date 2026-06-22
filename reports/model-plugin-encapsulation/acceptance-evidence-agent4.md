# Model Plugin Encapsulation Acceptance Evidence

Date: 2026-06-12
Workspace: `/workspace/users/yifeif/workspaces/agent-4/TensorRT-Model-Connect`
Container: `trtmc-dev-gb300-agent4`

## Container Build And Runtime Plugin Proof

- `cmake --build build-readme-gpu-agent4 --target trtmc_model_plugins test_model_plugin_loader -j 8`
  passed and rebuilt the changed loader plus all runtime model plugins.
- `ctest --test-dir build-readme-gpu-agent4 --output-on-failure`
  passed: 94/94 C++ tests.
- `readelf -d build-readme-gpu-agent4/models/*/libtrtmc_model_*.so`
  checked 27 model plugin DSOs and found no `libtrtmc_model_*` dependency on
  another model plugin DSO.
- `tools/model_plugin_isolation.py prepare --all --build-dir build-readme-gpu-agent4`
  staged 27 isolated model plugin DSOs under
  `/tmp/trtmc-agent4-all-runtime-plugins-proof`.
- `cmake --install build-readme-gpu-agent4 --prefix /tmp/trtmc-agent4-install-proof`
  staged 27 model plugin DSOs under
  `/tmp/trtmc-agent4-install-proof/lib/trtmc/models/<plugin>/`.

## Python And CI Selection Proof

- Container:
  `/opt/venv/bin/python -m pytest tests/tools/test_test_impact.py tests/tools/test_github_actions_ci.py tests/tools/test_model_plugin_isolation.py tests/tools/test_model_plugin_encapsulation_static.py tests/tools/test_e2e_origin_main_parity.py tests/tools/test_qwen3_omni_hidden_state_flow.py tests/tools/test_omni_comparator.py -q`
  passed: 194 tests.
- Host:
  `python3 tools/test_impact.py --validate` passed with the existing non-fatal
  warnings about stale broad-fallback allowlist entries.
- Model-local threshold example:
  `tools/test_impact.py --files tests/e2e/models/qwen/thresholds/qwen3-0.6b-fp16.json --json`
  selected only `qwen3-0.6b-fp16` and its model-owned pytest node.
- CI config and `tools/test_impact.py` edits now select the `tools` unit tier
  instead of being classified as no-impact.

## Installed Binary E2E Proof

Command:

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e LD_LIBRARY_PATH=/tmp/trtmc-agent4-install-proof/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:/opt/venv/lib/python3.12/site-packages/tensorrt_libs \
  -w /workspace/tensorrt-model-connect \
  trtmc-dev-gb300-agent4 \
  /opt/venv/bin/python -m pytest \
  tests/e2e/models/qwen3_omni/test_qwen3_omni_e2e.py::test_model_e2e[qwen3-omni-30b-a3b-instruct] \
  -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary /tmp/trtmc-agent4-install-proof/bin/trtmc \
  --hf-python /opt/venv/bin/python \
  --e2e-artifacts-dir /tmp/trtmc-agent4-installed-qwen3-omni-no-plugin-dir
```

Result: passed in 445.16 seconds without `--model-plugin-dir`.

Result JSON:
`/tmp/trtmc-agent4-installed-qwen3-omni-no-plugin-dir/qwen3-omni-30b-a3b-instruct/result.json`

Key result fields:

- `status`: `pass`
- `oracle_level`: `L4_invariants`
- `talker_decode`: `passed`
- generated WAV:
  `/tmp/trtmc-agent4-installed-qwen3-omni-no-plugin-dir/qwen3-omni-30b-a3b-instruct_talker_decode.wav`
- generated WAV bytes: 20524

## Origin/Main Parity Summary

`reports/model-plugin-encapsulation/e2e-parity-evidence-agent4.json` records:

- 198 unique current E2E model contracts.
- `failed_unique_models: []`.
- 197 models compared against an `origin/main` result.
- `qwen3-omni-30b-a3b-instruct` is `current_only`.

Local `github/main` inspection showed no model-owned Qwen3-Omni E2E contract:

- `git ls-tree -r --name-only github/main -- tests/e2e/models | rg 'qwen3_omni|qwen3-omni'`
  produced no matches.
- `git ls-tree -r --name-only github/main -- tests/e2e/models/qwen3_omni tests/e2e/models/qwen3_omni/test_qwen3_omni_e2e.py`
  produced no model-owned Qwen3-Omni E2E files.

That makes Qwen3-Omni an intentionally collected/fixed current contract rather
than a parity row with a saved `origin/main` user-contract result.
