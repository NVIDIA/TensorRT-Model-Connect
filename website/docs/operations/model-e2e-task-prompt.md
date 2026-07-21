# Model E2E Task Prompt

This archived task prompt is kept as operations reference for single-model E2E
bring-up work.

```text
You are working in tensorrt-model-connect. Add one model end-to-end so it is
  truly working and CI-ready, not just compiling.

  Hard requirements:
  1) Run everything inside your own container/workspace only.
  2) Do not ask for confirmation for routine commands; execute directly.
  3) Keep changes minimal and scoped to this model.
  4) Do not relax thresholds unless repeated evidence proves it is necessary.
  5) Validate functional quality (text/image/audio/video meaningfully), not
  only metric pass.

  Inputs you must set before starting:
  - MODEL_NAME: <e2e case name, e.g. xglm-564m>
  - HF_ID: <huggingface model id>
  - FAMILY: <family plugin name>
  - RUNTIME_STRATEGY: <decoder_kv_cache | vision_language | text_to_audio
  | ...>
  - BUNDLE_NAME: <model>.trtfb
  - PROMPT_OR_INPUT: <prompt/image/audio as applicable>
  - TRUST_REMOTE_CODE: <true|false>

  Execution plan (must follow in order):

  A) Environment precheck/build
  - python3 -c "import tensorrt, torch, transformers; print('ok')"
  - pip install --no-deps -e . -C py-only=true
  - cmake -S . -B build -G Ninja \
    -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
    -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/
  tensorrt_libs}/libnvinfer.so" \
    -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
    -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
  - cmake --build build -j

  B) Implement model support
  - If existing family supports the model:
    - Add/update manifest in tests/e2e/models/<FAMILY>/manifests/<MODEL_NAME>.json
    - Add/update tests/e2e/models/<FAMILY>/MODEL.toml so it lists the manifest.
  - If new family required:
    - Scaffold with scripts/new_family.py
    - Implement plugin in python/tensorrt_model_connect/families/<family>.py
    - Ensure family coverage check passes.

  C) Harness compatibility check
  - Ensure runtime->task mapping exists in tests/e2e_harness/contracts.py
  - Ensure manifest_loader can infer proper stage/reference.
  - Ensure suitable runner/reference/comparator/threshold profile exists for
  task strategy.
  - Add only missing wiring; do not refactor unrelated code.

  D) Single-model E2E validation (mandatory)
  Run:
  - python -m pytest tests/test_e2e.py::test_e2e[<MODEL_NAME>] -v \
    --engine-dir /work/engines \
    --trtmc-binary ./build/trtmc \
    --hf-python /opt/venv/bin/python \
    --e2e-artifacts-dir /work/results

  Then inspect artifacts/result.json and verify output quality:
  - text: continuation makes sense
  - audio: listen to wav + check rms/duration
  - image/video: open outputs and verify visually sensible
  - segmentation/detection: masks/boxes plausible

  E) Determinism/repro sanity
  - Re-run the same model at least twice.
  - If stochastic path, set/use seed controls.
  - Confirm stable behavior or quantify acceptable variance.
  - Only then decide if threshold override is needed.

  F) CI-equivalent local gates
  Run:
  - python scripts/check_family_coverage.py
  - python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py -n
  auto
  - python -m pytest tests/tools/ -v -n auto
  - ctest --test-dir build --output-on-failure
  - python -m pytest tests/builder/test_graph_ops.py tests/builder/
  test_graph_ops_extended.py tests/builder/test_graph_blocks.py -v -n auto
  - python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
    --engine-dir /work/engines \
    --trtmc-binary ./build/trtmc \
    --hf-python /opt/venv/bin/python \
    --rebuild-engines
  - Re-run the target model e2e once more after all fixes.

  G) Output/report format (final answer must be strict)
  Provide:

  1. Summary
  - What was added/fixed for MODEL_NAME.

  2. Files changed
  - Exact file list with one-line reason each.

  3. Commands run
  - Exact commands, in execution order.

  4. Validation results
  - Key metrics from result.json (including pass/fail).
  - Functional sanity observations (text/audio/image/video quality).
  - Determinism findings from reruns.

  5. CI readiness
  - Which CI-equivalent gates passed.
  - Any remaining risk or known limitation.

  6. Artifacts
  - Absolute paths to relevant outputs (result.json, logs, wav/png/frames).

  7. Commit
  - Create one clean commit with a cle

  Stop only when all required steps are complete and the model is CI-ready.
```
