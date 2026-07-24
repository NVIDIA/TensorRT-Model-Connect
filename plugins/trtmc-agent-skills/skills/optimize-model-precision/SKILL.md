---
name: optimize-model-precision
description: >-
  Use when finding the best low-precision or quantized configuration for a
  TensorRT-Model-Connect model. Covers FP16/BF16/quantization attempts, E2E
  validation, fake-FP16 detection, progress tracking, and builder fixes when
  precision is not actually threaded through the graph.
---

# Optimize Model Precision

## Goal

Given a model ID or local model path, find the best lower-precision
configuration that improves speed or memory while preserving the required
accuracy. This applies to text, speech, vision-language, diffusion, segmentation,
encoder-only, and embedding models.

## Inputs

Infer or ask for:

- `MODEL_ID`: HuggingFace model ID or local model path.
- `PROGRESS_FILE`: path for persistent attempt state.
- `ACCURACY_THRESHOLD`: minimum acceptable accuracy when the user specifies one.
- `CONTAINER`: dev container name when GPU work must run in Docker.
- `REPO`: repository path inside the container.

If `PROGRESS_FILE` exists, read it first and resume without repeating completed
attempts.

## Run Commands Inside The Container

Use this pattern when a container is required:

```bash
docker exec <container> bash -c "cd <repo> && <command>"
```

Build examples:

```bash
./build/trtmc build <model> -o <output>.trtfb \
  --precision fp16 \
  --max-cache-length 256

./build/trtmc build <model> -o <output>.trtfb \
  --precision fp16 \
  --quantize fp8 \
  --quant-scales <scales>.json \
  --max-cache-length 256
```

Inspect:

```bash
./build/trtmc inspect <bundle>.trtfb
```

Validate through the E2E harness:

```bash
/opt/venv/bin/python -m pytest tests/test_e2e.py::test_e2e[<manifest-name>] -v \
  --engine-dir <engine-dir> \
  --trtmc-binary ./build/trtmc \
  --hf-python /opt/venv/bin/python \
  --rebuild-engines
```

The E2E harness is the source of truth. A passing attempt means pytest exits 0;
a failing attempt means it does not.

## Detect Fake FP16

Many builders may accept `--precision fp16` before they actually thread dtype
through weights, inputs, and graph helpers.

How to detect:

1. Build an FP32 baseline bundle.
2. Build an FP16 bundle with otherwise identical settings.
3. Inspect both bundles and compare sizes.
4. If sizes are similar, usually within about 10 percent, FP16 probably did not
   take effect.

How to fix:

- Use `$fp16-trt-network`.
- Read the affected builder, not just the CLI surface.
- Thread `work_np_dtype` and `work_trt_dtype` through inputs, constants, graph
  ops, graph blocks, and weight loading.
- Keep norm and softmax boundaries in FP32.
- Cast logits or final comparison outputs to FP32 before marking outputs.

## E2E Manifests

If the model has an E2E manifest, reuse it. If not, create one under
`tests/e2e/models/<family>/manifests/` and list it in the owning
`tests/e2e/models/<family>/MODEL.toml`. For persistent FP16 variants, use a
distinct manifest name and copy the closest current manifest with the same
`task_strategy`:

```json
{
  "name": "<model-name>-fp16",
  "hf_id": "<org>/<model>",
  "bundle": "<model-name>-fp16.trtfb",
  "family": "<family>",
  "runtime_strategy": "<family_owned_strategy>",
  "task_strategy": "<task_strategy>",
  "precision": "fp16",
  "max_cache_length": 256,
  "trust_remote_code": false,
  "testcases": [
    {
      "name": "<model-name>-fp16",
      "trace_id": "IT-E2E-<MODEL>-FP16-01",
      "reference_family": "<reference_family>",
      "user_contract": "<user_contract>",
      "prompt": "test prompt",
      "max_new_tokens": 20
    }
  ]
}
```

The manifest loader requires a known runtime/task-strategy pair and a non-empty
`testcases` list. Audio, vision, diffusion, time-series, and other non-text
tasks use contract-specific testcase fields, so do not copy the text example
without comparing it to a current manifest for that task.

Only relax thresholds when repeated evidence shows the lower-precision output is
valid but not bitwise or token-identical to the FP32/HF reference. Record the
reason in the PR.

## Progress File

Update progress after every attempt, pass or fail:

```json
{
  "model": "MODEL_ID",
  "started": "ISO-8601 timestamp",
  "attempts": [
    {
      "id": 1,
      "precision": "fp16",
      "quantize": null,
      "status": "pass",
      "bundle_size_mb": 1557,
      "fp32_bundle_size_mb": 3111,
      "e2e_result": "PASSED",
      "bundle": "/path/to/bundle.trtfb",
      "manifest": "model-fp16",
      "verified": true,
      "error": null,
      "code_changes": ["<file>: threaded precision through build path"]
    }
  ],
  "best_passing": {
    "precision": "fp16",
    "quantize": null,
    "bundle_size_mb": 1557,
    "bundle": "/path/to/bundle.trtfb",
    "verified": true
  }
}
```

## Invariants

- Do not call the task complete until `best_passing.verified` is true and the
  best passing configuration is not plain FP32.
- Do not override failing E2E results.
- Do not claim FP16 works only because the CLI accepted the flag.
- If the precision change requires code edits, keep them scoped to the builder,
  manifest, and tests needed for that model path.
- Record the exact commands and results in the final report or PR body.

## Search Strategy

Start with FP16 because it is the most common low-risk win. If FP16 passes, try
available quantization modes only when the repo tools and the model family
support them. Prefer one change at a time so failures are attributable.
