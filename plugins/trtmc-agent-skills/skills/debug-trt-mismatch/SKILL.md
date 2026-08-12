---
name: debug-trt-mismatch
description: >-
  Use when TensorRT output diverges from a model reference, model-first
  validation fails, generated text or media is wrong, or a family change
  introduces a numerical mismatch. Routes the investigation by model modality
  and escalates from the first divergent boundary to the smallest responsible
  family-owned operation.
---

# Debug TRT Mismatch

## Goal

Find the first boundary where TensorRT and the declared reference disagree.
Preserve the failing workload, sampling settings, model revision, bundle, and
runtime strategy while narrowing the problem. Do not hide a mismatch by
loosening validation thresholds.

## Preflight

Record the exact revision and environment:

```bash
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python3 -c "import tensorrt as trt; print(trt.__version__)"
PYTHONPATH=python:. python3 -c "import tensorrt_model_connect; print('package: OK')"
test -x ./build/trtmc
```

If the local checkout does not have the required GPU or TensorRT environment,
use the existing team container rather than changing the investigation:

```bash
docker ps -a --filter "name=trtmc-dev-gb300" --format "{{.Names}} {{.Status}}"
./scripts/bootstrap_workspace.sh --id <team-id> --branch "$(git branch --show-current)" --detach
```

Run later commands inside `trtmc-dev-gb300-<team-id>` when needed.

## Reproduce Through The Owned Validation Path

List model-first workloads and dry-run the failing one:

```bash
PYTHONPATH=python:. python3 tools/trtmc_validate.py --list
PYTHONPATH=python:. python3 tools/trtmc_validate.py \
  <model> <workload> \
  --dry-run \
  --output /tmp/trtmc-validation
```

The model binding in `tests/validation/model_workloads.yaml`, workload contract
in `tests/validation/workloads.yaml`, family manifests, and their sidecars are
the source of truth for inputs, sampling, and comparison gates. Read
`tools/validation/README.md` before changing the engine: the persisted
`task_eval` artifact key remains intentionally stable even though executable
code moved to `tools/validation/`. Keep reference generation and comparison
separate in the report; an execution failure is not a numerical mismatch.

## Route By Model Capability

Inspect the model's Python, C++, and E2E `MODEL.toml` entries before selecting a
debugger:

| Model path | First focused tool |
|---|---|
| Decoder family with `decoder_debug` validation profile | `tools/diff_logits.py` |
| Decoder layer localization | `tools/diff_layers.py` |
| Vision-language | `tools/diff_vl.py` |
| Audio/Bark | `tools/diff_audio.py` |
| Diffusion | `tools/debug_diffusion_pipeline.py` |
| Python runner versus C++ | `tools/test_runner_parity.py` |

`scripts/validate_family.sh <model>` already routes declared family
capabilities. Do not force decoder-only tools onto audio, diffusion, or another
family that does not declare that profile.

## Decoder Escalation

Start with logits:

```bash
PYTHONPATH=python:. python3 tools/diff_logits.py \
  --model <model> \
  --prompt "The capital of France is" \
  --max-new-tokens 10 \
  --atol 1e-3 \
  --json /tmp/diff-logits.json \
  --verbose
```

Interpret the first divergent step:

| Pattern | Investigate |
|---|---|
| Step 0 diverges | weights, config, prefill graph |
| Error grows each step | precision boundary or normalization |
| Sudden step-N jump | RoPE position, mask, or KV cache |
| Top-1 agrees but max diff is high | close logits; inspect cosine and rank |
| Output differs with small logits diff | sampling, seed, or tie-breaking |

If logits identify a graph mismatch but not its origin, compare layers:

```bash
PYTHONPATH=python:. python3 tools/diff_layers.py \
  --model <model> \
  --prompt "Hello" \
  --atol 0.05 \
  --verbose
```

Use the testcase's declared tolerance as the authority. `0.01`, `0.05`, and
`0.1` are investigation starting points, not replacement pass criteria.

## Modality-Specific Escalation

Vision-language:

```bash
PYTHONPATH=python:. python3 tools/diff_vl.py \
  --bundle /tmp/model.bundle \
  --image /path/to/test.jpg \
  --model <model> \
  --binary ./build/trtmc \
  --hf-python <python> \
  --debug-layers
```

Audio:

```bash
PYTHONPATH=python:. python3 tools/diff_audio.py \
  --bundle /tmp/model.bundle \
  --binary ./build/trtmc \
  --model <model> \
  --hf-python <python> \
  --stage 1
```

Escalate audio stages only after the earlier stage passes. Stage 4 checks greedy
token parity and does not replace waveform or distribution checks.

Diffusion:

```bash
PYTHONPATH=python:. python3 tools/debug_diffusion_pipeline.py \
  --bundle /tmp/model.bundle \
  --model-id <model> \
  --num-steps <small-reproducer-steps>
```

Use deterministic inputs and seeds. Localize preprocessing, encoder, denoiser,
scheduler, and decoder boundaries before changing a family implementation.

## Runtime Boundary

When Python/reference parity passes but the C++ result differs:

```bash
PYTHONPATH=python:. python3 tools/test_runner_parity.py \
  --bundle /tmp/model.bundle \
  --binary ./build/trtmc \
  --hf-python <python> \
  --prompt "The capital of France is" \
  --max-new-tokens 20
```

Inspect tokenizer inputs, bundle metadata, runtime strategy selection, masks,
positions, and cache state. Keep the Python debug path aligned with the C++
runtime contract.

## Operation Isolation

After identifying a layer or stage, run the family-owned tests nearest to the
implementation. Graph helpers belong under the owning family; root graph helper
modules are intentionally absent and are enforced by
`tests/tools/test_model_plugin_encapsulation_static.py`.

For a missing focused test, create the smallest deterministic TensorRT graph
that reproduces the operation and compare it to the reference. Do not broaden a
model-local fix into shared runtime code without proof that multiple families
share the same contract.

## Report

Include:

- exact repository SHA, model revision, model/workload, runtime strategy, and
  bundle path or hash;
- exact reproduction and focused-debug commands;
- execution, reference, comparison, and validation status as separate facts;
- first divergent sample, token, layer, stage, or operation and its metrics;
- what passed and was ruled out;
- smallest ownership-aligned fix and the validation that must pass afterward.

When the mismatch came from Internal CI, keep private logs, artifacts, runner
details, and package coordinates private. Source PRs may cite only the
sanitized exact-head `trtmc/premerge/required` status and public reproduction
evidence.

<!-- Collaborative review anchor: batch 2. -->
