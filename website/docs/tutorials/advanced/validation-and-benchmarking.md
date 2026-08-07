---
title: Advanced Tutorial - Validation and Benchmarking
---

import Diagram from '@site/src/components/Diagram';

Validation should prove that the runtime under test matches an appropriate oracle. Benchmarking should state exactly what is measured.

## Learning objectives

By the end of this lab, you should be able to choose evidence that matches a
claim, run a focused model-owned contract, define a benchmark timing boundary,
and produce a receipt another developer can reproduce.

Select the CLI before running a standalone bundle command:

```bash
export TRTMC=trtmc
# Source build inside the development container:
# export TRTMC=./build/trtmc
```

The repository E2E and unit-test sections later on require a source checkout
and its configured `./build/trtmc`; they state that path explicitly.

<Diagram
  src="/img/diagrams/tutorials/advanced/validation-contract.svg"
  alt="Validation contract sending the same canonical input to a tested bundle and declared reference oracle before task-specific comparison and reporting"
  caption="The manifest defines the input, oracle, comparator, and thresholds; a passing report must retain exact artifact and revision provenance."
/>

For release model-profile comparisons, GB300 prerequisites, traffic-light
semantics, and retained performance evidence, use the
[Performance Benchmarking Reference](../../reference/benchmarking.md#release-performance-matrix).

## Focused E2E validation

```bash
ENGINE_DIR=/tmp/trtmc-engines
mkdir -p "${ENGINE_DIR}"
PYTHONPATH=python:. python3 -m pytest \
  'tests/e2e/models/qwen/test_qwen_e2e.py::test_model_e2e[qwen3-0.6b-fp16]' -v \
  --engine-dir "${ENGINE_DIR}" \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models
```

For this native case, the model manifest supplies the family, exact
`runtime_strategy`, bundle name, prompt or modality input, thresholds,
reference backend, and test contract.

An optimized-runtime case uses a different selection chain:
`IMPLEMENTATION.toml`, an exact profile under `profiles/*.toml`, its
semantic-source digest, and Source-side adapter/runtime-contract tests. The
resulting bundle contains `optimized_runtime.json`, implementation metadata,
integrity-bound artifacts, and an embedded `libtrtmc_impl_*.so`. Its public
`runtime_strategy` may be empty because it bypasses the native strategy,
model-DSO, and backend-DSO selection path.

The profile digest is not target-hardware proof. The public Source tree does
not publish the former qualification descriptor, runner, or retained target
artifacts; record separately retained external evidence whenever compatibility,
parity, or performance is claimed.

Read the manifest before debugging a failure. It tells you what the test is trying to prove.

| Manifest field category | Why it matters |
| --- | --- |
| Model and bundle fields | Identify the artifact and the native strategy or optimized implementation/profile. |
| Input fields | Define prompt, image, audio, numeric context, or generation settings. |
| Oracle fields | Define what implementation or verifier is trusted for comparison. |
| Threshold fields | Define acceptable numerical, textual, image, audio, or task-specific differences. |
| Artifact fields | Define where logs and outputs should be written. |

## C++ unit tests

```bash
ctest --test-dir build --output-on-failure --no-tests=error \
  -R 'test_pipeline_registry|test_pipeline_api'
```

Use this when changing pipeline interfaces, registries, plugin manifests, bundle parsing, or runtime core behavior.

C++ unit tests should make runtime failures local. For example, if a
model-owned text-generation pipeline fails an E2E, the unit tests should help
separate DSO/registry failure, bundle parsing, tokenizer behavior, cache
lifecycle, and sampler behavior.

## Builder tests

```bash
pytest tests/builder -q
```

Use builder tests when changing family plugins, graph ops, schedulers, quantization, config schemas, or bundle writing.

## Tool tests

```bash
pytest tests/tools -q
```

Use tool tests when changing diff tools, report generation, performance comparison, coverage maps, or scheduling utilities.

## Runtime microbenchmark

```bash
$TRTMC run Qwen3-0.6B.bundle \
  --prompt "Benchmark prompt" \
  --max-new-tokens 64 \
  --benchmark 20 \
  --warmup 3
```

Report:

- GPU, driver, CUDA, and TensorRT version.
- For a native bundle: exact `runtime_strategy`, model DSO, and backend DSO.
- For an optimized bundle: implementation ID, profile ID, profile
  semantic-source digest, embedded implementation DSO, and the separately
  retained external qualification evidence being cited.
- Bundle path and build command.
- Prompt length and generated token count.
- Whether the number is wall-clock CLI latency, per-token decode time, or raw engine enqueue time.

<Diagram
  src="/img/diagrams/tutorials/advanced/benchmark-timing-scopes.svg"
  alt="Comparison of wall-clock command latency, provider-reported per-token decode latency, and raw TensorRT engine enqueue latency"
  caption="Each metric covers a different boundary, so reports must name the scope and must not compare the three values as equivalents."
/>

These numbers answer different questions. Do not compare them as if they measure the same thing.

The `run` command prints `setup_ms`, `prefill_ms`, `decode_ms`, and the
prefill-plus-decode total returned through `TextResult`; no timing environment
variable is required. Those phase values are provider-reported, not guaranteed
wall-clock measurements. For repeatable phase measurements when the selected
pipeline populates them, use the same benchmark command with a fixed prompt,
warmup count, iteration count, and decoding flags:

```bash
$TRTMC run Qwen3-0.6B.bundle \
  --prompt "Benchmark prompt" \
  --max-new-tokens 64 \
  --greedy \
  --benchmark 20 \
  --warmup 3
```

The CLI benchmark averages the phase values returned by the pipeline; it does
not independently time each public call. Native text pipelines commonly
populate these fields. A provider that cannot expose a trustworthy phase split
may return zero. The qualified Qwen Edge-LLM adapter does exactly that, so its
zero prefill/decode values and any throughput derived from them are
unavailable—not zero-latency results.

When phase timing is unavailable, use a controlled target-environment
benchmark that synchronizes the device and measures wall time around the public
pipeline call. Record that result as public-call wall latency. Use a
model-specific profiler for engine-only timing, and identify the exact tool,
revision, artifact location, and measurement boundary in the report.

## Validation taxonomy

| Validation type | Best for | Limit |
| --- | --- | --- |
| Unit test | Local logic, edge cases, fast feedback. | Does not prove a full model works. |
| E2E smoke | User-visible model contract. | Usually one or a few inputs. |
| Numerical parity | Tensor output or score comparison. | Requires stable oracle and well-defined tolerance. |
| Semantic/text verifier | Generated text or transcript quality. | Can be nondeterministic unless generation settings are controlled. |
| Image/audio verifier | Modality-specific quality checks. | Requires careful artifact and threshold design. |
| Microbenchmark | Fixed-input latency. | Not a product workload or dataset evaluation. |
| Dataset evaluation | Accuracy or quality across data. | Slower and outside most quick CI paths. |

## Benchmark setup template

Use this template in reports:

```text
Model:
Bundle:
Family:
Execution path: native | optimized
Runtime strategy:
Implementation ID:
Profile ID:
Profile semantic-source digest:
External qualification evidence:
Build command:
Run command:
GPU:
Driver/CUDA/TensorRT:
Model DSO:
Backend DSO:
Embedded implementation DSO:
Input:
Output length or shape:
Warmup iterations:
Measured iterations:
Metric:
Result:
Notes:
```

The input is part of the benchmark. A one-token prompt, a 2K-token prompt, a single image, an 81-frame video, and a dataset run are different workloads.

## Self-check

1. Which evidence proves a documentation-only change, and which proves exact
   model parity on a GPU?
2. Why must a benchmark report its input and timing boundary?
3. Is a skipped hardware preflight a passing E2E result?

<details>
<summary>Check your answers</summary>

1. Site/link/build checks prove documentation integrity; exact model parity
   requires the named revision/configuration, target execution, reference
   oracle, comparator, and thresholds.
2. Different prompts, shapes, output lengths, and included setup/build/runtime
   work measure different workloads and cannot be compared honestly otherwise.
3. No. It proves only that the required environment was unavailable.

</details>

{/* Collaborative review anchor. */}
