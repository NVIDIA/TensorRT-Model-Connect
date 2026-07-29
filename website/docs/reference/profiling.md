# Profiling Guide

The supported profiling entry point is `tools/trtmc_profile.py`. It compares
the repository's Python TensorRT runner with Hugging Face eager execution and,
unless disabled, `torch.compile`. Its prebuilt-bundle loader currently supports
only the native bundle shape described below; it is not a generic profiler for
optimized-runtime bundles.

Run profiling in the project development image or another environment that
contains TensorRT, CUDA, PyTorch, Transformers, and the model checkpoint.

## Profile a model

Build the engine in-process and save machine-readable artifacts:

```bash
PYTHONPATH=python:. python3 tools/trtmc_profile.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 1 \
  --iterations 3 \
  --output-dir /tmp/qwen3-profile \
  --json
```

Profile an existing native bundle and include the C++ runtime:

```bash
PYTHONPATH=python:. python3 tools/trtmc_profile.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3-0.6b.trtfb \
  --trtmc-binary ./build/trtmc \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 1 \
  --iterations 3 \
  --output-dir /tmp/qwen3-profile \
  --json
```

The current `--bundle` path requires a top-level `engine_plan` section plus
`config.json` with a nonempty native `runtime_strategy`; it explicitly rejects
the `vision_language` strategy. Split-plan and optimized-runtime bundle shapes
do not satisfy this loader. `--trtmc-binary` requires `--bundle` and does not
bypass that Python-side load.

The model-only profiler path has its own diagnostic default of
`--max-cache-length 256` and calls the lower-level single-engine builder. It
therefore does not reproduce `trtmc build` family defaults. In particular, an
eligible dense Qwen3 or Llama `trtmc build` bundle uses split prefill/decode
plans and does not satisfy the profiler's current `--bundle` loader.

Use `--hf-python /path/to/python` only when the native runtime needs a Python
helper. Add `--trust-remote-code` only after reviewing the checkpoint
repository. `--no-compile` skips the `torch.compile` comparison, and
`--no-layer-profile` skips the TensorRT `IProfiler` pass.

For an optimized-runtime bundle, use its family-owned qualification workflow
and the public C++ benchmark path instead:

```bash
./build/trtmc run /path/to/optimized.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 1 \
  --benchmark 3
```

Record the qualified implementation/profile identity, exact bundle and commit,
target, downstream runtime, prompt, warmup/iteration counts, and the
provider-owned timing artifact. The public inspector currently confirms
optimized descriptor/artifact section presence but does not print descriptor
identity values.

With `--json`, the command writes:

- `perf_compare.json`
- `layer_profile.json` when the family supports layer profiling
- `report.html`
- `cpu_profile.json` when `--cpu-profile` is requested, the family supports
  CPU phase profiling, and `--bundle` is also supplied

The exact options are authoritative in the parser:

```bash
python3 tools/trtmc_profile.py --help
```

## Focused comparison tools

Use the focused tools when a performance result first needs a correctness
check:

```bash
PYTHONPATH=python:. python3 tools/diff_logits.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --json /tmp/qwen3-logits.json

PYTHONPATH=python:. python3 tools/diff_layers.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "The capital of France is" \
  --atol 0.001
```

`diff_logits.py` accepts `--battery` for the repository's standard prompt
battery. Neither command has a `--check` option; comparison types are separate
entry points.

## CPU phase breakdown

The unified profiler invokes `tools/cpu_profile.py` when `--cpu-profile` is
present and the selected family supports CPU phase profiling. In this revision,
that unified path also requires `--bundle`; without it the optional subprocess
fails and no `cpu_profile.json` is produced. The focused tool can also be run
directly:

```bash
PYTHONPATH=python:. python3 tools/cpu_profile.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 1 \
  --iterations 3 \
  --json /tmp/qwen3-cpu-profile.json
```

## Nsight status

Do not use `--nsight` in this revision. The option is still exposed by
`trtmc_profile.py`, but its implementation calls a removed Nsight collection
helper. Collect an external `nsys` trace manually if needed; there is currently
no repository-supported conversion command for that trace.

## Interpreting results

- Compare identical checkpoints, prompts, token limits, precision, profiles,
  warmups, and iteration counts.
- Establish parity before treating a speedup as meaningful.
- Do not compare the `IProfiler` pass itself against uninstrumented latency;
  the unified tool runs a separate uninstrumented TensorRT timing pass.
- Retain JSON artifacts with the tested commit, GPU, TensorRT version, and
  exact command when using results as qualification evidence.
