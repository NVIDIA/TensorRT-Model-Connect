---
name: profile-model
description: >-
  Use when profiling TensorRT-Model-Connect model performance, comparing TRT
  against HuggingFace or torch.compile, measuring CPU phase overhead, generating
  HTML reports, or validating optimization impact.
---

# Profile Model

## Preconditions

- GPU, CUDA, and TensorRT are available, usually inside a dev container.
- `tensorrt_model_connect` is installed in editable mode:
  `pip install --no-deps -e . -C py-only=true`.
- `./build/trtmc` is built when C++ timing is needed.
- The model is available as a HuggingFace ID or local path.

## Environment Check

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null && echo "GPU: OK" || echo "GPU: MISSING"
python3 -c "import tensorrt as trt; print(f'TRT: {trt.__version__}')" 2>/dev/null || echo "TRT: MISSING"
test -x ./build/trtmc && echo "trtmc: OK" || echo "trtmc: MISSING"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch: MISSING"
```

If required pieces are missing, use an existing `trtmc-dev-gb300-<team-id>`
container or bootstrap one:

```bash
./scripts/bootstrap_workspace.sh --id <team-id> --branch $(git branch --show-current) --detach
```

Run profiling commands inside the container with `docker exec` when appropriate.

## Select The Path

Infer or ask for:

- Model: HuggingFace repo ID or local path.
- Bundle: existing `.trtfb` path, or build on demand.
- Depth: quick (E2E timing only) or full (E2E, per-layer, CPU phases, C++).

Detect `runtime_strategy` from an E2E manifest or bundle:

```bash
rg '"runtime_strategy"' tests/e2e/models/<family>/manifests/<model-name>.json
./build/trtmc inspect <bundle.trtfb> | grep "Runtime strategy"
```

Use the unified profiler for decoder, recurrent, encoder, embedding, and
reranking paths. For diffusion, audio, and other multi-stage models, use the E2E
harness artifacts until dedicated profiler support is available.

## Build Or Inspect Bundle

```bash
./build/trtmc build <model> -o /tmp/<model-name>.trtfb --max-cache-length 256 --verbose
./build/trtmc inspect /tmp/<model-name>.trtfb
```

Skip the build when the user provides a bundle.

## Unified Profiler

Quick profile:

```bash
python tools/trtmc_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 3 --iterations 10 \
  --dtype float16 \
  --json --output-dir /tmp/<model-name>_profile
```

Full profile:

```bash
python tools/trtmc_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --warmup 3 --iterations 10 \
  --dtype float16 \
  --trtmc-binary ./build/trtmc \
  --hf-python /opt/venv/bin/python \
  --cpu-profile \
  --json --output-dir /tmp/<model-name>_profile
```

Useful flags:

| Flag | Use |
|------|-----|
| `--no-compile` | Skip torch.compile when unsupported |
| `--compile-mode max-autotune` | More thorough torch.compile comparison |
| `--no-layer-profile` | E2E-only quick check |
| `--cpu-profile` | Decode overhead investigation |
| `--nsight` | Unavailable in this revision; do not use |
| `--trust-remote-code` | HF custom code models |

The parser still exposes `--nsight`, but that path calls a removed Nsight
collection helper and cannot produce a supported trace. Use external `nsys`
tooling directly when a kernel-level trace is required.

## CPU Phase Deep Dive

```bash
python tools/cpu_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 10 \
  --warmup 3 --iterations 20 \
  --json /tmp/<model-name>_profile/cpu_profile_detailed.json
```

For SSM/Mamba:

```bash
python tools/cpu_profile.py \
  --model <model> \
  --bundle /tmp/<model-name>.trtfb \
  --runner family \
  --max-new-tokens 10 \
  --json /tmp/<model-name>_profile/cpu_profile_mamba.json
```

## Report Generation

`tools/trtmc_profile.py --json` generates the HTML report automatically. If
needed:

```bash
python tools/profile_report.py \
  --output-dir /tmp/<model-name>_profile \
  -o /tmp/<model-name>_profile/report.html
```

## Interpretation

| Condition | Classification | Likely next step |
|-----------|----------------|------------------|
| `d2h + argmax > 15%` | Sync bottleneck | On supported C++ decoder paths, compare `--set runtime.prefer_gpu_greedy=true`; the former `TRTMC_GPU_ARGMAX` environment variable is retired. |
| `tensor_bind > 10%` | Launch overhead | Evaluate CUDA Graph capture/replay |
| `execute > 75%` | Compute-bound | Evaluate FP16/BF16 or kernel quality |
| `execute < 50%`, no dominant phase | Mixed overhead | Combine GPU argmax, CUDA Graphs, and precision work |

Speedup guide:

| TRT vs HF | Meaning |
|-----------|---------|
| `< 2x` | TRT overhead is high relative to model compute |
| `2x-5x` | Normal for small models |
| `5x-10x` | Good kernel and runtime benefit |
| `> 10x` | Excellent; verify HF baseline is fair |

If profiling reveals correctness issues, switch to `$debug-trt-mismatch` before
making performance claims.

## Before/After Comparison

Profile both versions with identical prompts, token counts, warmups, and
iterations. Compare `perf_compare.json`, CPU phase JSON, and top per-layer
timings. Report deltas for decode latency, throughput, top CPU phase, and top
kernel layer.

## User Report

Include model, bundle, commands run, TRT C++/Python throughput, HF and
torch.compile baselines when available, bottleneck classification, top CPU
phase, top TRT layer, artifacts, and the next highest-impact recommendation.
