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

## Multi-device E2E

Multi-device manifests use `ci_tier="multi_device"` and are excluded from the
normal selective/full E2E lanes. They require TensorRT 11.0+ distributed
collectives and one test process to own multiple GPUs through `mpirun`, so do
not run them through `scripts/run_e2e_parallel.sh`.

The current Qwen TensorRT 11.0+ tensor-parallel manifests are:

- `qwen3-0.6b-fp16-tp2`
- `qwen3-0.6b-fp16-tp4`
- `qwen3-0.6b-fp16-tp8`
- `qwen3-4b-instruct-2507-tp2`
- `qwen3-4b-instruct-2507-tp4`
- `qwen3-4b-instruct-2507-tp8`

Each manifest sets `build_args.parallel.mode=tensor_parallel`,
`build_args.parallel.tp_size=<N>`, and `distributed_runtime.world_size=<N>`.
The runtime runner exports `TRTMC_NCCL_RENDEZVOUS` to all ranks, extracts rank-0
stdout for text comparison, launches the Python debug runner under `mpirun`
when `distributed_runtime.debug_logits=true`, and writes rank-0
`trt_full_logits.npy` for the normal text-generation logit comparator. It also
records a per-case `gpu_memory_samples.csv` plus peak GPU memory summary when
`distributed_runtime.capture_gpu_memory=true`.

Manual invocation in the A100 x86 TensorRT 11.0+ container looks like:

```bash
TRTMC_STORAGE_ROOT=/tmp/trtmc-storage \
TRTMC_HF_CACHE=/tmp/trtmc-hf-cache/hub \
./scripts/docker_run_a100x86_trt11.sh bash -lc '
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python3 -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b-fp16-tp4] -v \
    --engine-dir /tmp/trtmc-storage/engines \
    --trtmc-binary /tmp/trtmc-storage/build-a100x86-trt11/trtmc \
    --hf-python /opt/venv/bin/python \
    --e2e-artifacts-dir /tmp/trtmc-storage/e2e-artifacts \
    --rebuild-engines'
```

CI enablement is controlled by:

```bash
TRTMC_MULTI_DEVICE_E2E=true
TRTMC_MULTI_DEVICE_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRTMC_NCCL_LIB_DIR=/path/to/nccl-2.30-or-newer/lib
MULTI_DEVICE_E2E_TIMEOUT=120m
```

Set `TRTMC_MULTI_DEVICE_VISIBLE_DEVICES` to at least the largest selected
`distributed_runtime.world_size`; the current full lane includes TP=8.

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
