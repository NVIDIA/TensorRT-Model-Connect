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
