---
name: debug-trt-mismatch
description: >-
  Use when TensorRT output diverges from HuggingFace output, an E2E comparison
  fails, a model emits wrong text or media, or a family/plugin change introduces
  numerical mismatch. Provides an escalation path from logit diffs to layer
  diffs, vision-language checks, runner parity, and graph-op isolation.
---

# Debug TRT Mismatch

## First Step: Environment

Run the checks before diffing:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null && echo "GPU: OK" || echo "GPU: MISSING"
python3 -c "import tensorrt as trt; print(f'TRT: {trt.__version__}')" 2>/dev/null || echo "TRT: MISSING"
python3 -c "import tensorrt_model_connect; print('tensorrt_model_connect: OK')" 2>/dev/null || echo "tensorrt_model_connect: MISSING (run: pip install --no-deps -e . -C py-only=true)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch: MISSING"
```

For C++ parity checks:

```bash
test -x ./build/trtmc && echo "C++ binary: OK" || echo "C++ binary: MISSING"
```

If GPU, TensorRT, or the package is missing, look for an existing team
container:

```bash
docker ps -a --filter "name=trtmc-dev-gb300" --format "{{.Names}} {{.Status}}"
```

Use a running container with `docker exec trtmc-dev-gb300-<team-id> ...`, start
a stopped container, or bootstrap one:

```bash
./scripts/bootstrap_workspace.sh --id <team-id> --branch $(git branch --show-current) --detach
```

Then rerun the checks inside the container. Prefix subsequent commands with
`docker exec trtmc-dev-gb300-<team-id>` when needed.

## Escalation Path

Start at Level 1 and escalate only when the lower level does not isolate the
cause.

| Level | Tool | Question |
|-------|------|----------|
| 1 | `tools/diff_logits.py` | Which decode step diverges? |
| 2 | `tools/diff_layers.py` | Which transformer layer diverges? |
| 3 | `tools/diff_vl.py` | Is the issue in the vision encoder or text decoder? |
| 4 | `tools/test_runner_parity.py` | Does Python runner match the C++ binary? |
| 5 | Graph op tests/manual reproducer | Which operation inside the layer is wrong? |

## Level 1: Token Logits

Quick check:

```bash
python tools/diff_logits.py \
  --model <model> \
  --prompt "The capital of France is" \
  --max-new-tokens 10 \
  --atol 1e-3 \
  --verbose
```

Full battery:

```bash
python tools/diff_logits.py \
  --model <model> \
  --atol 1e-3 \
  --battery \
  --json /tmp/diff_logits.json
```

Interpretation:

| Pattern | Likely cause |
|---------|--------------|
| Step 0 diverges | Weight mapping, config parsing, or prefill graph bug |
| Error grows every step | Accumulating precision error, often norms |
| Sudden divergence at step N | RoPE position, mask, or KV cache issue |
| Huge max diff plus repeated tokens | Fundamentally wrong weight mapping |
| Good text with high max diff | Numerical noise or close logits; inspect top-1 and cosine |
| Wrong text with low max diff | Sampling/tie-breaking issue, not necessarily graph math |

Useful metrics: `cosine_p5`, `top1_match_rate`, `token_agreement`, and per-step
`max_diff`.

## Level 2: Layer Hidden States

```bash
python tools/diff_layers.py \
  --model <model> \
  --prompt "Hello" \
  --atol 0.05 \
  --verbose
```

Interpretation:

| Pattern | Likely cause |
|---------|--------------|
| Embedding diverges | Wrong embedding key or table transform |
| Layer 0 diverges | First attention/MLP weights or graph op |
| Linear growth across layers | Precision accumulation; inspect FP32 boundaries |
| One layer jumps | Fused QKV, gate/up split, RoPE, or layer-local config issue |
| Layers match but logits diverge | Final norm, LM head, or tied embeddings |

Default thresholds: use `0.01` for strict FP32, `0.05` for typical FP16/mixed
precision, and `0.1` only when the model is expected to be looser.

## Level 3: Vision-Language

Vision sanity:

```bash
python tools/diff_vl.py \
  --bundle /tmp/model.trtfb \
  --image /path/to/test.jpg \
  --vision-only
```

Vision features vs HF:

```bash
python tools/diff_vl.py \
  --bundle /tmp/model.trtfb \
  --image /path/to/test.jpg \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --atol 0.1
```

Full VL generation plus C++ parity:

```bash
python tools/diff_vl.py \
  --bundle /tmp/model.trtfb \
  --image /path/to/test.jpg \
  --model <hf-model> \
  --binary ./build/trtmc \
  --hf-python /opt/venv/bin/python \
  --max-new-tokens 30
```

If vision features do not match, test preprocessing variants such as
`--preprocessor-type simple_chw` and then read the model-specific preprocessor
code.

## Level 4: Python Runner vs C++ Binary

```bash
python tools/test_runner_parity.py \
  --bundle /tmp/model.trtfb \
  --binary ./build/trtmc \
  --hf-python /opt/venv/bin/python \
  --prompt "The capital of France is" \
  --max-new-tokens 20
```

If Python matches HF but C++ differs, focus on `src/runtime/` and tokenizer or
bundle loading logic. If C++ mask/cache/position logic changes, update the
Python debug runner and verify parity again.

## Level 5: Graph Op Isolation

Run existing graph tests first:

```bash
pytest tests/builder/test_graph_ops.py -v -m trt
pytest tests/builder/test_graph_ops_extended.py -v -m trt
pytest tests/builder/test_graph_blocks.py -v -m trt
```

If tests pass but the model still diverges, create a minimal TensorRT graph for
the suspected op and compare it to the PyTorch reference. The `trt_runner`
fixture in `tests/builder/conftest.py` is the preferred starting point.

## Reporting

Report the first failing level, model, bundle path, exact command, key metrics,
first divergent step/layer/op, likely root cause, and the smallest proposed
fix. Include what was ruled out so the next investigator does not repeat work.
