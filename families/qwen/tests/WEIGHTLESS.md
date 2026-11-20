# Validate the Qwen3-0.6B weightless build

For single-device BF16 text generation with the native TensorRT KV cache, the
Qwen3-0.6B builder loads the checkpoint weights once and stages the converted
arrays beside the requested output. Both workers map those arrays read-only
and compile the prefill and decode engines in parallel. Each worker builds
its engine with placeholders for the large projection matrices
and language-model head, refits the engine with the checkpoint weights, and
serializes a complete plan. The resulting bundle runs independently of the
checkpoint.

From the repository root, set `MODEL_DIR` to a local snapshot of
`Qwen/Qwen3-0.6B` and `BASELINE_ROOT` to a separate checkout of the version
you want to compare. Use the same Python environment and GPU for both builds.
Choose a new results directory for each comparison.

```bash
export PYTHONPATH=core/builder:.
RESULTS="$(mktemp -d)"

python families/qwen/tests/benchmark_weightless.py "$MODEL_DIR" "$RESULTS" \
  --baseline-root "$BASELINE_ROOT" --precisions bf16 \
  --max-sequence-length 16384 --repeats 2

python families/qwen/tests/test_weightless_build.py \
  "$RESULTS/bf16-0-baseline.bundle" "$RESULTS/bf16-1-candidate.bundle" \
  --report "$RESULTS/parity.json"
```

The benchmark records complete build time in `benchmark.json`. Two repetitions
run in baseline, candidate, candidate, baseline order. The parity check loads
the saved bundles and compares full logits and KV cache contents exactly for
prefill, decode, and continuation with populated caches.
