"""Performance benchmark check — TRT vs HF latency/throughput."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class PerfBenchmarkTest:
    name = "perf_benchmark"
    description = "TRT vs HF inference performance comparison (2-way or 3-way with torch.compile)"
    runtime_strategies = ["decoder_kv_cache", "decoder_moe", "ssm_recurrent"]
    requires_bundle = False
    requires_gpu = True
    required_inputs = ["model"]
    oracle_level = "performance_only"
    deterministic_seed = False
    output_metrics = ["hf_latency_ms", "trt_latency_ms", "speedup", "tokens_per_s"]
    failure_examples = [
        "accuracy passes but TRT latency regresses below the target speedup",
        "benchmark omits warmup or timing method metadata",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from perf_compare import run_as_diff_test
        include_compile = getattr(ctx, "options", {}).get("include_compile", False)
        return run_as_diff_test(ctx, include_compile=include_compile)
