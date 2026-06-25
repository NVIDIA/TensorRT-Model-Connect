"""Performance benchmark check — TRT vs HF latency/throughput."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class PerfBenchmarkTest:
    name = "perf_benchmark"
    description = "TRT vs HF inference performance comparison (2-way or 3-way with torch.compile)"
    runtime_strategies = []
    requires_bundle = False
    requires_gpu = True

    def run(self, ctx: TestContext) -> DiffResult:
        from perf_compare import run_as_diff_test
        include_compile = getattr(ctx, "options", {}).get("include_compile", False)
        return run_as_diff_test(ctx, include_compile=include_compile)
