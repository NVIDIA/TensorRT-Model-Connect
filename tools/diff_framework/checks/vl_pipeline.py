"""VL pipeline check — vision-language diff testing."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class VLPipelineTest:
    name = "vl_pipeline"
    description = "Vision-language pipeline: vision features, embed, generation, C++ parity"
    runtime_strategies = []
    requires_bundle = True
    requires_gpu = True

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_vl import run_as_diff_test
        return run_as_diff_test(ctx)
