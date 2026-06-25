"""Logit diff check — per-step logit comparison: TRT vs HF transformers."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class LogitDiffTest:
    name = "logit_diff"
    description = "Per-step logit comparison: TRT vs HF transformers"
    runtime_strategies = []
    requires_bundle = False
    requires_gpu = True

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_logits import run_as_diff_test
        return run_as_diff_test(ctx)
