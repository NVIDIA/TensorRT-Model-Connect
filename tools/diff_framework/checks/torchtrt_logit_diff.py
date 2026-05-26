"""Torch-TRT logit check — Torch-TRT path vs HF transformers."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class TorchTrtLogitDiffTest:
    name = "torchtrt_logit_diff"
    description = "Torch-TRT decoder logits: StaticCache path vs HF transformers"
    runtime_strategies = ["torchtrt_decoder"]
    requires_bundle = False
    requires_gpu = True
    required_inputs = ["model"]
    oracle_level = "hf_transformers"
    deterministic_seed = True
    output_metrics = [
        "top1_match_rate",
        "mean_cosine_sim",
        "max_abs_diff",
        "mean_top5_overlap",
    ]
    failure_examples = [
        "generated top-1 token diverges across more than 20 percent of steps",
        "Torch-TRT logits exceed the configured tolerance against HF",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_torchtrt import run_as_diff_test
        return run_as_diff_test(ctx)
