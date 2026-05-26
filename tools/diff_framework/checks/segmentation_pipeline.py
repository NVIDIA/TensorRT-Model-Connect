"""Segmentation pipeline check — per-pixel TRT vs HF comparison."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class SegmentationPipelineTest:
    name = "segmentation_pipeline"
    description = "Segmentation pipeline: HF logits, TRT logits, per-pixel class agreement"
    runtime_strategies = ["segmentation"]
    requires_bundle = True
    requires_gpu = True
    required_inputs = ["bundle", "image", "model"]
    oracle_level = "hf_transformers"
    deterministic_seed = True
    output_metrics = ["max_logit_diff", "pixel_agreement"]
    failure_examples = [
        "TRT segmentation logits exceed the configured tolerance",
        "per-pixel class predictions diverge from the HF reference",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_segmentation import run_as_diff_test
        return run_as_diff_test(ctx)
