"""VL pipeline check — vision-language diff testing."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class VLPipelineTest:
    name = "vl_pipeline"
    description = "Vision-language pipeline: vision features, embed, generation, C++ parity"
    runtime_strategies = ["vision_language"]
    requires_bundle = True
    requires_gpu = True
    required_inputs = ["bundle", "image"]
    oracle_level = "hf_transformers"
    deterministic_seed = True
    output_metrics = ["embedding_cosine", "normalized_edit_distance", "word_agreement"]
    failure_examples = [
        "vision embedding cosine drops below threshold",
        "answer text misses required reference substrings",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_vl import run_as_diff_test
        return run_as_diff_test(ctx)
