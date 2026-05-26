"""T5 encoder check — TRT encoder embeddings vs HF transformers."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class T5EncoderDiffTest:
    name = "t5_encoder_diff"
    description = "T5 encoder embeddings: TRT engine vs HF transformers"
    runtime_strategies = ["text_to_text"]
    requires_bundle = False
    requires_gpu = True
    required_inputs = ["model", "prompt"]
    oracle_level = "hf_transformers"
    deterministic_seed = True
    output_metrics = ["max_abs_diff", "mean_abs_diff"]
    failure_examples = [
        "encoder embedding max absolute diff exceeds tolerance",
        "tokenizer or text_encoder path resolves to the wrong model component",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_t5 import run_as_diff_test
        return run_as_diff_test(ctx)
