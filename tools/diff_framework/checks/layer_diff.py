"""Layer diff check — per-layer hidden state comparison: TRT vs HF."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class LayerDiffTest:
    name = "layer_diff"
    description = "Per-layer hidden state comparison: TRT vs HF transformers"
    runtime_strategies = ["decoder_kv_cache", "decoder_moe"]
    requires_bundle = False
    requires_gpu = True
    required_inputs = ["model"]
    oracle_level = "hf_transformers"
    deterministic_seed = True
    output_metrics = ["max_abs_diff", "relative_l2", "cosine_similarity"]
    failure_examples = [
        "hidden state exceeds layer tolerance",
        "layer ordering mismatch hides downstream logit drift",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_layers import run_as_diff_test
        return run_as_diff_test(ctx)
