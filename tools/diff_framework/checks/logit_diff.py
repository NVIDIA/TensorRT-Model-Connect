"""Logit diff check — per-step logit comparison: TRT vs HF transformers."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class LogitDiffTest:
    name = "logit_diff"
    description = "Per-step logit comparison: TRT vs HF transformers"
    runtime_strategies = ["decoder_kv_cache", "decoder_moe", "ssm_recurrent"]
    requires_bundle = False
    requires_gpu = True
    required_inputs = ["model"]
    oracle_level = "hf_transformers"
    deterministic_seed = True
    output_metrics = ["max_abs_diff", "cosine_similarity", "topk_overlap"]
    failure_examples = [
        "TRT token IDs diverge while decoded text remains similar",
        "logit cosine drops below the configured tolerance",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_logits import run_as_diff_test
        return run_as_diff_test(ctx)
