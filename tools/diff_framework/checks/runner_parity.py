"""Runner parity check — Python TrtRunner vs C++ trtmc binary."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class RunnerParityTest:
    name = "runner_parity"
    description = "Cross-validate Python TrtRunner vs C++ trtmc binary"
    runtime_strategies = ["decoder_kv_cache", "decoder_moe", "ssm_recurrent"]
    requires_bundle = True
    requires_gpu = True
    required_inputs = ["bundle", "binary"]
    oracle_level = "trt_python_runner"
    deterministic_seed = True
    output_metrics = ["token_match", "text_match", "logit_max_abs_diff"]
    failure_examples = [
        "C++ runner emits different token IDs than Python TrtRunner",
        "bundle runtime strategy maps to the wrong CLI path",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from test_runner_parity import run_as_diff_test
        return run_as_diff_test(ctx)
