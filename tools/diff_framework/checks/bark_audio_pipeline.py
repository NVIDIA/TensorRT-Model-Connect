"""Bark audio pipeline check — staged C++ TRT vs HF audio validation."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class BarkAudioPipelineTest:
    name = "bark_audio_pipeline"
    description = "Bark text-to-audio: C++ sampling, token distribution, codec parity"
    runtime_strategies = ["text_to_audio_bark"]
    requires_bundle = True
    requires_gpu = True
    required_inputs = ["bundle", "binary", "model", "prompt"]
    oracle_level = "hf_transformers"
    deterministic_seed = False
    output_metrics = [
        "stage_1_passed",
        "stage_2_passed",
        "stage_3_passed",
    ]
    failure_examples = [
        "C++ Bark output is near-silent",
        "semantic or coarse audio tokens fall outside the expected ranges",
        "TRT codec waveform diverges from HF EnCodec on the same tokens",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_audio import run_as_diff_test
        return run_as_diff_test(ctx)
