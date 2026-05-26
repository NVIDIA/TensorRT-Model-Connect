"""PersonaPlex speech-to-speech check — TRT pipeline vs official or golden reference."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class PersonaPlexPipelineTest:
    name = "personaplex_pipeline"
    description = "PersonaPlex speech-to-speech: TRT audio and token traces vs reference"
    runtime_strategies = ["speech_to_speech"]
    requires_bundle = True
    requires_gpu = True
    required_inputs = ["bundle", "binary", "audio", "reference_dir or official_repo"]
    oracle_level = "official_runtime_or_golden_snapshot"
    deterministic_seed = True
    output_metrics = ["depth_token_match", "audio_rms_ratio", "audio_cosine_sim"]
    failure_examples = [
        "TRT speech output is silent or much quieter than the reference",
        "depth token stream diverges from the official PersonaPlex snapshot",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_personaplex import run_as_diff_test
        return run_as_diff_test(ctx)
