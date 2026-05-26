"""Diffusion components check — component-by-component TRT vs HF comparison."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class DiffusionComponentsTest:
    name = "diffusion_components"
    description = "Diffusion pipeline: config, T5, DiT, scheduler, full pipeline"
    runtime_strategies = ["diffusion"]
    requires_bundle = True
    requires_gpu = True
    required_inputs = ["bundle", "prompt"]
    oracle_level = "hf_diffusers"
    deterministic_seed = True
    output_metrics = ["psnr", "ssim", "pixel_mean", "pixel_std"]
    failure_examples = [
        "blank image has low pixel variance",
        "component output diverges from HF Diffusers reference",
    ]

    def run(self, ctx: TestContext) -> DiffResult:
        from debug_diffusion_pipeline import run_as_diff_test
        return run_as_diff_test(ctx)
