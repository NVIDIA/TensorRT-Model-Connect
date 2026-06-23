"""ELF Flow model-owned E2E repro command provider."""

from __future__ import annotations

import shlex

from .contracts import E2ECase, ReproCommandProvider, RunContext


def _shell_quote(value: object) -> str:
    return shlex.quote(str(value))


class ElfFlowReproCommandProvider:
    """Build ELF Flow TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "elf_flow"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "diffusion_text_generation":
            return None

        infer_parts = [
            ctx.binary_path,
            "run",
            bundle_path,
            "--prompt",
            _shell_quote(
                case.inputs.get("prompt")
                or case.inputs.get("source_text")
                or case.inputs.get("condition_text")
                or ""
            ),
            "--output",
            "/tmp/trtmc_elf_samples.jsonl",
        ]
        if "max_new_tokens" in case.inputs:
            infer_parts.extend(["--max-new-tokens", str(case.inputs["max_new_tokens"])])
        if int(case.inputs.get("num_samples", 1)) > 1:
            infer_parts.extend(["--num-samples", str(case.inputs["num_samples"])])

        num_steps = case.inputs.get("num_sampling_steps", case.inputs.get("num_steps"))
        if num_steps is not None:
            infer_parts.extend(["--num-steps", str(num_steps)])

        self_cond = case.inputs.get("self_cond_cfg_scale", case.inputs.get("guidance_scale"))
        if self_cond is not None:
            infer_parts.extend(["--guidance-scale", str(self_cond)])
        if "cfg_scale" in case.inputs:
            infer_parts.extend(["--cfg-scale", str(case.inputs["cfg_scale"])])
        if "sde_gamma" in case.inputs:
            infer_parts.extend(["--sde-gamma", str(case.inputs["sde_gamma"])])
        if "seed" in case.inputs:
            infer_parts.extend(["--seed", str(case.inputs["seed"])])

        condition_latents = (
            case.inputs.get("condition_latents_raw")
            or case.inputs.get("condition_latents_path")
        )
        condition_mask = (
            case.inputs.get("condition_mask_raw")
            or case.inputs.get("condition_mask_path")
        )
        if condition_latents:
            infer_parts.extend(["--condition-latents-raw", str(condition_latents)])
        if condition_mask:
            infer_parts.extend(["--condition-mask-raw", str(condition_mask)])

        return infer_parts


repro_provider: ReproCommandProvider = ElfFlowReproCommandProvider()
