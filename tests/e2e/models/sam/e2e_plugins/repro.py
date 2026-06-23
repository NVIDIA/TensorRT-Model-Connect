"""SAM model-owned E2E repro command provider."""

from __future__ import annotations

import shlex

from .contracts import E2ECase, ReproCommandProvider, RunContext


def _shell_quote(value: object) -> str:
    return shlex.quote(str(value))


class SamReproCommandProvider:
    """Build SAM TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "sam"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "prompted_segmentation":
            return None

        image = (
            case.inputs.get("image")
            or case.inputs.get("test_image")
            or case.inputs.get("image_path")
            or ""
        )
        infer_parts = [
            ctx.binary_path,
            "segment-prompted",
            bundle_path,
            "--image",
            _shell_quote(image),
            "--output",
            "/tmp/trtmc_masks",
            "--point-x",
            str(case.inputs.get("point_x", 0.5)),
            "--point-y",
            str(case.inputs.get("point_y", 0.5)),
        ]
        if not case.inputs.get("is_foreground", True):
            infer_parts.append("--background")
        return infer_parts


repro_provider: ReproCommandProvider = SamReproCommandProvider()
