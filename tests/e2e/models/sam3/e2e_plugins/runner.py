"""Sam3 model-owned E2E runner plugins."""

from __future__ import annotations

from .contracts import E2ECase
from .runners.segmentation import PromptedSegmentationRunner


class Sam3PromptedSegmentationRunner(PromptedSegmentationRunner):
    """Use Sam3 text prompts for prompted_segmentation."""

    def _text_prompt(self, case: E2ECase) -> str:
        return str(
            case.inputs.get("prompt")
            or case.inputs.get("text_prompt")
            or case.metadata.get("text_prompt")
            or ""
        )

    def _prompt_cli_args(self, case: E2ECase) -> list[str]:
        return ["--prompt", self._text_prompt(case)]

    def _prompt_output_data(self, case: E2ECase) -> dict:
        return {
            "num_expected_masks": case.inputs.get("num_expected_masks", 4),
            "point_prompt": None,
            "text_prompt": self._text_prompt(case),
        }


runner = Sam3PromptedSegmentationRunner()
