"""BuildStrategy protocol — interface for Torch-TRT build strategies.

Each strategy knows how to wrap a model for export and produce the
appropriate example inputs for torch.export. Strategies are selected
by the family plugin's ``runtime_strategy`` attribute (defaults to
``"decoder"`` when absent).
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class BuildStrategy(Protocol):
    """Interface for a Torch-TRT build strategy.

    Attributes:
        name: Strategy identifier (e.g. "decoder", "encoder_only").
        runtime_strategy: Value written to the bundle's runtime_strategy field.

    Methods:
        wrap_model(): Wrap a raw HF model with strategy-specific I/O adapter.
        make_export_args(): Build example input tensors for torch.export.
        pre_export_setup(): Run any global setup needed before torch.export.
    """

    name: str
    runtime_strategy: str

    def wrap_model(
        self,
        model: "nn.Module",
        config,
        max_cache_length: int,
        *,
        compute_dtype: "torch.dtype | None" = None,
    ) -> "nn.Module":
        """Wrap the HF model with strategy-specific I/O format.

        Args:
            model: HF model (on CUDA, eval mode).
            config: HF PretrainedConfig.
            max_cache_length: Maximum cache/sequence length.
            compute_dtype: Internal compute dtype (default: torch.float16).

        Returns:
            Wrapped nn.Module ready for torch.export.
        """
        ...

    def make_export_args(
        self,
        config,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        """Build example input tensors for torch.export.

        Args:
            config: HF PretrainedConfig.
            max_cache_length: Maximum cache/sequence length.
            precision: Compute precision hint.

        Returns:
            Tuple of tensors matching the wrapper's forward() signature.
        """
        ...

    def pre_export_setup(self) -> None:
        """Run any global setup needed before torch.export (e.g. monkey-patches)."""
        ...
