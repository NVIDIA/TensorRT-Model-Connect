"""TorchTrtFamilyPlugin protocol — interface for Torch-TRT family plugins.

Unlike the raw-TRT FamilyPlugin (which requires load_weights + build_engine),
this protocol works with live PyTorch models:
  - load_model(): returns an nn.Module ready for StatelessCacheWrapper
  - get_export_args(): returns example input tensors for torch.export
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from ..config import ModelConfig


class TorchTrtFamilyPlugin(Protocol):
    """Interface for a Torch-TRT model family plugin.

    Required attributes:
        name: Human-readable family name (e.g. "qwen", "llama").

    Optional attributes:
        runtime_strategy: Build strategy name (e.g. "decoder", "encoder_only").
            Defaults to "decoder" if absent. Used by compiler.py to select
            the appropriate BuildStrategy for model wrapping and export.

    Required methods:
        matches(): Checks if this plugin handles a given model_type.
        load_model(): Loads the HF model as nn.Module, ready for export.
        get_export_args(): Returns tracing inputs for torch.export.
    """

    name: str

    def matches(self, model_type: str) -> bool:
        """Return True if this plugin handles the given model_type."""
        ...

    def load_model(
        self,
        model_dir: str,
        config: "ModelConfig",
        max_cache_length: int,
        *,
        dtype: "torch.dtype | None" = None,
    ) -> "nn.Module":
        """Load the HF model and prepare it for torch.export.

        The returned module should be:
          - In eval mode
          - On CUDA
          - Using the specified dtype (defaults to torch.float16 if None)

        Args:
            model_dir: Path to model directory.
            config: Parsed model config.
            max_cache_length: Maximum KV cache length.
            dtype: Compute dtype for model weights. Defaults to torch.float16.

        Returns:
            model: nn.Module ready for StatelessCacheWrapper wrapping
        """
        ...

    def get_export_args(
        self,
        model: "nn.Module",
        config: "ModelConfig",
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        """Return example input tensors for torch.export.

        The model parameter is the raw HF model (NOT wrapped).
        Use model.config (HF PretrainedConfig) for StaticCache creation.

        Returns a tuple of tensors matching StatelessCacheWrapper.forward():
          (input_ids, cache_position, attention_mask, cache_kv_0, cache_kv_1, ...)
        """
        ...
