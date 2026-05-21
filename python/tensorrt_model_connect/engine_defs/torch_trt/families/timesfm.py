"""TimesFM family plugin for Torch-TRT.

The family plugin loads Hugging Face TimesFM checkpoints and registers the
`timesfm` build strategy without relying on the shared registry to discover
optional model families that may not exist in this checkout yet.
"""

from __future__ import annotations

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from ..config import ModelConfig
from ..strategies.timesfm import TimesFmBuildStrategy


def _bootstrap_strategy_registry() -> None:
    """Seed the strategy registry with the core strategies plus TimesFM.

    The shared registry file in this checkout may reference optional model
    families that are not present yet. Populating `_strategies` here keeps the
    TimesFM path usable without depending on those extra modules.
    """

    from .. import strategies as strategy_pkg

    current = getattr(strategy_pkg, "_strategies", None)
    if current is None:
        current = {}

    if HAS_TORCH:
        from ..strategies.decoder import DecoderBuildStrategy
        from ..strategies.diffusion import DiffusionBuildStrategy
        from ..strategies.encoder_only import EncoderOnlyBuildStrategy

        for cls in (DecoderBuildStrategy, EncoderOnlyBuildStrategy, DiffusionBuildStrategy):
            strategy = cls()
            current[strategy.name] = strategy

    current[TimesFmBuildStrategy().name] = TimesFmBuildStrategy()

    strategy_pkg._strategies = current


_bootstrap_strategy_registry()


class TimesFmTorchTrtPlugin:
    name = "timesfm"
    runtime_strategy = "timesfm"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return "timesfm" in mt or "times_fm" in mt

    def load_model(
        self,
        model_dir: str,
        config: ModelConfig,
        max_cache_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.nn.Module:
        import torch
        from transformers import TimesFmModelForPrediction

        del config, max_cache_length
        if dtype is None:
            dtype = torch.float16

        model = TimesFmModelForPrediction.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="cuda",
        )
        model.eval()
        return model

    def get_export_args(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        del precision
        from ..strategies.timesfm import TimesFmBuildStrategy

        strategy = TimesFmBuildStrategy()
        return strategy.make_export_args(
            model.config, max_cache_length, precision="fp16")


plugin = TimesFmTorchTrtPlugin()
