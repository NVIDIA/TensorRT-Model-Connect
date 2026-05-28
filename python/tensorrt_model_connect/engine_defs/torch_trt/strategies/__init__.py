"""Strategy registry for Torch-TRT build strategies.

Each strategy handles a different model architecture (decoder, encoder-only, etc.)
and knows how to wrap the model, produce export args, and run pre-export setup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BuildStrategy

# Lazy registry — populated on first access to avoid circular imports.
_strategies: dict[str, BuildStrategy] | None = None


def _init_registry() -> dict[str, BuildStrategy]:
    from .chronos_bolt import ChronosBoltBuildStrategy
    from .decoder import DecoderBuildStrategy
    from .patchtst import PatchTSTBuildStrategy
    from .patchtsmixer import PatchTSMixerBuildStrategy
    from .encoder_only import EncoderOnlyBuildStrategy
    from .diffusion import DiffusionBuildStrategy
    from .timesfm import TimesFmBuildStrategy

    strategies: dict[str, BuildStrategy] = {}
    for cls in (
        DecoderBuildStrategy,
        EncoderOnlyBuildStrategy,
        DiffusionBuildStrategy,
        PatchTSTBuildStrategy,
        PatchTSMixerBuildStrategy,
        TimesFmBuildStrategy,
        ChronosBoltBuildStrategy,
    ):
        s = cls()
        strategies[s.name] = s
    return strategies


def get_strategy(name: str) -> BuildStrategy:
    """Return the build strategy for the given name.

    Raises ValueError if the strategy is not found.
    """
    global _strategies
    if _strategies is None:
        _strategies = _init_registry()

    if name not in _strategies:
        available = ", ".join(sorted(_strategies.keys()))
        raise ValueError(
            f"Unknown build strategy {name!r}. Available: {available}")
    return _strategies[name]
