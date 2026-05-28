"""tensorrt_model_connect — TRT engine builder for the trtmc runtime."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "build",
    "build_bundle",
    "write_bundle",
    "ModelConfig",
    "Pipeline",
]


def __getattr__(name: str) -> Any:
    """Lazily import heavyweight helpers so lightweight utilities stay cheap.

    This also keeps TensorRT API access out of package load so --rtx can
    select the backend before graph_ops binds trt_compat.get_trt().
    """
    if name in {"build", "build_bundle"}:
        from .engine_builder import build, build_bundle

        mapping = {
            "build": build,
            "build_bundle": build_bundle,
        }
        return mapping[name]

    if name == "write_bundle":
        from .bundle_writer import write_bundle

        return write_bundle

    if name == "ModelConfig":
        from .config import ModelConfig

        return ModelConfig

    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
