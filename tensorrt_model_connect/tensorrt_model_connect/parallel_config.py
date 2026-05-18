"""Distributed build request metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import TYPE_CHECKING

from .distributed_plan import DISTRIBUTED_PLAN_SECTION

if TYPE_CHECKING:
    from .runtime_config import ConfigBundle


@dataclass(frozen=True)
class ParallelConfig:
    mode: str = "single"
    tp_size: int = 1
    rank: int = -1
    require_mpirun: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode == "tensor_parallel" and self.tp_size > 1

    def for_rank(self, rank: int) -> "ParallelConfig":
        return replace(self, rank=rank)

    def validate(self) -> None:
        if self.mode not in {"single", "tensor_parallel"}:
            raise ValueError(f"Unsupported parallel.mode={self.mode!r}")
        if self.tp_size not in {1, 2, 4, 8}:
            raise ValueError("parallel.tp_size must be one of 1, 2, 4, 8")
        if self.rank < -1:
            raise ValueError("parallel.rank must be -1 or a non-negative rank")
        if self.rank >= self.tp_size:
            raise ValueError(
                f"parallel.rank={self.rank} must be smaller than tp_size={self.tp_size}")
        if self.mode == "single" and self.tp_size != 1:
            raise ValueError("parallel.mode=single requires parallel.tp_size=1")

    def to_config_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "mode": self.mode,
            "tp_size": self.tp_size,
            "rank": self.rank,
            "require_mpirun": self.require_mpirun,
        }

    def to_bundle_config_fields(self) -> dict[str, object]:
        if not self.enabled:
            return {}
        return {
            "parallelism": self.to_config_dict(),
            "distributed_plan_section": DISTRIBUTED_PLAN_SECTION,
        }


def parallel_config_from_bundle(bundle: "ConfigBundle | None") -> ParallelConfig:
    if bundle is None:
        return ParallelConfig()
    cfg = ParallelConfig(
        mode=str(bundle.get("parallel", "mode")),
        tp_size=int(bundle.get("parallel", "tp_size")),
        rank=int(bundle.get("parallel", "rank")),
        require_mpirun=bool(bundle.get("parallel", "require_mpirun")),
    )
    cfg.validate()
    return cfg


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    cfg = value or ParallelConfig()
    cfg.validate()
    return cfg


def require_tensorrt_11_for_tensor_parallel(
    parallel: ParallelConfig,
    *,
    feature: str = "Tensor-parallel builds",
) -> None:
    """Raise unless an enabled tensor-parallel request is running on TRT 11.0+."""
    parallel.validate()
    if not parallel.enabled:
        return

    from . import trt_compat

    version = trt_compat.tensorrt_version()
    match = re.search(r"(\d+)", version or "")
    major = int(match.group(1)) if match else 0
    if major < 11:
        found = version or "unavailable"
        raise RuntimeError(f"{feature} requires TensorRT 11.0+; found {found}")
