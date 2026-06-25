"""Retired shared graph operation compatibility stub.

Concrete TensorRT graph operations are owned by
``tensorrt_model_connect.families.<family>.graph_ops`` modules.
"""

from __future__ import annotations


class RetiredSharedGraphOpsError(RuntimeError):
    """Raised when legacy shared graph operation entrypoints are used."""


def __getattr__(name: str):
    raise RetiredSharedGraphOpsError(
        "tensorrt_model_connect.graph_ops is retired; "
        "use the owning family's graph_ops module instead."
    )
