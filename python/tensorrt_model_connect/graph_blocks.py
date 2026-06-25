"""Retired shared graph block compatibility stub.

Concrete TensorRT graph blocks are owned by
``tensorrt_model_connect.families.<family>.graph_blocks`` modules.
"""

from __future__ import annotations


class RetiredSharedGraphBlocksError(RuntimeError):
    """Raised when legacy shared graph block entrypoints are used."""


def __getattr__(name: str):
    raise RetiredSharedGraphBlocksError(
        "tensorrt_model_connect.graph_blocks is retired; "
        "use the owning family's graph_blocks module instead."
    )
