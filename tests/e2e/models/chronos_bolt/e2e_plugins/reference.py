"""chronos_bolt model-owned E2E reference plugins."""

from __future__ import annotations

from .references.torch_reference import TorchReference


class ChronosBoltTorchReferenceReference(TorchReference):
    """chronos_bolt local reference for torch_reference."""

reference = ChronosBoltTorchReferenceReference()
