"""personaplex model-owned E2E reference plugins."""

from __future__ import annotations

from .references.torch_reference import TorchReference


class PersonaplexTorchReferenceReference(TorchReference):
    """personaplex local reference for torch_reference."""

reference = PersonaplexTorchReferenceReference()
