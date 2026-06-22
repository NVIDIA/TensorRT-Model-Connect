"""deepseek_ocr model-owned E2E reference plugins."""

from __future__ import annotations

from .references.golden_snapshot import GoldenSnapshotReference


class DeepseekOcrGoldenSnapshotReference(GoldenSnapshotReference):
    """deepseek_ocr local reference for golden_snapshot."""

reference = DeepseekOcrGoldenSnapshotReference()
