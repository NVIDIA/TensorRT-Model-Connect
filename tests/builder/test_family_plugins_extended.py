"""Generic guard for extended family plugin weight test ownership.

Concrete extended family plugin load_weights assertions live under
``tests/e2e/models/<family>/``. This shared file intentionally keeps no
model-specific checkpoint keys or plugin assertions.
"""

from __future__ import annotations

from pathlib import Path


def test_extended_family_plugin_weight_tests_are_model_owned() -> None:
    models_dir = Path(__file__).resolve().parents[1] / "e2e" / "models"
    owned_tests = sorted(models_dir.glob("*/test_*family_plugin_weights.py"))

    assert owned_tests, "expected family-owned plugin weight tests"
    assert all("/tests/e2e/models/" in test.as_posix() for test in owned_tests)
