from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.registry import (
    activate_model_plugins,
    list_comparators,
    list_references,
    list_runners,
    reset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shared_e2e_discovery_registers_no_concrete_behavior() -> None:
    reset()

    assert list_runners() == {}
    assert list_references() == {}
    assert list_comparators() == {}


def test_model_activation_registers_only_model_owned_behavior() -> None:
    model_dir = REPO_ROOT / "tests" / "e2e" / "models" / "qwen"
    activate_model_plugins(model_dir)

    prefix = "tests.e2e.models.qwen.e2e_plugins."
    plugins = [
        *list_runners().values(),
        *list_references().values(),
        *list_comparators().values(),
    ]
    assert plugins
    assert all(type(plugin).__module__.startswith(prefix) for plugin in plugins)

    reset()
