# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for model-owned validation entrypoints.

Trace: ARCH-MODPLUG-001
Intent: keep developer validation scripts aligned with model-local E2E tests.
Preconditions: validation scripts are present in the repository.
Postconditions: family validation runs model-owned pytest nodes and exposes
isolated model-plugin validation.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_validate_family_uses_model_owned_e2e_entrypoint() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent validate_family.sh from scheduling the shared E2E test node.
    Preconditions: scripts/validate_family.sh exists.
    Postconditions: the script builds tests/e2e/models/<family> node ids.
    """
    text = (REPO_ROOT / "scripts" / "validate_family.sh").read_text(encoding="utf-8")

    assert "tests/test_e2e.py::test_e2e" not in text
    assert "tests/e2e/models/${E2E_FAMILY}/test_${E2E_FAMILY}_e2e.py" in text
    assert "--model-plugin-dir" in text
    assert "--isolate-model-plugin" in text
    assert 'export TRTMC_MODEL_PLUGIN_DIR="$MODEL_PLUGIN_DIR"' in text
    assert "export TRTMC_MODEL_PLUGIN_STRICT=1" in text


def test_autopilot_prompt_uses_model_owned_e2e_entrypoint() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep generated autopilot instructions on model-local E2E tests.
    Preconditions: scripts/autopilot/autorun.py exists.
    Postconditions: the final E2E command points at tests/e2e/models/<family>.
    """
    text = (REPO_ROOT / "scripts" / "autopilot" / "autorun.py").read_text(encoding="utf-8")

    assert "tests/test_e2e.py::test_e2e[{family_name}]" not in text
    assert "tests/e2e/models/{family_name}/test_{family_name}_e2e.py" in text
    assert "src/runtime/plugins/" not in text
    assert "REGISTER_PIPELINE_PLUGIN_WITH_FORCE_LINK" not in text
    assert "RUNTIME_TO_TASK_STRATEGY" not in text
    assert "tools/check_runtime_strategy_matrix.py" in text
    assert "src/runtime/models/{family_name}/MODEL.toml" in text


def test_dispatch_prompt_uses_complete_model_owned_capsule() -> None:
    text = (REPO_ROOT / "scripts" / "autopilot" / "dispatch.py").read_text(encoding="utf-8")

    assert '"runtime_strategy": "decoder_kv_cache"' not in text
    assert "python/tensorrt_model_connect/families/{family_name}.py" not in text
    assert "python/tensorrt_model_connect/families/{family_name}/MODEL.toml" in text
    assert "src/runtime/models/{family_name}/MODEL.toml" in text
    assert "tests/runtime_strategy_matrix.yaml" in text
    assert '"reference_family": "causal_base_continuation"' in text
    assert '"user_contract": "continuation_parity"' in text
    assert '"reference_family": "causal_lm"' not in text
    assert '"user_contract": "text_generation"' not in text
