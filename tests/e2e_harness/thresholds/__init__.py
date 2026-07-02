# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Threshold profile loading and resolution.

Provides functions to load threshold defaults from YAML files and
resolve the full threshold chain: defaults -> profile -> per-model -> inline.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"
_OVERRIDES_DIR = Path(__file__).resolve().parent / "overrides"


def load_defaults(task_strategy: str) -> dict[str, float]:
    """Load default thresholds for a task strategy from YAML.

    Returns an empty dict if no defaults file exists.
    Falls back to JSON if PyYAML is not available.
    """
    yaml_path = _DEFAULTS_DIR / f"{task_strategy}.yaml"
    json_path = _DEFAULTS_DIR / f"{task_strategy}.json"

    if yaml_path.is_file():
        try:
            import yaml
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            return data.get("metrics", data) if isinstance(data, dict) else {}
        except ImportError:
            logger.debug("PyYAML not available, trying JSON fallback")

    if json_path.is_file():
        import json
        with open(json_path) as f:
            data = json.load(f)
        return data.get("metrics", data) if isinstance(data, dict) else {}

    return {}


def load_overrides(model_name: str) -> dict[str, float]:
    """Load per-model threshold overrides from YAML/JSON.

    Returns an empty dict if no overrides file exists.
    """
    yaml_path = _OVERRIDES_DIR / f"{model_name}.yaml"
    json_path = _OVERRIDES_DIR / f"{model_name}.json"

    if yaml_path.is_file():
        try:
            import yaml
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            return data.get("metrics", data) if isinstance(data, dict) else {}
        except ImportError:
            pass

    if json_path.is_file():
        import json
        with open(json_path) as f:
            data = json.load(f)
        return data.get("metrics", data) if isinstance(data, dict) else {}

    return {}


def resolve_threshold(
    task_strategy: str,
    profile_name: str = "default",
    model_name: str | None = None,
    inline_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Resolve the full threshold chain.

    Resolution order (later overrides earlier):
    1. Strategy defaults (tests/e2e_harness/thresholds/defaults/<strategy>.yaml)
    2. Per-model overrides (tests/e2e_harness/thresholds/overrides/<model>.yaml)
    3. Inline overrides from manifest
    """
    result: dict[str, float] = {}

    # 1. Strategy defaults
    result.update(load_defaults(task_strategy))

    # 2. Per-model overrides
    if model_name:
        result.update(load_overrides(model_name))

    # 3. Inline overrides
    if inline_overrides:
        result.update(inline_overrides)

    return result
