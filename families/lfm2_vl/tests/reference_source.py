# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned source metadata and scoped input checks for the 5.2 reference tools."""

import json
from pathlib import Path

CHECKPOINT = "LiquidAI/LFM2.5-VL-1.6B"
REVISION = "919fde3d022e3f90a4716006f993938ee8c2eb97"

_SELECTORS = {
    name: frozenset(values)
    for kind, values in (
        ("attn", ("eager", "sdpa")),
        ("experts", ("eager", "grouped_mm", "batched_mm")),
    )
    for name in (
        f"{kind}_implementation",
        f"_{kind}_implementation",
        f"_{kind}_implementation_internal",
    )
}


def _guard_selector(value, allowed, path):
    if value is None or isinstance(value, str) and value in allowed:
        return
    if isinstance(value, dict):
        for name, selected in value.items():
            _guard_selector(selected, allowed, f"{path}.{name}")
        return
    raise ValueError(f"unsupported reference kernel selector at {path}")


def guard_reference_config(config):
    """Reject Hub/fallback selectors without modifying the original configuration."""
    if not isinstance(config, dict):
        raise ValueError("reference config.json must contain an object")

    def visit(value, path):
        if isinstance(value, dict):
            for name, item in value.items():
                field = f"{path}.{name}"
                if name == "configuration_files" and item:
                    raise ValueError("reference configuration-file redirection is unsupported")
                if name in _SELECTORS:
                    _guard_selector(item, _SELECTORS[name], field)
                visit(item, field)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(config, "config")


def guard_reference_model(model):
    """Check an already constructed model before the sampling helper uses it."""
    if model.use_kernels:
        raise ValueError("reference sampling does not support use_kernels")
    pending = [model.config]
    seen = set()
    while pending:
        config = pending.pop()
        if id(config) in seen:
            continue
        seen.add(id(config))
        # to_dict() omits the private selector fields; inspect the actual values.
        guard_reference_config(vars(config))
        pending.extend(
            child
            for name in getattr(config, "sub_configs", {})
            if (child := getattr(config, name, None)) is not None
        )


def _guard_reference_directory(checkpoint, options):
    if Path(checkpoint).is_dir():
        config_path = Path(checkpoint) / "config.json"
    else:
        from huggingface_hub import constants, hf_hub_download

        config_path = Path(
            hf_hub_download(
                checkpoint,
                "config.json",
                revision=options["revision"],
                local_files_only=constants.HF_HUB_OFFLINE,
            )
        )
    if (config_path.parent / "adapter_config.json").exists():
        raise ValueError("reference validation requires a full checkpoint, not an adapter")
    guard_reference_config(json.loads(config_path.read_text(encoding="utf-8")))


def reference_source(checkpoint):
    if Path(checkpoint).is_dir():
        options = {"local_files_only": True}
        _guard_reference_directory(checkpoint, options)
        return options, {
            "checkpoint": "local",
            "revision": None,
            "provenance": {"source": "local", "revision_verified": False},
        }
    if checkpoint != CHECKPOINT:
        raise ValueError("expected the pinned repository ID or an existing local directory")
    options = {"revision": REVISION}
    _guard_reference_directory(checkpoint, options)
    return options, {
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "provenance": {"source": "hugging_face", "revision_verified": True},
    }
