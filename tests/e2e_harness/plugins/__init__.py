# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test plugin auto-discovery.

Scans this directory for modules with a module-level ``plugin`` attribute
implementing ContractTestPlugin.  Same pattern as builder family plugins.

Usage:
    from tests.e2e_harness.plugins import find_plugin

    plugin = find_plugin("chat_instruct_template")
    if plugin:
        config = plugin.configure_reference(case)
        result = plugin.verify(trt_output, ref_output, case, threshold)
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Optional

from .base import ContractTestPlugin

logger = logging.getLogger(__name__)

_plugins: Dict[str, ContractTestPlugin] = {}
_discovered = False


def register_plugin(plugin: ContractTestPlugin, *, source: str = "") -> None:
    """Register a contract plugin object.

    Model-local E2E plugin activation uses this to add family-owned contract
    checks before shared contract plugins are discovered.
    """
    if not isinstance(plugin, ContractTestPlugin):
        raise TypeError(f"{plugin!r} is not a ContractTestPlugin")

    for family in plugin.reference_families:
        if family in _plugins:
            logger.warning(
                "Contract plugin for family %s already registered, "
                "overwriting with %s",
                family,
                source or type(plugin).__module__,
            )
        _plugins[family] = plugin
        logger.debug(
            "Registered contract plugin %s for family %s",
            source or type(plugin).__module__,
            family,
        )


def _discover() -> None:
    """Scan this directory for contract test plugins."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    pkg_dir = Path(__file__).resolve().parent
    pkg_prefix = __package__

    for _importer, mod_name, _is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
        if mod_name.startswith("_") or mod_name == "base":
            continue
        full_name = f"{pkg_prefix}.{mod_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            logger.warning("Failed to import contract plugin %s", full_name, exc_info=True)
            continue

        plugin = getattr(mod, "plugin", None)
        if plugin is None:
            continue

        if not isinstance(plugin, ContractTestPlugin):
            logger.warning(
                "Module %s has plugin attribute but it is not a ContractTestPlugin",
                full_name,
            )
            continue

        for family in plugin.reference_families:
            if family in _plugins:
                logger.debug(
                    "Keeping pre-registered contract plugin for family %s "
                    "instead of shared plugin %s",
                    family,
                    full_name,
                )
                continue
            register_plugin(plugin, source=full_name)


def find_plugin(reference_family: str) -> Optional[ContractTestPlugin]:
    """Look up a contract test plugin by reference family.

    Returns None if no plugin is registered for the given family.
    Triggers auto-discovery on first call.
    """
    if not _discovered:
        _discover()
    return _plugins.get(reference_family)


def list_plugins() -> Dict[str, ContractTestPlugin]:
    """Return a copy of all registered contract plugins keyed by family."""
    if not _discovered:
        _discover()
    return dict(_plugins)


def reset() -> None:
    """Clear all registrations and reset discovery. For testing only."""
    global _discovered
    _plugins.clear()
    _discovered = False
