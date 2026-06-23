"""Plugin registry for strategy runners, reference backends, comparators,
and contract test plugins.

Supports both explicit registration and auto-discovery from subdirectories.
Auto-discovery scans ``runners/``, ``references/``, ``comparators/``, and
``plugins/`` sibling packages for modules that expose a module-level
``plugin`` attribute implementing the corresponding protocol.

Usage:
    from tests.e2e_harness.registry import get_runner, get_comparator, get_contract_plugin

    runner = get_runner("text_generation_causal")
    output = runner.run_stage(case, stage, ctx)

    contract = get_contract_plugin("chat_instruct_template")
    result = contract.verify(trt_output, ref_output, case, threshold)

Auto-discovery is lazy: it runs on first access if the registry is empty.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
import re
from typing import Dict, Iterable, Optional

from .contracts import (
    Comparator,
    ReferenceBackendRunner,
    ReproCommandProvider,
    TaskStrategyRunner,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal registries
# ---------------------------------------------------------------------------

_strategy_runners: Dict[str, TaskStrategyRunner] = {}
_reference_backends: Dict[str, ReferenceBackendRunner] = {}
_comparators: Dict[str, Comparator] = {}
_repro_command_providers: Dict[str, ReproCommandProvider] = {}
_discovered = False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_runner(runner: TaskStrategyRunner) -> None:
    """Register a TaskStrategyRunner by its strategy_name."""
    name = runner.strategy_name
    if name in _strategy_runners:
        logger.warning("Overwriting strategy runner for %s", name)
    _strategy_runners[name] = runner


def register_reference(ref: ReferenceBackendRunner) -> None:
    """Register a ReferenceBackendRunner by its backend_name."""
    name = ref.backend_name
    if name in _reference_backends:
        logger.warning("Overwriting reference backend for %s", name)
    _reference_backends[name] = ref


def register_comparator(comp: Comparator) -> None:
    """Register a Comparator by its task_strategy."""
    name = comp.task_strategy
    if name in _comparators:
        logger.warning("Overwriting comparator for %s", name)
    _comparators[name] = comp


def register_repro_command_provider(provider: ReproCommandProvider) -> None:
    """Register a ReproCommandProvider by its model family."""
    name = provider.family_name
    if name in _repro_command_providers:
        logger.warning("Overwriting repro command provider for %s", name)
    _repro_command_providers[name] = provider


def _register_plugin_object(module_name: str, plugin: object) -> None:
    """Register one plugin object exposed by a shared or model-local module."""
    if isinstance(plugin, TaskStrategyRunner):
        register_runner(plugin)
        logger.debug("Registered runner from %s: %s", module_name, plugin.strategy_name)
        return
    if isinstance(plugin, ReferenceBackendRunner):
        register_reference(plugin)
        logger.debug("Registered reference from %s: %s", module_name, plugin.backend_name)
        return
    if isinstance(plugin, Comparator):
        register_comparator(plugin)
        logger.debug("Registered comparator from %s: %s", module_name, plugin.task_strategy)
        return
    if isinstance(plugin, ReproCommandProvider):
        register_repro_command_provider(plugin)
        logger.debug(
            "Registered repro command provider from %s: %s",
            module_name,
            plugin.family_name,
        )
        return
    try:
        from .plugins import register_plugin as register_contract_plugin
        from .plugins.base import ContractTestPlugin
    except ImportError:
        ContractTestPlugin = None  # type: ignore[assignment]
        register_contract_plugin = None  # type: ignore[assignment]
    if (
        ContractTestPlugin is not None
        and register_contract_plugin is not None
        and isinstance(plugin, ContractTestPlugin)
    ):
        register_contract_plugin(plugin, source=module_name)
        logger.debug(
            "Registered contract plugin from %s: %s",
            module_name,
            plugin.reference_families,
        )
        return
    logger.warning(
        "Module %s exposed a plugin object that does not match a known E2E "
        "protocol: %r",
        module_name,
        plugin,
    )


def _iter_module_plugins(mod) -> Iterable[object]:
    for attr_name in ("runner", "reference", "comparator", "repro_provider", "plugin"):
        plugin = getattr(mod, attr_name, None)
        if plugin is None:
            continue
        if isinstance(plugin, (list, tuple)):
            yield from plugin
        else:
            yield plugin


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _scan_package(package_dir: Path, package_prefix: str) -> list:
    """Scan a directory for modules with a ``plugin`` attribute.

    Returns list of (module_name, plugin_object) pairs.
    """
    plugins = []
    if not package_dir.is_dir():
        return plugins

    for importer, mod_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if mod_name.startswith("_"):
            continue
        full_name = f"{package_prefix}.{mod_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            logger.warning("Failed to import plugin module %s", full_name, exc_info=True)
            continue
        plugin = getattr(mod, "plugin", None)
        if plugin is not None:
            plugins.append((full_name, plugin))
    return plugins


def discover_plugins() -> None:
    """Scan runners/, references/, and comparators/ for plugin modules.

    Each scanned module should have a module-level ``plugin`` attribute that
    is an instance of TaskStrategyRunner, ReferenceBackendRunner, or
    Comparator respectively. Plugins are registered automatically.

    Safe to call multiple times; subsequent calls are no-ops unless
    ``_discovered`` is reset.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    harness_dir = Path(__file__).resolve().parent

    # Discover strategy runners
    runners_dir = harness_dir / "runners"
    runners_prefix = __package__ + ".runners" if __package__ else "e2e_harness.runners"
    for mod_name, plugin in _scan_package(runners_dir, runners_prefix):
        if isinstance(plugin, TaskStrategyRunner):
            register_runner(plugin)
            logger.debug("Auto-registered runner from %s: %s", mod_name, plugin.strategy_name)
        else:
            logger.warning(
                "Module %s has plugin attribute but it is not a TaskStrategyRunner", mod_name
            )

    # Discover reference backends
    refs_dir = harness_dir / "references"
    refs_prefix = __package__ + ".references" if __package__ else "e2e_harness.references"
    for mod_name, plugin in _scan_package(refs_dir, refs_prefix):
        if isinstance(plugin, ReferenceBackendRunner):
            register_reference(plugin)
            logger.debug("Auto-registered reference from %s: %s", mod_name, plugin.backend_name)
        else:
            logger.warning(
                "Module %s has plugin attribute but it is not a ReferenceBackendRunner", mod_name
            )

    # Discover comparators
    comps_dir = harness_dir / "comparators"
    comps_prefix = __package__ + ".comparators" if __package__ else "e2e_harness.comparators"
    for mod_name, plugin in _scan_package(comps_dir, comps_prefix):
        if isinstance(plugin, Comparator):
            register_comparator(plugin)
            logger.debug("Auto-registered comparator from %s: %s", mod_name, plugin.task_strategy)
        else:
            logger.warning(
                "Module %s has plugin attribute but it is not a Comparator", mod_name
            )


def activate_model_plugins(model_dir: str | Path | None) -> None:
    """Reset registry state and register plugins from one model folder.

    Model-owned plugins live in ``tests/e2e/models/<family>/e2e_plugins/*.py``.
    Each module may expose ``runner``, ``reference``, ``comparator``, or
    ``plugin`` objects implementing the E2E protocol contracts.
    """
    global _discovered
    reset()

    if not model_dir:
        discover_plugins()
        return
    plugin_dir = Path(model_dir) / "e2e_plugins"
    if not plugin_dir.is_dir():
        _discovered = True
        return

    family = re.sub(r"[^0-9A-Za-z_]+", "_", Path(model_dir).name)
    package_prefix = f"tests.e2e.models.{family}.e2e_plugins"
    _discovered = True
    for plugin_path in sorted(plugin_dir.glob("*.py")):
        if plugin_path.name.startswith("_"):
            continue
        if plugin_path.stem in {"contracts", "registry", "runtime_config"}:
            continue
        module_name = f"{package_prefix}.{plugin_path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            logger.warning(
                "Failed to import model-local E2E plugin module %s",
                plugin_path,
                exc_info=True,
            )
            continue
        for plugin in _iter_module_plugins(mod):
            _register_plugin_object(module_name, plugin)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def _ensure_discovered() -> None:
    """Trigger auto-discovery on first access if not already done."""
    if not _discovered:
        discover_plugins()


def get_runner(strategy_name: str) -> Optional[TaskStrategyRunner]:
    """Look up a registered TaskStrategyRunner by strategy name.

    Returns None if no runner is registered for the given strategy.
    Triggers auto-discovery on first call.
    """
    _ensure_discovered()
    return _strategy_runners.get(strategy_name)


def get_reference(backend_name: str) -> Optional[ReferenceBackendRunner]:
    """Look up a registered ReferenceBackendRunner by backend name.

    Returns None if no backend is registered for the given name.
    Triggers auto-discovery on first call.
    """
    _ensure_discovered()
    return _reference_backends.get(backend_name)


def get_comparator(task_strategy: str) -> Optional[Comparator]:
    """Look up a registered Comparator by task strategy.

    Returns None if no comparator is registered for the given strategy.
    Triggers auto-discovery on first call.
    """
    _ensure_discovered()
    return _comparators.get(task_strategy)


def get_repro_command_provider(family_name: str) -> Optional[ReproCommandProvider]:
    """Look up a registered ReproCommandProvider by model family."""
    _ensure_discovered()
    return _repro_command_providers.get(family_name)


def list_runners() -> Dict[str, TaskStrategyRunner]:
    """Return a copy of all registered strategy runners."""
    _ensure_discovered()
    return dict(_strategy_runners)


def list_references() -> Dict[str, ReferenceBackendRunner]:
    """Return a copy of all registered reference backends."""
    _ensure_discovered()
    return dict(_reference_backends)


def list_comparators() -> Dict[str, Comparator]:
    """Return a copy of all registered comparators."""
    _ensure_discovered()
    return dict(_comparators)


def list_repro_command_providers() -> Dict[str, ReproCommandProvider]:
    """Return a copy of all registered repro command providers."""
    _ensure_discovered()
    return dict(_repro_command_providers)


def get_contract_plugin(reference_family: str):
    """Look up a contract test plugin by reference family.

    Returns None if no plugin is registered for the given family.
    Delegates to the plugins sub-package which has its own auto-discovery.
    """
    from .plugins import find_plugin
    return find_plugin(reference_family)


def reset() -> None:
    """Clear all registries and reset discovery state. For testing only."""
    global _discovered
    _strategy_runners.clear()
    _reference_backends.clear()
    _comparators.clear()
    _repro_command_providers.clear()
    _discovered = False
    # Also reset contract plugins
    try:
        from .plugins import reset as reset_plugins
        reset_plugins()
    except ImportError:
        pass
