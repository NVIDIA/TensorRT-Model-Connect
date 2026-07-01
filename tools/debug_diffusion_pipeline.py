#!/usr/bin/env python3
"""Diffusion pipeline debug entrypoint.

Concrete pipeline-debug behavior is owned by model-family modules under
``tools/families/*/debug_diffusion_pipeline.py``. This
shared tool only discovers and dispatches to those handlers.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def _family_roots() -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        Path(__file__).resolve().parent / "families",
        repo_root / "python/tensorrt_model_connect/families",
    )


def _family_handler_paths(filename: str) -> list[Path]:
    handlers: dict[str, Path] = {}
    for root in reversed(_family_roots()):
        handlers.update({path.parent.name: path for path in root.glob(f"*/{filename}")})
    return [handlers[family] for family in sorted(handlers)]


@lru_cache(maxsize=1)
def _family_debug_modules() -> tuple[ModuleType, ...]:
    modules: list[ModuleType] = []
    for handler_path in _family_handler_paths("debug_diffusion_pipeline.py"):
        module_name = f"_trtmc_debug_diffusion_pipeline_{handler_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:
            print(f"[debug_diffusion_pipeline] WARN: cannot load family debug handler "
                  f"{handler_path}", file=sys.stderr)
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[debug_diffusion_pipeline] WARN: failed to import family debug handler "
                  f"{handler_path}: {exc}", file=sys.stderr)
            continue
        modules.append(module)
    return tuple(modules)


def _select_family_from_argv(argv: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", default=None)
    ns, _ = parser.parse_known_args(argv)
    return ns.family


def _find_family_debug_handler(argv: list[str] | None = None) -> ModuleType:
    args = list(sys.argv[1:] if argv is None else argv)
    requested_family = _select_family_from_argv(args)
    modules = _family_debug_modules()

    if requested_family:
        for module in modules:
            if Path(str(module.__file__)).parent.name == requested_family:
                return module
        raise SystemExit(f"No diffusion debug handler for family {requested_family!r}")

    claimants = [
        module for module in modules
        if callable(getattr(module, "handles_debug_diffusion_pipeline_args", None))
        and module.handles_debug_diffusion_pipeline_args(args)
    ]
    if len(claimants) == 1:
        return claimants[0]
    if len(claimants) > 1:
        names = ", ".join(Path(str(module.__file__)).parent.name
                          for module in claimants)
        raise SystemExit(f"Multiple diffusion debug handlers matched: {names}")
    if len(modules) == 1:
        return modules[0]

    raise SystemExit("No diffusion debug handler matched; pass --family <name>")


def _strip_wrapper_args(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--family":
            skip_next = True
            continue
        if arg.startswith("--family="):
            continue
        stripped.append(arg)
    return stripped


def __getattr__(name: str) -> Any:
    return getattr(_find_family_debug_handler([]), name)


def main() -> None:
    argv = sys.argv[1:]
    handler = _find_family_debug_handler(argv)
    original_argv = sys.argv
    sys.argv = [original_argv[0], *_strip_wrapper_args(argv)]
    try:
        raise SystemExit(handler.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
