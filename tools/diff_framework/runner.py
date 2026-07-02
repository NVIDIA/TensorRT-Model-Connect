# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified runner — detect strategy, list tests, run tests."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Literal

from .protocol import DiffResult, TestContext
from .registry import (
    get_all_tests,
    get_strategies_for_test,
    get_tests_for_strategy,
    get_test_by_name,
)

UNDETECTED_RUNTIME_STRATEGY = ""


@dataclass(frozen=True)
class StrategyDetection:
    """Outcome of runtime strategy detection."""

    runtime_strategy: str | None
    status: Literal["ok", "warning", "skip", "error"]
    message: str = ""
    error: Exception | None = None


def _finalize_detection(
    detection: StrategyDetection,
    *,
    raise_on_error: bool,
) -> str:
    """Compatibility bridge for legacy string-return callers."""
    if detection.status == "error":
        if raise_on_error:
            if detection.error is not None:
                raise detection.error
            raise ValueError(detection.message)
        warnings.warn(detection.message, RuntimeWarning, stacklevel=3)
        return UNDETECTED_RUNTIME_STRATEGY

    if detection.status in {"warning", "skip"} and detection.message:
        warnings.warn(detection.message, RuntimeWarning, stacklevel=3)

    return detection.runtime_strategy or UNDETECTED_RUNTIME_STRATEGY


def _classify_detected_strategy(strategy: str, source: str) -> StrategyDetection:
    if get_tests_for_strategy(strategy):
        return StrategyDetection(runtime_strategy=strategy, status="ok")
    return StrategyDetection(
        runtime_strategy=strategy,
        status="skip",
        message=(
            f"Detected runtime_strategy {strategy!r} from {source}, but no diff "
            "tests are registered for it. Skip auto-discovery or pass explicit "
            "--test selections."
        ),
    )


def detect_runtime_strategy(
    model: str,
    *,
    with_status: bool = False,
) -> str | StrategyDetection:
    """Auto-detect runtime_strategy from HF config via family plugin.

    Default return type is `str` for backward compatibility.
    Set `with_status=True` to receive a StrategyDetection object with explicit
    warning/skip/error semantics.
    """
    try:
        from tensorrt_model_connect.engine_builder import _resolve_model
        from tensorrt_model_connect.config import ModelConfig
        from tensorrt_model_connect.families import find_plugin

        model_dir = _resolve_model(model)
        config = ModelConfig.from_dir(model_dir)
        plugin = find_plugin(config.model_type)
    except Exception as exc:
        detection = StrategyDetection(
            runtime_strategy=None,
            status="warning",
            message=(
                f"Could not detect runtime_strategy for model {model!r}; "
                "no default runtime strategy is assumed. "
                f"Details: {type(exc).__name__}: {exc}"
            ),
        )
    else:
        if plugin is None:
            detection = StrategyDetection(
                runtime_strategy=None,
                status="warning",
                message=(
                    f"No family plugin resolved for model {model!r}; "
                    "no default runtime strategy is assumed."
                ),
            )
        else:
            strategy = getattr(plugin, "runtime_strategy", None)
            if not strategy:
                detection = StrategyDetection(
                    runtime_strategy=None,
                    status="warning",
                    message=(
                        f"Family plugin for model {model!r} did not provide "
                        "runtime_strategy; no default runtime strategy is assumed."
                    ),
                )
            else:
                detection = _classify_detected_strategy(
                    strategy=strategy, source=f"model {model!r}")

    if with_status:
        return detection
    return _finalize_detection(detection, raise_on_error=False)


def detect_runtime_strategy_from_bundle(
    bundle_path: str,
    *,
    with_status: bool = False,
) -> str | StrategyDetection:
    """Read runtime_strategy from a bundle's config.json.

    Default return type is `str` for backward compatibility.
    Set `with_status=True` to receive a StrategyDetection object with explicit
    warning/skip/error semantics.
    """
    import json
    import struct

    try:
        with open(bundle_path, "rb") as f:
            _magic = f.read(8)
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            sections = header.get("sections", {})
            data_start = 16 + header_len

            if "config.json" not in sections:
                detection = StrategyDetection(
                    runtime_strategy=None,
                    status="warning",
                    message=(
                        f"Bundle {bundle_path!r} has no config.json; "
                        "no default runtime strategy is assumed."
                    ),
                )
            else:
                meta = sections["config.json"]
                f.seek(data_start + meta["offset"])
                cfg = json.loads(f.read(meta["size"]).decode("utf-8"))
                strategy = cfg.get("runtime_strategy")
                if not strategy:
                    detection = StrategyDetection(
                        runtime_strategy=None,
                        status="warning",
                        message=(
                            f"Bundle {bundle_path!r} config.json has no "
                            "runtime_strategy; no default runtime strategy is assumed."
                        ),
                    )
                else:
                    detection = _classify_detected_strategy(
                        strategy=strategy,
                        source=f"bundle {bundle_path!r}",
                    )
    except Exception as exc:
        detection = StrategyDetection(
            runtime_strategy=None,
            status="error",
            message=(
                f"Failed to read runtime_strategy from bundle {bundle_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ),
            error=exc,
        )

    if with_status:
        return detection
    return _finalize_detection(detection, raise_on_error=True)


def list_tests(runtime_strategy: str | None = None) -> list[dict]:
    """List available tests, optionally filtered by strategy."""
    if runtime_strategy:
        classes = get_tests_for_strategy(runtime_strategy)
    else:
        classes = get_all_tests()

    return [
        {
            "name": cls.name,
            "description": cls.description,
            "runtime_strategies": get_strategies_for_test(cls),
            "requires_bundle": cls.requires_bundle,
            "requires_gpu": cls.requires_gpu,
        }
        for cls in classes
    ]


def run_tests(
    ctx: TestContext,
    test_names: list[str] | None = None,
) -> list[DiffResult]:
    """Run applicable tests. Auto-discovers if test_names is None.

    Returns list of DiffResult objects.
    """
    if test_names is not None:
        classes = []
        for name in test_names:
            cls = get_test_by_name(name)
            if cls is None:
                raise ValueError(f"Unknown test: {name!r}. Available: "
                                 f"{[c.name for c in get_all_tests()]}")
            classes.append(cls)
    else:
        classes = get_tests_for_strategy(ctx.runtime_strategy)

    results = []
    for cls in classes:
        # Skip tests that require a bundle if none provided
        if cls.requires_bundle and not ctx.bundle_path:
            results.append(DiffResult.skip(
                cls.name, ctx.model, ctx.runtime_strategy,
                "No bundle provided (--bundle required)"))
            continue

        instance = cls()
        t0 = time.monotonic()
        try:
            result = instance.run(ctx)
        except Exception as e:
            result = DiffResult.error(
                cls.name, ctx.model, ctx.runtime_strategy,
                str(e), details=f"{type(e).__name__}: {e}")
        result.duration_s = time.monotonic() - t0
        results.append(result)

    return results
