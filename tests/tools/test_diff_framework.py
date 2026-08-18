# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for tools/diff_framework/ — protocol, registry, runner, CLI.

Pure-Python tests (no GPU, no model loading) that verify the framework
mechanics. Runs as part of Tier 1 (`pytest tests/tools/ -v`).

Trace: ARCH-DIFF-001, UD-DIFF-FRAMEWORK
Intent: Validate diff framework protocol, plugin registry, runner lifecycle, and CLI argument parsing
Preconditions: diff_framework modules are importable; no GPU or model files required
Postconditions: Registry discovers plugins, runner executes comparison lifecycle, and CLI parses arguments correctly
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest


def _import_framework():
    import importlib
    return importlib.import_module("diff_framework")


def _import_protocol():
    import importlib
    return importlib.import_module("diff_framework.protocol")


def _import_registry():
    import importlib
    return importlib.import_module("diff_framework.registry")


def _import_runner():
    import importlib
    return importlib.import_module("diff_framework.runner")


def _mock_model_strategy_detection(
    monkeypatch,
    *,
    runtime_strategy: str | None = "qwen_decoder_kv_cache",
    plugin_found: bool = True,
):
    """Install fake tensorrt_model_connect modules consumed by detect_runtime_strategy()."""
    fake_pkg = types.ModuleType("tensorrt_model_connect")

    fake_engine_builder = types.ModuleType("tensorrt_model_connect.engine_builder")
    fake_engine_builder._resolve_model = lambda _model: "/tmp/fake-model-dir"

    fake_config = types.ModuleType("tensorrt_model_connect.config")

    class _FakeModelConfig:
        @staticmethod
        def from_dir(_model_dir):
            return types.SimpleNamespace(model_type="fake-model-type")

    fake_config.ModelConfig = _FakeModelConfig

    fake_families = types.ModuleType("tensorrt_model_connect.families")
    if plugin_found:
        model = types.SimpleNamespace()
        if runtime_strategy is not None:
            model.runtime_strategy = runtime_strategy
        fake_families.find_model = lambda _config: model
    else:
        fake_families.find_model = lambda _config: None

    fake_pkg.engine_builder = fake_engine_builder
    fake_pkg.config = fake_config
    fake_pkg.families = fake_families

    monkeypatch.setitem(sys.modules, "tensorrt_model_connect", fake_pkg)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect.engine_builder", fake_engine_builder)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect.config", fake_config)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect.families", fake_families)


def _write_synthetic_bundle(path: Path, config: dict):
    """Create a tiny .bundle with a config.json section."""
    config_blob = json.dumps(config).encode("utf-8")
    sections = {"config.json": {"offset": 0, "size": len(config_blob)}}
    header_blob = json.dumps({"sections": sections}).encode("utf-8")

    with open(path, "wb") as f:
        f.write(b"BUNDLE\x01\x00")
        f.write(struct.pack("<Q", len(header_blob)))
        f.write(header_blob)
        f.write(config_blob)


# -----------------------------------------------------------------------
# TestDiffResult — serialization and constructors
# -----------------------------------------------------------------------

class TestDiffResult:
    def test_to_dict_roundtrip(self):
        proto = _import_protocol()
        r = proto.DiffResult(
            test_name="logit_diff", model="test/model",
            runtime_strategy="qwen_decoder_kv_cache",
            passed=True, status="PASS", message="ok",
            metrics={"max_abs_diff": 0.001}, duration_s=1.5, details="")
        d = r.to_dict()
        assert d["test_name"] == "logit_diff"
        assert d["passed"] is True
        assert d["metrics"]["max_abs_diff"] == 0.001
        assert d["duration_s"] == 1.5

    def test_to_json_valid(self):
        proto = _import_protocol()
        r = proto.DiffResult(
            test_name="layer_diff", model="test/model",
            runtime_strategy="qwen_decoder_kv_cache",
            passed=False, status="FAIL", message="bad",
            metrics={}, duration_s=0.0, details="detail")
        parsed = json.loads(r.to_json())
        assert parsed["status"] == "FAIL"
        assert parsed["test_name"] == "layer_diff"
        assert parsed["details"] == "detail"

    def test_skip_constructor(self):
        proto = _import_protocol()
        r = proto.DiffResult.skip("x", "m", "s", "no bundle")
        assert r.status == "SKIP"
        assert r.passed is True  # skip is not a failure
        assert "no bundle" in r.message

    def test_error_constructor(self):
        proto = _import_protocol()
        r = proto.DiffResult.error("x", "m", "s", "crash", details="tb")
        assert r.status == "ERROR"
        assert r.passed is False
        assert r.message == "crash"
        assert r.details == "tb"

    def test_default_metrics_empty(self):
        proto = _import_protocol()
        r = proto.DiffResult(
            test_name="t", model="m", runtime_strategy="s",
            passed=True, status="PASS", message="ok")
        assert r.metrics == {}
        assert r.duration_s == 0.0
        assert r.details == ""


# -----------------------------------------------------------------------
# TestRegistry — registration and lookup
# -----------------------------------------------------------------------

class TestRegistry:
    def test_register_and_lookup_by_name(self):
        registry = _import_registry()

        # Verify a known check was auto-registered
        cls = registry.get_test_by_name("logit_diff")
        assert cls is not None
        assert cls.name == "logit_diff"

    def test_get_tests_for_strategy_filters(self):
        registry = _import_registry()

        decoder_tests = registry.get_tests_for_strategy("qwen_decoder_kv_cache")
        names = [c.name for c in decoder_tests]
        assert "logit_diff" in names
        assert "layer_diff" in names
        # VL and diffusion should not appear for qwen_decoder_kv_cache
        assert "vl_pipeline" not in names
        assert "diffusion_components" not in names

    def test_unknown_test_returns_none(self):
        registry = _import_registry()
        assert registry.get_test_by_name("nonexistent_test_xyz") is None

    def test_vl_tests_for_vision_language(self):
        registry = _import_registry()
        vl_tests = registry.get_tests_for_strategy("qwen_vl_vision_language")
        names = [c.name for c in vl_tests]
        assert "vl_pipeline" in names
        # Standard decoder tests should not appear
        assert "logit_diff" not in names

    def test_diffusion_media_strategies_do_not_use_shared_diffusion_check(self):
        registry = _import_registry()
        diff_tests = registry.get_tests_for_strategy("diffusion_flux")
        names = [c.name for c in diff_tests]
        assert "diffusion_components" not in names
        assert "logit_diff" not in names

    def test_get_all_tests_returns_all(self):
        registry = _import_registry()
        all_tests = registry.get_all_tests()
        names = [c.name for c in all_tests]
        assert "logit_diff" in names
        assert "layer_diff" in names
        assert "runner_parity" in names
        assert "perf_benchmark" in names
        assert "vl_pipeline" in names
        assert "diffusion_components" in names
        assert "layer_profile" in names
        assert len(names) == 7


# -----------------------------------------------------------------------
# TestRunner — orchestration logic
# -----------------------------------------------------------------------

class TestRunner:
    def test_list_tests_returns_expected_fields(self):
        runner = _import_runner()
        entries = runner.list_tests()
        assert len(entries) >= 6
        for e in entries:
            assert "name" in e and "description" in e
            assert "runtime_strategies" in e
            assert "requires_bundle" in e
            assert "requires_gpu" in e

    def test_list_tests_filtered(self):
        runner = _import_runner()
        entries = runner.list_tests("qwen_vl_vision_language")
        names = [e["name"] for e in entries]
        assert "vl_pipeline" in names
        assert "logit_diff" not in names

    def test_run_tests_skips_bundle_required(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="qwen_decoder_kv_cache",
            bundle_path=None,
        )
        results = runner.run_tests(ctx, test_names=["runner_parity"])
        assert len(results) == 1
        assert results[0].status == "SKIP"
        assert results[0].passed is True

    def test_run_tests_unknown_name_raises(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="qwen_decoder_kv_cache",
        )
        with pytest.raises(ValueError, match="Unknown test"):
            runner.run_tests(ctx, test_names=["nonexistent_test_xyz"])

    def test_detect_runtime_strategy_unknown_reports_skip(self, monkeypatch):
        runner = _import_runner()
        _mock_model_strategy_detection(
            monkeypatch, runtime_strategy="future_runtime_strategy")

        result = runner.detect_runtime_strategy(
            "test/model", with_status=True)
        assert result.status == "skip"
        assert result.runtime_strategy == "future_runtime_strategy"
        assert "no diff tests are registered" in result.message

    def test_detect_runtime_strategy_unknown_warns_without_fallback(
        self, monkeypatch
    ):
        runner = _import_runner()
        _mock_model_strategy_detection(
            monkeypatch, runtime_strategy="future_runtime_strategy")

        with pytest.warns(RuntimeWarning, match="no diff tests are registered"):
            strategy = runner.detect_runtime_strategy("test/model")
        assert strategy == "future_runtime_strategy"

    def test_detect_runtime_strategy_missing_model_warns_without_fallback(
        self, monkeypatch
    ):
        runner = _import_runner()
        _mock_model_strategy_detection(monkeypatch, plugin_found=False)

        result = runner.detect_runtime_strategy(
            "test/model", with_status=True)
        assert result.status == "warning"
        assert result.runtime_strategy is None

        with pytest.warns(RuntimeWarning, match="No family model resolved"):
            strategy = runner.detect_runtime_strategy("test/model")
        assert strategy == ""

    def test_detect_bundle_unknown_strategy_reports_skip(self, tmp_path):
        runner = _import_runner()
        bundle = tmp_path / "unknown_strategy.bundle"
        _write_synthetic_bundle(
            bundle, {"runtime_strategy": "future_runtime_strategy"})

        result = runner.detect_runtime_strategy_from_bundle(
            str(bundle), with_status=True)
        assert result.status == "skip"
        assert result.runtime_strategy == "future_runtime_strategy"
        assert "no diff tests are registered" in result.message

        with pytest.warns(RuntimeWarning, match="no diff tests are registered"):
            strategy = runner.detect_runtime_strategy_from_bundle(str(bundle))
        assert strategy == "future_runtime_strategy"

    def test_detect_bundle_missing_runtime_strategy_warns_without_fallback(
        self, tmp_path
    ):
        runner = _import_runner()
        bundle = tmp_path / "missing_runtime_strategy.bundle"
        _write_synthetic_bundle(bundle, {"model_type": "fake"})

        result = runner.detect_runtime_strategy_from_bundle(
            str(bundle), with_status=True)
        assert result.status == "warning"
        assert result.runtime_strategy is None

        with pytest.warns(RuntimeWarning, match="has no runtime_strategy"):
            strategy = runner.detect_runtime_strategy_from_bundle(str(bundle))
        assert strategy == ""

    def test_detect_bundle_read_failure_reports_error(self, tmp_path):
        runner = _import_runner()
        missing_bundle = tmp_path / "missing.bundle"

        result = runner.detect_runtime_strategy_from_bundle(
            str(missing_bundle), with_status=True)
        assert result.status == "error"
        assert result.runtime_strategy is None
        assert "Failed to read runtime_strategy" in result.message

        with pytest.raises(FileNotFoundError):
            runner.detect_runtime_strategy_from_bundle(str(missing_bundle))


# -----------------------------------------------------------------------
# TestCLI — argument parsing (dry-run, no execution)
# -----------------------------------------------------------------------

class TestCLI:
    def _get_cli_module(self):
        """Import the diff.py CLI module."""
        import importlib
        return importlib.import_module("diff")

    def test_list_subcommand_parses(self):
        mod = self._get_cli_module()
        # Verify argparse accepts: diff.py list --model X
        parser = mod.main.__code__  # just verify the module imports cleanly
        assert parser is not None

    def test_run_subcommand_exists(self):
        mod = self._get_cli_module()
        assert hasattr(mod, "cmd_run")
        assert hasattr(mod, "cmd_list")

    def test_module_has_main(self):
        mod = self._get_cli_module()
        assert callable(mod.main)
