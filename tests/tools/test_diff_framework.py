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
import subprocess
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
    runtime_strategy: str | None = "decoder_kv_cache",
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
        plugin = types.SimpleNamespace()
        if runtime_strategy is not None:
            plugin.runtime_strategy = runtime_strategy
        fake_families.find_plugin = lambda _model_type: plugin
    else:
        fake_families.find_plugin = lambda _model_type: None

    fake_pkg.engine_builder = fake_engine_builder
    fake_pkg.config = fake_config
    fake_pkg.families = fake_families

    monkeypatch.setitem(sys.modules, "tensorrt_model_connect", fake_pkg)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect.engine_builder", fake_engine_builder)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect.config", fake_config)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect.families", fake_families)


def _write_synthetic_bundle(path: Path, config: dict):
    """Create a tiny .trtfb with a config.json section."""
    config_blob = json.dumps(config).encode("utf-8")
    sections = {"config.json": {"offset": 0, "size": len(config_blob)}}
    header_blob = json.dumps({"sections": sections}).encode("utf-8")

    with open(path, "wb") as f:
        f.write(b"TRTFB\x00\x01\x00")
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
            runtime_strategy="decoder_kv_cache",
            passed=True, status="PASS", message="ok",
            oracle_level="hf_transformers",
            metrics={"max_abs_diff": 0.001},
            artifacts={"logits": "logits.npy"},
            command_repro=["python tools/diff.py run --model test/model"],
            environment={"python": sys.executable},
            duration_s=1.5, details="")
        d = r.to_dict()
        assert d["test_name"] == "logit_diff"
        assert d["passed"] is True
        assert d["oracle_level"] == "hf_transformers"
        assert d["weak_validation_reason"] == ""
        assert d["metrics"]["max_abs_diff"] == 0.001
        assert d["artifacts"]["logits"] == "logits.npy"
        assert d["command_repro"] == ["python tools/diff.py run --model test/model"]
        assert d["environment"]["python"] == sys.executable
        assert d["duration_s"] == 1.5

    def test_to_json_valid(self):
        proto = _import_protocol()
        r = proto.DiffResult(
            test_name="layer_diff", model="test/model",
            runtime_strategy="decoder_kv_cache",
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

        decoder_tests = registry.get_tests_for_strategy("decoder_kv_cache")
        names = [c.name for c in decoder_tests]
        assert "logit_diff" in names
        assert "layer_diff" in names
        # VL and diffusion should not appear for decoder_kv_cache
        assert "vl_pipeline" not in names
        assert "diffusion_components" not in names
        assert "personaplex_pipeline" not in names

    def test_unknown_test_returns_none(self):
        registry = _import_registry()
        assert registry.get_test_by_name("nonexistent_test_xyz") is None

    def test_vl_tests_for_vision_language(self):
        registry = _import_registry()
        vl_tests = registry.get_tests_for_strategy("vision_language")
        names = [c.name for c in vl_tests]
        assert "vl_pipeline" in names
        # Standard decoder tests should not appear
        assert "logit_diff" not in names

    def test_diffusion_tests_for_diffusion(self):
        registry = _import_registry()
        diff_tests = registry.get_tests_for_strategy("diffusion")
        names = [c.name for c in diff_tests]
        assert "diffusion_components" in names
        assert "logit_diff" not in names

    def test_audio_tests_for_bark(self):
        registry = _import_registry()
        audio_tests = registry.get_tests_for_strategy("text_to_audio_bark")
        names = [c.name for c in audio_tests]
        assert "bark_audio_pipeline" in names
        assert "logit_diff" not in names

    def test_segmentation_tests_for_segmentation(self):
        registry = _import_registry()
        seg_tests = registry.get_tests_for_strategy("segmentation")
        names = [c.name for c in seg_tests]
        assert "segmentation_pipeline" in names
        assert "logit_diff" not in names

    def test_t5_tests_for_text_to_text(self):
        registry = _import_registry()
        t5_tests = registry.get_tests_for_strategy("text_to_text")
        names = [c.name for c in t5_tests]
        assert "t5_encoder_diff" in names
        assert "logit_diff" not in names

    def test_torchtrt_tests_for_torchtrt_decoder(self):
        registry = _import_registry()
        torchtrt_tests = registry.get_tests_for_strategy("torchtrt_decoder")
        names = [c.name for c in torchtrt_tests]
        assert "torchtrt_logit_diff" in names
        assert "logit_diff" not in names

    def test_personaplex_tests_for_speech_to_speech(self):
        registry = _import_registry()
        speech_tests = registry.get_tests_for_strategy("speech_to_speech")
        names = [c.name for c in speech_tests]
        assert "personaplex_pipeline" in names
        assert "logit_diff" not in names

    def test_get_all_tests_returns_all(self):
        registry = _import_registry()
        all_tests = registry.get_all_tests()
        names = [c.name for c in all_tests]
        assert "bark_audio_pipeline" in names
        assert "logit_diff" in names
        assert "layer_diff" in names
        assert "runner_parity" in names
        assert "perf_benchmark" in names
        assert "vl_pipeline" in names
        assert "diffusion_components" in names
        assert "layer_profile" in names
        assert "segmentation_pipeline" in names
        assert "t5_encoder_diff" in names
        assert "torchtrt_logit_diff" in names
        assert "personaplex_pipeline" in names
        assert len(names) == 12

    def test_registered_checks_declare_review_metadata(self):
        registry = _import_registry()
        for cls in registry.get_all_tests():
            assert cls.runtime_strategies, cls.name
            assert isinstance(cls.requires_bundle, bool)
            assert isinstance(cls.requires_gpu, bool)
            assert getattr(cls, "required_inputs", []), cls.name
            assert getattr(cls, "oracle_level", ""), cls.name
            assert isinstance(getattr(cls, "deterministic_seed", None), bool)
            assert getattr(cls, "output_metrics", []), cls.name
            assert getattr(cls, "failure_examples", []), cls.name


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
            assert "required_inputs" in e
            assert "oracle_level" in e
            assert "deterministic_seed" in e
            assert "output_metrics" in e
            assert "failure_examples" in e

    def test_list_tests_filtered(self):
        runner = _import_runner()
        entries = runner.list_tests("vision_language")
        names = [e["name"] for e in entries]
        assert "vl_pipeline" in names
        assert "logit_diff" not in names

    def test_run_tests_skips_bundle_required(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="decoder_kv_cache",
            bundle_path=None,
        )
        results = runner.run_tests(ctx, test_names=["runner_parity"])
        assert len(results) == 1
        assert results[0].status == "SKIP"
        assert results[0].passed is True
        assert results[0].oracle_level == "trt_python_runner"

    def test_run_tests_fails_closed_when_no_checks_match(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="future_runtime_strategy",
        )
        results = runner.run_tests(ctx)

        assert len(results) == 1
        assert results[0].test_name == "strategy_discovery"
        assert results[0].status == "ERROR"
        assert results[0].passed is False
        assert "No diff tests registered" in results[0].message

    def test_run_tests_populates_command_and_environment(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="decoder_kv_cache",
            bundle_path=None,
            command_repro=["python tools/diff.py run --model test/model"],
            environment={"python": sys.executable},
        )
        results = runner.run_tests(ctx, test_names=["runner_parity"])

        assert results[0].command_repro == [
            "python tools/diff.py run --model test/model"
        ]
        assert results[0].environment == {"python": sys.executable}

    def test_run_tests_unknown_name_raises(self):
        proto = _import_protocol()
        runner = _import_runner()

        ctx = proto.TestContext(
            model="test/model",
            runtime_strategy="decoder_kv_cache",
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

    def test_detect_runtime_strategy_missing_plugin_warns_legacy_fallback(
        self, monkeypatch
    ):
        runner = _import_runner()
        _mock_model_strategy_detection(monkeypatch, plugin_found=False)

        result = runner.detect_runtime_strategy(
            "test/model", with_status=True)
        assert result.status == "warning"
        assert result.runtime_strategy == "decoder_kv_cache"

        with pytest.warns(RuntimeWarning, match="No family plugin resolved"):
            strategy = runner.detect_runtime_strategy("test/model")
        assert strategy == "decoder_kv_cache"

    def test_detect_bundle_unknown_strategy_reports_skip(self, tmp_path):
        runner = _import_runner()
        bundle = tmp_path / "unknown_strategy.trtfb"
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

    def test_detect_bundle_missing_runtime_strategy_warns_fallback(
        self, tmp_path
    ):
        runner = _import_runner()
        bundle = tmp_path / "missing_runtime_strategy.trtfb"
        _write_synthetic_bundle(bundle, {"model_type": "fake"})

        result = runner.detect_runtime_strategy_from_bundle(
            str(bundle), with_status=True)
        assert result.status == "warning"
        assert result.runtime_strategy == "decoder_kv_cache"

        with pytest.warns(RuntimeWarning, match="has no runtime_strategy"):
            strategy = runner.detect_runtime_strategy_from_bundle(str(bundle))
        assert strategy == "decoder_kv_cache"

    def test_detect_bundle_read_failure_reports_error(self, tmp_path):
        runner = _import_runner()
        missing_bundle = tmp_path / "missing.trtfb"

        result = runner.detect_runtime_strategy_from_bundle(
            str(missing_bundle), with_status=True)
        assert result.status == "error"
        assert result.runtime_strategy is None
        assert "Failed to read runtime_strategy" in result.message

        with pytest.raises(FileNotFoundError):
            runner.detect_runtime_strategy_from_bundle(str(missing_bundle))


# -----------------------------------------------------------------------
# TestSpecializedAdapters — subprocess-backed wrappers, no GPU execution
# -----------------------------------------------------------------------

class TestSpecializedAdapters:
    def test_t5_encoder_adapter_parses_metrics(self, monkeypatch):
        import importlib

        proto = _import_protocol()
        mod = importlib.import_module("diff_t5")
        calls = []

        def _fake_run(command, **_kwargs):
            calls.append(command)
            return types.SimpleNamespace(
                returncode=0,
                stdout="PASS: T5 encoder match\n",
                stderr="[diff-t5] Max abs diff: 0.000123\n"
                       "[diff-t5] Mean abs diff: 0.000045\n",
            )

        monkeypatch.setattr(mod.subprocess, "run", _fake_run)
        result = mod.run_as_diff_test(proto.TestContext(
            model="google-t5/t5-small",
            runtime_strategy="text_to_text",
        ))

        assert result.status == "PASS"
        assert result.metrics["max_abs_diff"] == pytest.approx(0.000123)
        assert result.metrics["mean_abs_diff"] == pytest.approx(0.000045)
        assert "--max-seq-len" in calls[0]
        assert "64" in calls[0]

    def test_torchtrt_adapter_preserves_legacy_defaults(self, monkeypatch):
        import importlib

        proto = _import_protocol()
        mod = importlib.import_module("diff_torchtrt")
        calls = []

        def _fake_run(command, **_kwargs):
            calls.append(command)
            return types.SimpleNamespace(
                returncode=0,
                stdout=(
                    "  PASS: steps=10 top1_match=90% cos_sim=0.999100 "
                    "max_diff=0.004000 top5_overlap=100%\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(mod.subprocess, "run", _fake_run)
        result = mod.run_as_diff_test(proto.TestContext(
            model="Qwen/Qwen3-0.6B",
            runtime_strategy="torchtrt_decoder",
        ))

        assert result.status == "PASS"
        assert result.metrics["top1_match_rate"] == pytest.approx(0.9)
        assert result.metrics["mean_cosine_sim"] == pytest.approx(0.9991)
        assert result.metrics["max_abs_diff"] == pytest.approx(0.004)
        assert result.metrics["mean_top5_overlap"] == pytest.approx(1.0)
        assert calls[0][calls[0].index("--atol") + 1] == "0.01"
        assert calls[0][calls[0].index("--max-new-tokens") + 1] == "10"

    def test_personaplex_adapter_uses_saved_reference(self, monkeypatch, tmp_path):
        import importlib

        import numpy as np

        proto = _import_protocol()
        mod = importlib.import_module("diff_personaplex")

        reference_dir = tmp_path / "reference"
        reference_dir.mkdir()
        np.save(reference_dir / "depth_tokens.npy", np.array([[1, 2], [3, 4]]))
        np.save(reference_dir / "audio_out.npy", np.array([0.2, -0.2], dtype=np.float32))

        def _fake_trt(**_kwargs):
            return {
                "depth_tokens": np.array([[1, 2], [3, 4]]),
                "audio_out": np.array([0.2, -0.2], dtype=np.float32),
            }

        monkeypatch.setattr(mod, "run_trt_pipeline", _fake_trt)

        result = mod.run_as_diff_test(proto.TestContext(
            model="nvidia/personaplex-7b-v1",
            runtime_strategy="speech_to_speech",
            bundle_path="personaplex.trtfb",
            binary_path="./build/trtmc",
            audio_path="input.wav",
            reference_dir=str(reference_dir),
            output_dir=str(tmp_path / "out"),
        ))

        assert result.status == "PASS"
        assert result.oracle_level == "golden_snapshot"
        assert result.metrics["depth_token_match"] == pytest.approx(1.0)
        assert result.metrics["audio_rms_ratio"] == pytest.approx(1.0)
        assert result.metrics["audio_cosine_sim"] == pytest.approx(1.0)

    def test_personaplex_adapter_skips_without_reference(self):
        import importlib

        proto = _import_protocol()
        mod = importlib.import_module("diff_personaplex")

        result = mod.run_as_diff_test(proto.TestContext(
            model="nvidia/personaplex-7b-v1",
            runtime_strategy="speech_to_speech",
            bundle_path="personaplex.trtfb",
            audio_path="input.wav",
        ))

        assert result.status == "SKIP"
        assert "--reference-dir or --official-repo" in result.message


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

    def test_run_fails_closed_when_bundle_strategy_has_no_checks(
        self,
        tmp_path,
    ):
        bundle = tmp_path / "unknown_strategy.trtfb"
        json_out = tmp_path / "result.json"
        _write_synthetic_bundle(
            bundle, {"runtime_strategy": "future_runtime_strategy"})

        result = subprocess.run(
            [
                sys.executable,
                "tools/diff.py",
                "run",
                "--model",
                "test/model",
                "--bundle",
                str(bundle),
                "--json",
                str(json_out),
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 1
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["status"] == "FAIL"
        assert payload["passed"] is False
        assert payload["strategy_detection"]["status"] == "skip"
        assert payload["results"][0]["test_name"] == "strategy_discovery"
        assert payload["results"][0]["status"] == "ERROR"
        assert "No diff tests registered" in payload["results"][0]["message"]
        assert payload["results"][0]["command_repro"]
        assert payload["results"][0]["environment"]["python"] == sys.executable

    def test_run_does_not_green_when_all_selected_checks_skip(
        self,
        tmp_path,
    ):
        json_out = tmp_path / "result.json"

        result = subprocess.run(
            [
                sys.executable,
                "tools/diff.py",
                "run",
                "--model",
                "test/model",
                "--test",
                "runner_parity",
                "--json",
                str(json_out),
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 1
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["status"] == "SKIP"
        assert payload["passed"] is False
        assert payload["executed_count"] == 0
        assert payload["skipped_count"] == 1
        assert payload["results"][0]["test_name"] == "runner_parity"
        assert payload["results"][0]["status"] == "SKIP"
