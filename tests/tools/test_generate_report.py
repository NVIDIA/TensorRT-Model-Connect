# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scripts/generate_e2e_report.py — HTML E2E report generator.

Tests cover data loading, modality classification, rendering, file encoding,
and graceful handling of missing/corrupt data.  All tests are pure-Python
with no GPU or TRT dependency.

Trace: ARCH-REPORT-001, UD-REPORT-E2E
Intent: Validate E2E report generator data loading, modality classification, and HTML rendering
Preconditions: scripts/ is on sys.path; synthetic E2E result data is available
Postconditions: Report correctly classifies modalities and produces valid HTML with expected sections
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Lazy import (follows the repo convention for tools tests)
# ---------------------------------------------------------------------------


def _import_report():
    """Import generate_e2e_report from scripts/."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("generate_e2e_report")


def _import_vlm_assessment():
    """Import the VLM assessment report component from scripts/."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("reporting.vlm_assessment")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    name: str = "test-model",
    status: str = "pass",
    task_strategy: str = "text_generation_causal",
    family: str = "example_decoder",
    hf_id: str = "example-org/example-decoder",
    prompt: str = "Hello world",
    trt_text: str = "Hello world! The",
    ref_text: str = "Hello world! The",
    metrics: Dict[str, Any] | None = None,
    timing: Dict[str, float] | None = None,
    detailed_timing: Dict[str, float] | None = None,
    failure_type: str | None = None,
    repro_commands: Dict[str, str] | None = None,
    artifacts: Dict[str, Any] | None = None,
    stage_outputs: Dict[str, Any] | None = None,
    model_name: str | None = None,
) -> Dict[str, Any]:
    """Build a synthetic result.json dict."""
    if metrics is None:
        metrics = {
            "logit_cosine_p5": {
                "value": 0.998,
                "threshold": 0.99,
                "operator": ">=",
                "passed": True,
            },
            "token_agreement_rate": {
                "value": 0.95,
                "threshold": 0.8,
                "operator": ">=",
                "passed": True,
            },
        }
    if timing is None:
        timing = {"build_s": 1.0, "trt_generate_s": 2.5, "ref_generate_s": 3.0}
    return {
        "case_name": name,
        "status": status,
        "failure_type": failure_type,
        "oracle_level": "L1_external_reference",
        "timestamp": "2026-02-28T10:00:00Z",
        "case_config": {
            "name": name,
            "hf_id": hf_id,
            "family": family,
            "runtime_strategy": "example_decoder_decoder_kv_cache",
            "task_strategy": task_strategy,
            "reference_backend": "hf_transformers",
            "inputs": {"prompt": prompt},
            "metadata": {"model_name": model_name or name},
        },
        "env_fingerprint": {
            "gpu_name": "NVIDIA GB300",
            "cuda_version": "CUDA 12.8",
            "tensorrt_version": "10.8.0",
            "python_version": "3.10.12",
        },
        "stages": {
            "generate": {
                "status": "passed",
                "metrics": metrics,
                "message": "All metrics passed",
            }
        },
        "stage_outputs": stage_outputs
        or {
            "trt_generate": {
                "stage_name": "generate",
                "timing_s": 2.5,
                "text": trt_text,
                "data": {},
                "metadata": {},
            },
            "ref_generate": {
                "stage_name": "generate",
                "timing_s": 3.0,
                "text": ref_text,
                "data": {},
                "metadata": {},
            },
        },
        "timing": timing,
        "detailed_timing": detailed_timing or {},
        "repro_commands": repro_commands or {},
        "artifacts": artifacts or {},
    }


def _write_result(tmp_path: Path, name: str, result: Dict[str, Any]) -> Path:
    """Write a result.json into a model subdirectory."""
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return model_dir


def _write_junit(e2e_root: Path, body: str) -> None:
    e2e_root.mkdir(parents=True, exist_ok=True)
    (e2e_root / "junit-gpu0-shared-w0.xml").write_text(
        f'<?xml version="1.0" encoding="utf-8"?><testsuite>{body}</testsuite>',
        encoding="utf-8",
    )


def _make_tiny_png(path: Path) -> None:
    """Write a minimal valid 1x1 red PNG file."""
    # Minimal PNG: 8-byte sig + IHDR + IDAT + IEND
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        raw = ctype + data
        crc = struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + raw + crc

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    raw_data = zlib.compress(b"\x00\xff\x00\x00")  # filter=None, R=255,G=0,B=0
    png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", raw_data) + _chunk(b"IEND", b"")
    path.write_bytes(png)


def _make_tiny_wav(path: Path) -> None:
    """Write a minimal valid WAV file (1 sample, 16-bit mono)."""
    import struct as st

    # RIFF header
    sample = st.pack("<h", 1000)  # one 16-bit sample
    data_chunk = b"data" + st.pack("<I", len(sample)) + sample
    fmt_chunk = b"fmt " + st.pack("<I", 16) + st.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    wav = b"RIFF" + st.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk
    path.write_bytes(wav)


class TestInputMediaPathResolution:
    """Input media from isolated /src results stays portable and confined."""

    def test_rebases_isolated_image_audio_and_video_paths(self, tmp_path):
        mod = _import_report()
        project_dir = tmp_path / "checkout"
        media_dir = project_dir / "tests/e2e/models/example/data"
        media_dir.mkdir(parents=True)
        media = {
            "input.png": _make_tiny_png,
            "input.wav": _make_tiny_wav,
            "input.mp4": lambda path: path.write_bytes(
                b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
            ),
        }

        for filename, writer in media.items():
            target = media_dir / filename
            writer(target)
            isolated_path = f"/src/tests/e2e/models/example/data/{filename}"

            assert mod._resolve_input_media(isolated_path, project_dir) == target.resolve()
            assert mod._embeddable(target)

    def test_rebase_rejects_other_absolute_traversal_and_symlink_paths(self, tmp_path):
        mod = _import_report()
        project_dir = tmp_path / "checkout"
        source_dir = project_dir / "tests/e2e/models/example/data"
        source_dir.mkdir(parents=True)
        outside = tmp_path / "outside.wav"
        _make_tiny_wav(outside)
        (source_dir / "escape.wav").symlink_to(outside)

        assert mod._resolve_input_media(str(outside), project_dir) is None
        assert mod._resolve_input_media("/src/../outside.wav", project_dir) is None
        assert (
            mod._resolve_input_media(
                "/src/tests/e2e/models/example/data/escape.wav", project_dir
            )
            is None
        )
        assert mod._resolve_input_media("/src-adjacent/input.wav", project_dir) is None

    def test_single_model_existing_absolute_project_path_is_unchanged(self, tmp_path):
        mod = _import_report()
        project_dir = tmp_path / "standalone-source"
        source = project_dir / "tests/e2e/models/example/data/input.wav"
        source.parent.mkdir(parents=True)
        _make_tiny_wav(source)

        assert mod._resolve_input_media(source, project_dir) == source.resolve()


# ---------------------------------------------------------------------------
# Tests: classify_modality
# ---------------------------------------------------------------------------


class TestClassifyModality:
    """Tests for classify_modality()."""

    def test_text_generation(self):
        mod = _import_report()
        r = _make_result(task_strategy="text_generation_causal")
        assert mod.classify_modality(r) == "text"

    def test_vision_language(self):
        mod = _import_report()
        r = _make_result(task_strategy="vision_language_generation")
        assert mod.classify_modality(r) == "vl"

    def test_diffusion(self):
        mod = _import_report()
        r = _make_result(task_strategy="diffusion_media_generation")
        assert mod.classify_modality(r) == "diffusion"

    def test_audio_strategies(self):
        mod = _import_report()
        for ts in ("text_to_audio", "speech_to_text", "speech_to_speech"):
            r = _make_result(task_strategy=ts)
            assert mod.classify_modality(r) == "audio", f"Failed for {ts}"

    def test_segmentation_strategies(self):
        mod = _import_report()
        for ts in ("segmentation", "prompted_segmentation"):
            r = _make_result(task_strategy=ts)
            assert mod.classify_modality(r) == "segmentation", f"Failed for {ts}"

    def test_numeric_and_reranking_strategies(self):
        mod = _import_report()
        for ts in ("encoder_only_nlp", "embedding"):
            r = _make_result(task_strategy=ts)
            assert mod.classify_modality(r) == "numeric", f"Failed for {ts}"
        assert mod.classify_modality(
            _make_result(task_strategy="reranking")) == "reranking"

    def test_all_live_specialized_strategies(self):
        mod = _import_report()
        expected = {
            "diffusion_text_generation": "diffusion_text",
            "image_classification": "classification",
            "neural_operator": "neural_operator",
            "object_detection": "detection",
            "omni_multimodal": "omni",
        }
        for strategy, modality in expected.items():
            assert mod.classify_modality(
                _make_result(task_strategy=strategy)) == modality

    def test_unknown_strategy_defaults_generic(self):
        mod = _import_report()
        r = _make_result(task_strategy="some_future_strategy")
        assert mod.classify_modality(r) == "generic"

    def test_missing_case_config(self):
        mod = _import_report()
        r = {"case_name": "x"}
        assert mod.classify_modality(r) == "generic"

    def test_every_manifest_task_strategy_has_an_explicit_renderer(self):
        mod = _import_report()
        repo_root = Path(__file__).resolve().parents[2]
        strategies = set()
        for path in (repo_root / "tests/e2e/models").glob("*/manifests/*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("task_strategy"):
                strategies.add(payload["task_strategy"])
        assert strategies
        assert strategies <= set(mod._TASK_STRATEGY_TO_MODALITY)


# ---------------------------------------------------------------------------
# Tests: load_all_results
# ---------------------------------------------------------------------------


class TestLoadAllResults:
    """Tests for load_all_results()."""

    def test_loads_valid_results(self, tmp_path):
        mod = _import_report()
        r1 = _make_result(name="model-a")
        r2 = _make_result(name="model-b", status="fail")
        _write_result(tmp_path, "model-a", r1)
        _write_result(tmp_path, "model-b", r2)
        results = mod.load_all_results(tmp_path)
        assert len(results) == 2
        names = {r["case_name"] for r in results}
        assert names == {"model-a", "model-b"}

    def test_skips_corrupt_json(self, tmp_path):
        mod = _import_report()
        _write_result(tmp_path, "good", _make_result(name="good"))
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "result.json").write_text("{invalid json", encoding="utf-8")
        results = mod.load_all_results(tmp_path)
        assert len(results) == 1
        assert results[0]["case_name"] == "good"

    def test_empty_dir(self, tmp_path):
        mod = _import_report()
        assert mod.load_all_results(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        mod = _import_report()
        assert mod.load_all_results(tmp_path / "does_not_exist") == []

    def test_stashes_artifact_dir(self, tmp_path):
        mod = _import_report()
        _write_result(tmp_path, "m1", _make_result(name="m1"))
        results = mod.load_all_results(tmp_path)
        assert results[0]["_artifact_dir"] == str(tmp_path / "m1")

    def test_xfail_result_is_rendered_as_waived_skip(self, tmp_path):
        mod = _import_report()
        e2e_root = tmp_path / "e2e_artifacts"
        artifacts_dir = e2e_root / "artifacts"
        _write_result(
            artifacts_dir,
            "encoder-base",
            _make_result(
                name="encoder-base",
                status="fail",
                failure_type="compare_fail",
            ),
        )
        _write_junit(
            e2e_root,
            """
            <testcase classname="tests.test_e2e" name="test_e2e[encoder-base]">
              <skipped type="pytest.xfail" message="(known representation parity gap)" />
            </testcase>
            """,
        )

        results = mod.load_all_results(artifacts_dir)
        assert results[0]["status"] == "skip"
        assert results[0]["_raw_status"] == "fail"

        html = mod.render_report(results)
        assert "0 Failed" in html
        assert "1 Skipped" in html
        assert "Pytest outcome: <strong>XFAIL</strong>" in html
        assert "known representation parity gap" in html
        assert "Failure type: <strong>compare_fail</strong>" not in html

    def test_model_owned_xfail_result_is_rendered_as_waived_skip(self, tmp_path):
        mod = _import_report()
        e2e_root = tmp_path / "e2e_artifacts"
        artifacts_dir = e2e_root / "artifacts"
        _write_result(
            artifacts_dir,
            "fnet-base",
            _make_result(
                name="fnet-base",
                status="fail",
                failure_type="compare_fail",
            ),
        )
        _write_junit(
            e2e_root,
            """
            <testcase classname="tests.e2e.models.fnet.test_fnet_e2e"
                      name="test_model_e2e[fnet-base]">
              <skipped type="pytest.xfail"
                       message="(encoder representation parity below minimum contract floor)" />
            </testcase>
            """,
        )

        results = mod.load_all_results(artifacts_dir)
        assert results[0]["status"] == "skip"
        assert results[0]["_raw_status"] == "fail"

        html = mod.render_report(results)
        assert "0 Failed" in html
        assert "1 Skipped" in html
        assert "Pytest outcome: <strong>XFAIL</strong>" in html
        assert "encoder representation parity below minimum contract floor" in html
        assert "Failure type: <strong>compare_fail</strong>" not in html

    def test_model_results_are_rendered_as_testcase_list(self, tmp_path):
        mod = _import_report()
        e2e_root = tmp_path / "e2e_artifacts"
        artifacts_dir = e2e_root / "artifacts"
        base_result = _make_result(
            name="canary-1b-v2",
            task_strategy="speech_to_text",
            family="canary",
            hf_id="nvidia/canary-1b-v2",
        )
        _write_result(artifacts_dir, "canary-1b-v2", base_result)
        for probe_name in (
            "canary-1b-v2-asr-probe01",
            "canary-1b-v2-asr-probe02",
        ):
            _write_result(
                artifacts_dir,
                probe_name,
                _make_result(
                    name=probe_name,
                    task_strategy="speech_to_text",
                    family="canary",
                    hf_id="nvidia/canary-1b-v2",
                    model_name="canary-1b-v2",
                ),
            )
        _write_junit(
            e2e_root,
            """
            <testcase classname="tests.e2e.models.canary.test_canary_e2e"
                      name="test_model_e2e[canary-1b-v2]" />
            <testcase classname="tests.e2e.models.canary.test_canary_e2e"
                      name="test_model_e2e[canary-1b-v2-asr-probe01]" />
            <testcase classname="tests.e2e.models.canary.test_canary_e2e"
                      name="test_model_e2e[canary-1b-v2-asr-probe02]">
              <failure message="probe comparison failed" />
            </testcase>
            """,
        )

        results = mod.load_all_results(artifacts_dir)
        names = {result["case_name"] for result in results}
        assert names == {
            "canary-1b-v2",
            "canary-1b-v2-asr-probe01",
            "canary-1b-v2-asr-probe02",
        }
        assert len(results) == 3
        probe02 = next(
            result
            for result in results
            if result["case_name"] == "canary-1b-v2-asr-probe02"
        )
        assert probe02["status"] == "fail"
        assert probe02["_pytest_outcome"]["reason"] == "probe comparison failed"

        html = mod.render_report(results)
        assert "Grouped Bundle Testcases" not in html
        assert html.count('class="summary-row"') == 1
        assert 'class="summary-model-details"' in html
        assert 'class="summary-subtest-table"' in html
        assert "canary-1b-v2-asr-probe01" in html
        assert "canary-1b-v2-asr-probe02" in html
        assert "3 testcases" in html

    def test_parent_pytest_node_does_not_create_schema_less_result(self, tmp_path):
        mod = _import_report()
        e2e_root = tmp_path / "e2e_artifacts"
        artifacts_dir = e2e_root / "artifacts"
        parent_name = "multi-case-model"
        for suffix in ("ar", "diffusion", "linear-spec"):
            case_name = f"{parent_name}-{suffix}"
            _write_result(
                artifacts_dir,
                case_name,
                _make_result(name=case_name, model_name=parent_name),
            )
        _write_junit(
            e2e_root,
            f"""
            <testcase classname="tests.e2e.models.multi.test_multi_e2e"
                      name="test_model_e2e[{parent_name}]" />
            """,
        )

        results = mod.load_all_results(artifacts_dir)

        assert {result["case_name"] for result in results} == {
            f"{parent_name}-ar",
            f"{parent_name}-diffusion",
            f"{parent_name}-linear-spec",
        }
        assert all(result["case_config"] for result in results)
        assert all(
            result["_pytest_model_outcome"]["pytest_status"] == "PASSED"
            for result in results
        )
        assert mod.validate_evidence(results, project_dir=None) == []


class TestLoadModelManifests:
    """Tests for the model and testcase inventory shown in the summary."""

    def test_loads_every_model_with_its_testcases(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "canary"
        manifests_dir = model_dir / "manifests"
        manifests_dir.mkdir(parents=True)
        (model_dir / "MODEL.toml").write_text(
            'test_manifests = ["manifests/multi.json", "manifests/single.json"]\n',
            encoding="utf-8",
        )
        (manifests_dir / "multi.json").write_text(
            json.dumps(
                {
                    "name": "canary-1b-v2",
                    "family": "canary",
                    "bundle": "canary.trtfb",
                    "task_strategy": "speech_to_text",
                    "testcases": [
                        {"name": "canary-1b-v2"},
                        {
                            "name": "canary-1b-v2-asr-probe01",
                            "ci_tier": "nightly_only",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        (manifests_dir / "single.json").write_text(
            json.dumps(
                {
                    "name": "single",
                    "family": "example",
                    "testcases": [{"name": "single"}],
                }
            ),
            encoding="utf-8",
        )

        models = mod.load_model_manifests(tmp_path)

        assert [model["name"] for model in models] == ["canary-1b-v2", "single"]
        assert models[0]["testcases"] == [
            {
                "name": "canary-1b-v2",
                "ci_tier": "default",
                "task_strategy": "speech_to_text",
            },
            {
                "name": "canary-1b-v2-asr-probe01",
                "ci_tier": "nightly_only",
                "task_strategy": "speech_to_text",
            },
        ]
        assert models[1]["testcases"] == [
            {
                "name": "single",
                "ci_tier": "default",
                "task_strategy": "",
            }
        ]


# ---------------------------------------------------------------------------
# Tests: encode_file_base64
# ---------------------------------------------------------------------------


class TestEncodeFileBase64:
    """Tests for encode_file_base64()."""

    def test_encodes_small_file(self, tmp_path):
        mod = _import_report()
        p = tmp_path / "test.png"
        _make_tiny_png(p)
        uri = mod.encode_file_base64(p, "image/png")
        assert uri is not None
        assert uri.startswith("data:image/png;base64,")

    def test_returns_none_for_missing(self, tmp_path):
        mod = _import_report()
        assert mod.encode_file_base64(tmp_path / "nope.png", "image/png") is None

    def test_returns_none_for_oversized(self, tmp_path):
        mod = _import_report()
        p = tmp_path / "big.bin"
        # Write a file just over the limit
        p.write_bytes(b"\x00" * (mod._MAX_EMBED_BYTES + 1))
        assert mod.encode_file_base64(p, "application/octet-stream") is None


# ---------------------------------------------------------------------------
# Tests: render_report (integration)
# ---------------------------------------------------------------------------


class TestRenderReport:
    """Integration tests for the full render_report pipeline."""

    def test_external_assets_are_loaded_from_files(self):
        mod = _import_report()
        css = mod._load_report_asset(mod._REPORT_CSS_FILENAME)
        js = mod._load_report_asset(mod._REPORT_JS_FILENAME)

        assert ".summary-table" in css
        assert "function filterModels()" in js

    def test_render_report_embeds_loaded_external_assets(self, tmp_path, monkeypatch):
        mod = _import_report()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / mod._REPORT_CSS_FILENAME).write_text(
            ".external-asset-test { color: red; }\n",
            encoding="utf-8",
        )
        (assets_dir / mod._REPORT_JS_FILENAME).write_text(
            "function externalAssetTest() { return true; }\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "_REPORT_ASSETS_DIR", assets_dir)

        html = mod.render_report([], title="External Asset Report")

        assert "<style>.external-asset-test { color: red; }\n</style>" in html
        assert "<script>function externalAssetTest() { return true; }\n</script>" in html

    def test_empty_results(self):
        mod = _import_report()
        html = mod.render_report([], title="Empty Report")
        assert "<!DOCTYPE html>" in html
        assert "Empty Report" in html
        assert "0 Results" in html

    def test_single_text_model(self):
        mod = _import_report()
        r = _make_result(name="example-decoder", prompt="Test prompt")
        html = mod.render_report([r], title="Test Report")
        assert "example-decoder" in html
        assert "Test prompt" in html
        assert "PASS" in html
        assert "logit_cosine_p5" in html
        assert "1 Passed" in html

    def test_failed_model_shows_failure_type(self):
        mod = _import_report()
        r = _make_result(name="bad-model", status="fail", failure_type="compare_fail")
        html = mod.render_report([r])
        assert "compare_fail" in html
        assert "FAIL" in html
        assert "1 Failed" in html

    def test_multiple_models_all_present(self):
        mod = _import_report()
        results = [
            _make_result(name=f"model-{i}", status="pass" if i % 2 == 0 else "fail")
            for i in range(5)
        ]
        html = mod.render_report(results)
        for i in range(5):
            assert f"model-{i}" in html
        assert "3 Passed" in html
        assert "2 Failed" in html

    def test_repro_commands_rendered(self):
        mod = _import_report()
        r = _make_result(
            repro_commands={
                "build_bundle": "./build/trtmc build X -o y.trtfb",
                "trt_inference": "./trtmc run y.trtfb --prompt 'Hi'",
            }
        )
        html = mod.render_report([r])
        assert "./build/trtmc build X" in html
        assert "Copy" in html

    def test_timing_rendered(self):
        mod = _import_report()
        r = _make_result(timing={"build_s": 10.0, "trt_generate_s": 5.5})
        html = mod.render_report([r])
        assert "Detailed Timing" in html
        assert "Weights loading" in html
        assert "TRT compile" in html
        assert "Inference" in html
        assert "Comparison" in html
        assert "TRT validation wall time" not in html
        assert "<summary>Raw Timing Phases</summary>" in html
        assert "10.00s" in html
        assert "5.50s" in html
        assert "15.50s" in html  # total

    def test_raw_trt_engine_timing_is_inference(self):
        mod = _import_report()
        r = _make_result(
            timing={
                "trt_generate_s": 5.5,
                "trt_engine_generate_s": 1.25,
                "trt_load_deserialize_generate_s": 0.75,
            }
        )
        details = mod._normalize_detailed_timing(r)
        assert details["inference_s"] == 1.25
        assert details["trt_load_deserialization_s"] == 0.75
        assert details["trt_validation_s"] == 5.5

    def test_raw_trt_wall_time_is_not_inference(self):
        mod = _import_report()
        r = _make_result(
            timing={"trt_generate_s": 5.5, "trt_load_deserialize_generate_s": 0.75},
            detailed_timing={"inference_s": 5.5},
        )
        details = mod._normalize_detailed_timing(r)
        assert "inference_s" not in details
        assert details["trt_load_deserialization_s"] == 0.75
        assert details["trt_validation_s"] == 5.5

    def test_raw_trt_engine_timing_overrides_persisted_inference(self):
        mod = _import_report()
        r = _make_result(
            timing={"trt_generate_s": 5.5, "trt_engine_generate_s": 1.25},
            detailed_timing={"inference_s": 5.5},
        )
        details = mod._normalize_detailed_timing(r)
        assert details["inference_s"] == 1.25

    def test_detailed_timing_rendered_from_result(self):
        mod = _import_report()
        r = _make_result(
            timing={
                "bundle_build_s": 20.0,
                "trt_generate_s": 3.0,
                "trt_engine_generate_s": 3.0,
                "trt_load_deserialize_generate_s": 1.0,
                "contract_generate_s": 0.5,
            },
            detailed_timing={
                "weights_loading_s": 4.0,
                "trt_compile_s": 15.0,
                "inference_s": 3.0,
                "comparison_s": 0.5,
            },
        )
        html = mod.render_report([r])
        assert "Weights loading" in html
        assert "4.00s" in html
        assert "TRT compile" in html
        assert "15.00s" in html
        assert "TRT engine execution" in html
        assert "TRT engine load/deserialization" in html
        assert "4.00s" in html
        assert "1.00s" in html
        assert "Comparison" in html
        assert "0.50s" in html
        assert "TRT validation wall time" not in html

    def test_component_weight_and_compile_rows_rendered(self):
        mod = _import_report()
        r = _make_result(
            detailed_timing={
                "weights_loading_s": 13.0,
                "weights_loading_decoder_block_s": 8.0,
                "weights_loading_denoiser_s": 5.0,
                "trt_compile_s": 88.0,
                "trt_compile_decoder_block_s": 30.0,
                "trt_compile_denoiser_s": 50.0,
            },
        )
        html = mod.render_report([r])
        assert "<summary>Weights loading</summary>" in html
        assert "decoder block" in html
        assert "8.00s" in html
        assert "denoiser" in html
        assert "5.00s" in html
        assert "<summary>TRT compile</summary>" in html
        assert "decoder block" in html
        assert "30.00s" in html
        assert "denoiser" in html
        assert "50.00s" in html
        assert "unattributed" in html
        assert "8.00s" in html

    def test_detailed_timing_table_folds_compile_breakdown(self):
        mod = _import_report()
        r = _make_result(
            detailed_timing={
                "weights_loading_s": 1.0,
                "trt_compile_s": 10.0,
                "trt_compile_main_engine_s": 6.0,
                "trt_compile_vision_engine_s": 4.0,
                "inference_s": 2.0,
                "comparison_s": 0.1,
            },
        )
        html = mod.render_report([r])
        assert "<summary>TRT compile</summary>" in html
        assert "10.00s" in html
        assert "main engine" in html
        assert "vision engine" in html

    def test_detailed_timing_table_folds_component_inference_breakdown(self):
        mod = _import_report()
        r = _make_result(
            timing={
                "trt_end_to_end_s": 9.0,
                "trt_engine_end_to_end_s": 0.75,
                "trt_component_engine_end_to_end_denoiser_plan_s": 0.6,
                "trt_component_engine_end_to_end_vae_decoder_plan_s": 0.15,
                "trt_load_deserialize_end_to_end_s": 3.0,
                "trt_component_load_deserialize_end_to_end_denoiser_plan_s": 2.5,
                "trt_component_load_deserialize_end_to_end_vae_decoder_plan_s": 0.5,
            },
            detailed_timing={
                "weights_loading_s": 1.0,
                "trt_compile_s": 2.0,
                "comparison_s": 0.1,
            },
        )
        html = mod.render_report([r])
        assert "<summary>Inference</summary>" in html
        assert "engine execution: denoiser" in html
        assert "engine execution: vae decoder" in html
        assert "load/deserialization: denoiser" in html
        assert "load/deserialization: vae decoder" in html
        assert "3.75s" in html
        assert "TRT validation wall time" not in html

    def test_detailed_timing_table_sums_component_inference_across_stages(self):
        mod = _import_report()
        r = _make_result(
            timing={
                "trt_engine_prefill_s": 0.25,
                "trt_engine_decode_s": 0.75,
                "trt_component_engine_prefill_engine_plan_s": 0.25,
                "trt_component_engine_decode_engine_plan_s": 0.75,
                "trt_load_deserialize_prefill_s": 0.10,
                "trt_load_deserialize_decode_s": 0.20,
                "trt_component_load_deserialize_prefill_engine_plan_s": 0.10,
                "trt_component_load_deserialize_decode_engine_plan_s": 0.20,
            },
            detailed_timing={
                "weights_loading_s": 1.0,
                "trt_compile_s": 2.0,
            },
        )
        html = mod.render_report([r])
        assert "engine execution: engine" in html
        assert "load/deserialization: engine" in html
        assert "1.00s" in html
        assert "0.30s" in html
        assert "1.30s" in html

    def test_detailed_timing_table_recovers_runtime_timings_from_stage_outputs(self):
        mod = _import_report()
        r = _make_result(
            timing={"trt_end_to_end_s": 9.0},
            detailed_timing={
                "weights_loading_s": 1.0,
                "trt_compile_s": 2.0,
                "tokenizer_json_ensure_s": 0.0,
            },
            stage_outputs={
                "trt_end_to_end": {
                    "stage_name": "end_to_end",
                    "data": {
                        "stderr": "\n".join(
                            [
                                '[trtmc.load_timing] label="denoiser_plan" '
                                "load_deserialize_ms=2500.000000 plan_bytes=1",
                                '[trtmc.engine_timing] label="denoiser_plan" '
                                "execute_ms=600.000000 launches=20",
                            ]
                        ),
                    },
                },
            },
        )
        html = mod.render_report([r])
        assert "engine execution: denoiser" in html
        assert "load/deserialization: denoiser (1 B plan)" in html
        assert "3.10s" in html
        assert "Tokenizer JSON ensure" not in html

    def test_detailed_timing_does_not_double_count_saved_stderr_log(self, tmp_path):
        mod = _import_report()
        log_text = "\n".join(
            [
                '[trtmc.load_timing] label="denoiser_plan" '
                "load_deserialize_ms=2500.000000 plan_bytes=1024",
                '[trtmc.engine_timing] label="denoiser_plan" execute_ms=600.000000 launches=20',
            ]
        )
        log_path = tmp_path / "end_to_end_stderr.log"
        log_path.write_text(log_text, encoding="utf-8")
        r = _make_result(
            timing={"trt_end_to_end_s": 9.0},
            stage_outputs={
                "trt_end_to_end": {
                    "stage_name": "end_to_end",
                    "data": {
                        "stderr": log_text,
                        "stderr_log": str(log_path),
                    },
                },
            },
        )
        r["_artifact_dir"] = str(tmp_path)
        details = mod._normalize_detailed_timing(r)
        assert details["trt_load_deserialization_s"] == 2.5
        assert details["inference_s"] == 0.6

    def test_detailed_timing_table_sums_load_plan_sizes_across_stages(self):
        mod = _import_report()
        r = _make_result(
            timing={},
            stage_outputs={
                "trt_prefill": {
                    "stage_name": "prefill",
                    "data": {
                        "stderr": (
                            '[trtmc.load_timing] label="engine_plan" '
                            "load_deserialize_ms=100.000000 plan_bytes=1073741824"
                        ),
                    },
                },
                "trt_decode": {
                    "stage_name": "decode",
                    "data": {
                        "stderr": (
                            '[trtmc.load_timing] label="engine_plan" '
                            "load_deserialize_ms=200.000000 plan_bytes=2147483648"
                        ),
                    },
                },
            },
        )
        html = mod.render_report([r])
        assert "load/deserialization: engine (2 loads, 3.0 GiB total plan bytes)" in html
        assert "0.30s" in html

    def test_detailed_timing_table_replaces_generic_extra_compile_with_components(self):
        mod = _import_report()
        r = _make_result(
            detailed_timing={
                "weights_loading_s": 1.0,
                "trt_compile_s": 20.0,
                "trt_compile_main_engine_s": 5.0,
                "trt_compile_extra_engines_s": 15.0,
                "trt_compile_extra_speech_depth_codebook_00_s": 4.0,
                "trt_compile_extra_mimi_audio_decoder_s": 9.0,
                "trt_compile_extra_mimi_audio_encoder_s": 2.0,
            },
        )
        html = mod.render_report([r])
        assert "speech depth codebook 00" in html
        assert "mimi audio decoder" in html
        assert "mimi audio encoder" in html
        assert "extra engines" not in html

    def test_detailed_timing_table_excludes_overlapping_build_summaries(self):
        mod = _import_report()
        r = _make_result(
            timing={"bundle_build_s": 30.0},
            detailed_timing={
                "weights_loading_s": 1.0,
                "trt_compile_s": 20.0,
                "build_total_s": 25.0,
                "bundle_total_s": 30.0,
                "build_overhead_s": 4.0,
                "inference_s": 2.0,
                "comparison_s": 0.1,
            },
        )
        html = mod.render_report([r])
        assert "Weights loading" in html
        assert "TRT compile" in html
        assert "20.00s" in html
        assert "Bundle build total" not in html
        assert "Build overhead" not in html

    def test_detailed_timing_uses_structured_result_only(self, tmp_path):
        mod = _import_report()
        r = _make_result(
            timing={
                "bundle_build_s": 30.0,
                "trt_generate_s": 3.0,
                "ref_generate_s": 2.0,
                "compare_generate_s": 0.2,
            },
            detailed_timing={
                "weights_loading_s": 4.5,
                "trt_compile_s": 20.0,
                "bundle_write_s": 1.0,
            },
        )
        model_dir = _write_result(tmp_path, "test-model", r)
        (model_dir / "e2e_run.log").write_text(
            "\n".join(
                [
                    "[trtmc build] Weights loaded [999.0s]",
                    "[trtmc build] Engine built [888.0s] (10.0 MB)",
                ]
            ),
            encoding="utf-8",
        )
        loaded = mod.load_all_results(tmp_path)[0]
        html = mod.render_report([loaded])
        assert "4.50s" in html
        assert "20.00s" in html
        assert "999.00s" not in html
        assert "888.00s" not in html
        assert "Bundle write" in html
        assert "1.00s" in html
        assert "3.00s" in html
        assert "0.20s" in html

    def test_env_section_rendered(self):
        mod = _import_report()
        r = _make_result()
        html = mod.render_report([r])
        assert "NVIDIA GB300" in html
        assert "CUDA 12.8" in html

    def test_html_escaping(self):
        mod = _import_report()
        r = _make_result(
            name="xss<test>",
            prompt='<script>alert("xss")</script>',
            trt_text="a < b && c > d",
        )
        html = mod.render_report([r])
        # User-supplied XSS payload must be escaped (the report's own
        # inline <script> for copyCmd/filterModels is expected).
        assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html
        assert "a &lt; b" in html


# ---------------------------------------------------------------------------
# Tests: modality-specific renderers
# ---------------------------------------------------------------------------


class TestRenderTextModel:
    """Tests for render_text_model()."""

    def test_shows_prompt_and_comparison(self):
        mod = _import_report()
        r = _make_result(prompt="Capital of France", trt_text="Paris", ref_text="Paris")
        html = mod.render_text_model(r)
        assert "Capital of France" in html
        assert "TRT Output" in html
        assert "Reference Output" in html
        assert "Paris" in html


class TestRenderGenericModel:
    """Tests for render_generic_model()."""

    def test_shows_encoder_reference_feature_output(self):
        mod = _import_report()
        r = _make_result(
            task_strategy="encoder_only_nlp",
            family="encoder_family",
            stage_outputs={
                "trt_full_inference": {
                    "stage_name": "full_inference",
                    "data": {"cls_embedding": [1.0, 2.0, 3.0]},
                    "metadata": {},
                },
                "ref_full_inference": {
                    "stage_name": "full_inference",
                    "data": {"cls_embedding": [1.0, 2.0, 3.0]},
                    "metadata": {},
                },
            },
        )
        html = mod.render_generic_model(r)
        assert "TRT Output" in html
        assert "Reference Output" in html
        assert "cls_embedding (3 values)" in html
        assert "preview[0:3]" in html
        assert "l2_norm" in html


class TestAdditionalStrategyRenderers:
    def test_diffusion_text_uses_paired_text_renderer(self):
        mod = _import_report()
        result = _make_result(
            task_strategy="diffusion_text_generation",
            prompt="Translate this",
            trt_text="Bonjour",
            ref_text="Bonjour",
        )

        rendered = mod.render_model_section(result, project_dir=None)

        assert "Translate this" in rendered
        assert "TRT Output" in rendered
        assert "Reference Output" in rendered
        assert "Bonjour" in rendered

    def test_diffusion_text_shows_live_sampling_settings(self):
        mod = _import_report()
        result = _make_result(
            task_strategy="diffusion_text_generation",
            trt_text="sample",
            ref_text="sample",
        )
        result["case_config"]["inputs"].update({
            "source_text": "seed text",
            "sampling_method": "ddpm",
            "num_sampling_steps": 32,
        })

        rendered = mod.render_diffusion_text_model(result)

        assert "seed text" in rendered
        assert "ddpm" in rendered
        assert "32" in rendered

    def test_classification_shows_input_and_paired_prediction(self, tmp_path):
        mod = _import_report()
        image = tmp_path / "input.png"
        _make_tiny_png(image)
        result = _make_result(
            task_strategy="image_classification",
            stage_outputs={
                "trt_full_inference": {
                    "data": {"top_class": 42, "top_score": 0.9, "num_classes": 1000},
                },
                "ref_full_inference": {
                    "data": {"top_class": 42, "top_score": 0.89, "num_classes": 1000},
                },
            },
        )
        result["case_config"]["inputs"]["image"] = "input.png"

        rendered = mod.render_classification_model(result, tmp_path)

        assert "Classification Input" in rendered
        assert "data:image/png;base64," in rendered
        assert "Top class" in rendered
        assert "42" in rendered
        assert "0.9000" in rendered
        assert "0.8900" in rendered

    def test_reranking_shows_documents_and_paired_scores(self):
        mod = _import_report()
        result = _make_result(
            task_strategy="reranking",
            stage_outputs={
                "trt_full_inference": {
                    "data": {"documents": ["Mars", "Venus"], "scores": [0.9, 0.1]},
                },
                "ref_full_inference": {
                    "data": {"documents": ["Mars", "Venus"], "scores": [0.8, 0.2]},
                },
            },
        )
        result["case_config"]["inputs"].update({
            "query": "Which is the red planet?",
            "documents": ["Mars", "Venus"],
        })

        rendered = mod.render_reranking_model(result)

        assert "Which is the red planet?" in rendered
        assert "Mars" in rendered
        assert "TRT / Base score" in rendered
        assert "Reference score" in rendered
        assert "TRT rank" in rendered
        assert "Reference rank" in rendered

    def test_neural_operator_shows_plot_and_structured_values(self):
        mod = _import_report()
        result = _make_result(
            task_strategy="neural_operator",
            stage_outputs={
                "trt_full_inference": {
                    "data": {"output_field": [0.0, 1.0, 2.0], "output_shape": [3]},
                },
                "ref_full_inference": {
                    "data": {"output_field": [0.0, 1.1, 2.1], "output_shape": [3]},
                },
            },
        )
        result["case_config"]["inputs"]["branch_input"] = [0.0, 0.5, 1.0]

        rendered = mod.render_neural_operator_model(result)

        assert "Model Inputs" in rendered
        assert "branch_input" in rendered
        assert "Output Series Comparison" in rendered
        assert "<svg" in rendered
        assert "TRT / Base" in rendered
        assert "Reference" in rendered
        assert "output_shape" in rendered

    def test_object_detection_shows_input_and_paired_detections(self, tmp_path):
        mod = _import_report()
        image = tmp_path / "street.png"
        _make_tiny_png(image)
        result = _make_result(
            task_strategy="object_detection",
            stage_outputs={
                "trt_full_inference": {
                    "data": {
                        "detections": [{
                            "label": "car", "score": 0.92,
                            "box": [1.0, 2.0, 8.0, 9.0],
                        }],
                    },
                },
                "ref_full_inference": {
                    "data": {
                        "detections": [{
                            "label": "car", "score": 0.91,
                            "box": [1.1, 2.1, 8.1, 9.1],
                        }],
                    },
                },
            },
        )
        result["case_config"]["inputs"]["image"] = "street.png"

        rendered = mod.render_detection_model(result, tmp_path)

        assert "Detection Input" in rendered
        assert "data:image/png;base64," in rendered
        assert "TRT / Base Detections" in rendered
        assert "Reference Detections" in rendered
        assert "car" in rendered
        assert mod.validate_evidence([result], project_dir=tmp_path) == []

    def test_omni_finds_audio_in_stage_metadata(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "omni"
        model_dir.mkdir()
        for filename in ("trt.wav", "ref.wav"):
            _make_tiny_wav(model_dir / filename)
        result = _make_result(
            name="omni",
            task_strategy="omni_multimodal",
            stage_outputs={
                "trt_talker_decode": {
                    "text": "hello",
                    "data": {"token_ids": [1, 2]},
                    "metadata": {"audio_output_path": str(model_dir / "trt.wav")},
                },
                "ref_talker_decode": {
                    "text": "hello",
                    "data": {"token_ids": [1, 2]},
                    "metadata": {"audio_output_path": str(model_dir / "ref.wav")},
                },
            },
        )
        result["_artifact_dir"] = str(model_dir)

        rendered = mod.render_omni_model(result, project_dir=tmp_path)

        assert "TRT / Base Audio" in rendered
        assert "Reference Audio" in rendered
        assert rendered.count("data:audio/wav;base64,") == 2
        assert "token_ids" in rendered


class TestRenderVlModel:
    """Tests for render_vl_model()."""

    def test_embeds_image(self, tmp_path):
        mod = _import_report()
        img = tmp_path / "test.png"
        _make_tiny_png(img)
        r = _make_result(task_strategy="vision_language_generation")
        r["case_config"]["inputs"]["image"] = str(img.relative_to(tmp_path))
        html = mod.render_vl_model(r, project_dir=tmp_path)
        assert "data:image/png;base64," in html

    def test_missing_image_graceful(self):
        mod = _import_report()
        r = _make_result(task_strategy="vision_language_generation")
        r["case_config"]["inputs"]["image"] = "nonexistent.jpg"
        html = mod.render_vl_model(r, project_dir=Path("/tmp"))
        assert "not found" in html.lower() or "Image" in html


class TestRenderDiffusionModel:
    """Tests for render_diffusion_model()."""

    def test_vlm_assessment_missing_artifact_message(self):
        mod = _import_vlm_assessment()
        html = mod.render_diffusion_vlm_assessment({})
        assert "No VLM assessment artifact was found for this model" in html

    def test_vlm_assessment_malformed_judgment_defaults_to_pass(self):
        mod = _import_vlm_assessment()
        html = mod.render_diffusion_vlm_assessment(
            {
                "vlm_assessment": {
                    "model_id": "Judge <model>",
                    "vlm_judgment": "not a structured judgment",
                }
            }
        )
        assert "Judge &lt;model&gt;" in html
        assert "<strong>Gate:</strong> PASS" in html
        assert "Semantic similarity" not in html

    def test_frame_gallery(self, tmp_path):
        mod = _import_report()
        frames_dir = tmp_path / "model-diff" / "frames"
        frames_dir.mkdir(parents=True)
        for i in range(8):
            _make_tiny_png(frames_dir / f"frame_{i:03d}.png")
        r = _make_result(
            name="model-diff",
            task_strategy="diffusion_media_generation",
        )
        r["_artifact_dir"] = str(tmp_path / "model-diff")
        html = mod.render_diffusion_model(r)
        assert "data:image/png;base64," in html
        # Should have at most _MAX_DIFFUSION_FRAMES images
        count = html.count("data:image/png;base64,")
        assert count <= mod._MAX_DIFFUSION_FRAMES

    def test_no_frames_no_crash(self):
        mod = _import_report()
        r = _make_result(task_strategy="diffusion_media_generation")
        r["_artifact_dir"] = "/nonexistent"
        html = mod.render_diffusion_model(r)
        assert isinstance(html, str)

    def test_trt_frames_dir_artifact_is_expanded(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "model-diff"
        frames_dir = model_dir / "frames"
        frames_dir.mkdir(parents=True)
        for i in range(3):
            _make_tiny_png(frames_dir / f"frame_{i:03d}.png")
        r = _make_result(
            name="model-diff",
            task_strategy="diffusion_media_generation",
            artifacts={"trt_frames": "frames"},
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_diffusion_model(r)
        assert "data:image/png;base64," in html
        assert "Frame too large" not in html

    def test_ref_frames_dir_artifact_is_expanded(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "model-diff"
        ref_frames_dir = model_dir / "ref_frames"
        ref_frames_dir.mkdir(parents=True)
        for i in range(2):
            _make_tiny_png(ref_frames_dir / f"frame_{i:03d}.png")
        r = _make_result(
            name="model-diff",
            task_strategy="diffusion_media_generation",
            artifacts={"ref_frames": "ref_frames"},
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_diffusion_model(r)
        assert "Reference Frames" in html
        assert "data:image/png;base64," in html

    def test_pairs_trt_and_reference_video_frames(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "video-model"
        for dirname in ("frames", "ref_frames"):
            frame_dir = model_dir / dirname
            frame_dir.mkdir(parents=True)
            for index in range(8):
                _make_tiny_png(frame_dir / f"frame_{index:03d}.png")
        result = _make_result(
            name="video-model",
            task_strategy="diffusion_media_generation",
            artifacts={"trt_frames": "frames", "ref_frames": "ref_frames"},
        )
        result["_artifact_dir"] = str(model_dir)

        rendered = mod.render_diffusion_model(result)

        assert "Visual Review: TRT vs Reference" in rendered
        assert "TRT / Base" in rendered
        assert rendered.count("data:image/png;base64,") == 12

    def test_paired_frames_preserve_jpeg_mime_and_unmatched_frame(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "jpeg-video"
        for dirname, count in (("frames", 2), ("ref_frames", 1)):
            frame_dir = model_dir / dirname
            frame_dir.mkdir(parents=True)
            for index in range(count):
                (frame_dir / f"frame_{index:03d}.jpg").write_bytes(
                    b"\xff\xd8\xff\xd9"
                )
        result = _make_result(
            name="jpeg-video",
            task_strategy="diffusion_media_generation",
            artifacts={"trt_frames": "frames", "ref_frames": "ref_frames"},
        )
        result["_artifact_dir"] = str(model_dir)

        rendered = mod.render_diffusion_model(result)

        assert rendered.count("data:image/jpeg;base64,") == 3
        assert "frame_001.jpg" in rendered

    def test_vlm_assessment_is_rendered(self):
        mod = _import_report()
        r = _make_result(
            name="model-diff",
            task_strategy="diffusion_media_generation",
        )
        r["vlm_assessment"] = {
            "model_id": "example-org/example-vl-judge",
            "vlm_judgment": {
                "trt_description": "a clear cat image",
                "hf_description": "a cat image",
                "semantic_similarity_0_to_5": 4.5,
                "trt_prompt_alignment_0_to_5": 4.0,
                "hf_prompt_alignment_0_to_5": 4.0,
                "trt_visual_quality_0_to_5": 3.5,
                "hf_visual_quality_0_to_5": 4.0,
                "trt_relative_to_hf": "similar",
                "is_regression": False,
                "reason": "same main subject",
                "vlm_gate": {"failed": False, "reasons": []},
            },
        }
        html = mod.render_diffusion_model(r)
        assert "VLM Semantic Assessment" in html
        assert "example-org/example-vl-judge" in html
        assert "Semantic similarity" in html
        assert "4.5000" in html
        assert "same main subject" in html

    def test_vlm_assessment_shows_failed_gate_even_if_artifact_has_legacy_waive(self):
        mod = _import_report()
        r = _make_result(
            name="model-diff",
            task_strategy="diffusion_media_generation",
        )
        r["vlm_assessment"] = {
            "model_id": "example-org/example-vl-judge",
            "vlm_judgment": {
                "semantic_similarity_0_to_5": 4,
                "trt_prompt_alignment_0_to_5": 4,
                "hf_prompt_alignment_0_to_5": 5,
                "trt_visual_quality_0_to_5": 4,
                "hf_visual_quality_0_to_5": 5,
                "is_regression": False,
                "vlm_gate": {
                    "failed": True,
                    "waived": True,
                    "waive_reason": "XFAIL allows reference-only VLM gate failure",
                    "reasons": ["HF reference description suggests non-photo/stylized output"],
                },
            },
        }
        html = mod.render_diffusion_model(r)
        assert "<strong>Gate:</strong> FAIL" in html
        assert "XFAIL allows reference-only VLM gate failure" not in html
        assert "HF reference description suggests non-photo/stylized output" in html

    def test_load_all_results_attaches_vlm_assessment(self, tmp_path):
        mod = _import_report()
        artifacts_dir = tmp_path / "artifacts"
        result = _make_result(
            name="model-diff",
            task_strategy="diffusion_media_generation",
        )
        _write_result(artifacts_dir, "model-diff", result)
        (tmp_path / "diffusion_vlm_assessment.json").write_text(
            json.dumps(
                {
                    "model_id": "example-org/example-vl-judge",
                    "results": [
                        {
                            "case_name": "model-diff",
                            "vlm_judgment": {
                                "semantic_similarity_0_to_5": 4.25,
                                "vlm_gate": {"failed": False, "reasons": []},
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        loaded = mod.load_all_results(artifacts_dir)
        assert loaded[0]["vlm_assessment"]["model_id"] == ("example-org/example-vl-judge")
        assert loaded[0]["vlm_assessment"]["vlm_judgment"]["semantic_similarity_0_to_5"] == 4.25


class TestRenderAudioModel:
    """Tests for render_audio_model()."""

    def test_embeds_wav(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "audio-model"
        model_dir.mkdir()
        trt_wav_path = model_dir / "trt_output.wav"
        ref_wav_path = model_dir / "ref_output.wav"
        _make_tiny_wav(trt_wav_path)
        _make_tiny_wav(ref_wav_path)
        r = _make_result(
            name="audio-model",
            task_strategy="text_to_audio",
            artifacts={"trt_wav": "trt_output.wav", "ref_wav": "ref_output.wav"},
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_audio_model(r)
        assert "<audio" in html
        assert "TRT / Base Audio" in html
        assert "Reference Audio" in html
        assert html.count("<audio") == 2
        assert html.count("data:audio/wav;base64,") == 2

    def test_missing_reference_audio_failure_is_visible(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "audio-reference-failure"
        model_dir.mkdir()
        trt_wav_path = model_dir / "trt_output.wav"
        _make_tiny_wav(trt_wav_path)
        r = _make_result(
            name="audio-reference-failure",
            task_strategy="text_to_audio",
            artifacts={"trt_wav": "trt_output.wav"},
            stage_outputs={
                "trt_full_generation": {
                    "stage_name": "full_generation",
                    "data": {},
                    "metadata": {},
                },
                "ref_full_generation": {
                    "stage_name": "full_generation",
                    "data": {
                        "returncode": 1,
                        "stderr_truncated": "missing offline cache",
                        "stderr_log": "reference_stderr.log",
                    },
                    "metadata": {"returncode": 1},
                },
            },
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_audio_model(r)
        assert "Reference Audio unavailable" in html
        assert "ref_full_generation" in html
        assert "missing offline cache" in html

    def test_speech_to_text_shows_transcript(self):
        mod = _import_report()
        r = _make_result(
            task_strategy="speech_to_text",
            trt_text="Hello from speech model",
            ref_text="Hello from speech model",
        )
        html = mod.render_audio_model(r)
        assert "Transcript Comparison" in html
        assert "Hello from speech model" in html

    def test_speech_to_text_embeds_source_audio(self, tmp_path):
        mod = _import_report()
        source = tmp_path / "source.wav"
        _make_tiny_wav(source)
        result = _make_result(
            task_strategy="speech_to_text",
            trt_text="hello",
            ref_text="hello",
        )
        result["case_config"]["inputs"]["audio"] = "source.wav"

        rendered = mod.render_audio_model(result, project_dir=tmp_path)

        assert "Input / Source Audio" in rendered
        assert rendered.count("data:audio/wav;base64,") == 1

    def test_speech_to_speech_embeds_input_and_both_outputs(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "speech-to-speech"
        model_dir.mkdir()
        for filename in ("input.wav", "trt.wav", "ref.wav"):
            target = tmp_path / filename if filename == "input.wav" else model_dir / filename
            _make_tiny_wav(target)
        result = _make_result(
            name="speech-to-speech",
            task_strategy="speech_to_speech",
            artifacts={"trt_wav": "trt.wav", "ref_wav": "ref.wav"},
        )
        result["case_config"]["inputs"]["audio"] = "input.wav"
        result["_artifact_dir"] = str(model_dir)

        rendered = mod.render_audio_model(result, project_dir=tmp_path)

        assert "Input / Source Audio" in rendered
        assert "TRT / Base Audio" in rendered
        assert "Reference Audio" in rendered
        assert rendered.count("data:audio/wav;base64,") == 3
        assert mod.validate_evidence([result], project_dir=tmp_path) == []

    def test_text_to_audio_shows_prompt_and_audio_metadata(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "tts"
        model_dir.mkdir()
        for filename in ("trt.wav", "ref.wav"):
            _make_tiny_wav(model_dir / filename)
        result = _make_result(
            name="tts",
            task_strategy="text_to_audio",
            prompt="Read this sentence",
            artifacts={"trt_wav": "trt.wav", "ref_wav": "ref.wav"},
            stage_outputs={
                "trt_generate": {
                    "data": {
                        "wav_path": str(model_dir / "trt.wav"),
                        "duration_s": 1.25,
                        "sample_rate": 24000,
                        "rms": 0.1,
                    },
                },
                "ref_generate": {
                    "data": {
                        "wav_path": str(model_dir / "ref.wav"),
                        "duration_s": 1.3,
                        "sample_rate": 24000,
                        "rms": 0.09,
                    },
                },
            },
        )
        result["_artifact_dir"] = str(model_dir)

        rendered = mod.render_audio_model(result)

        assert "Read this sentence" in rendered
        assert "Audio Metadata" in rendered
        assert "Duration (seconds)" in rendered
        assert "Sample rate" in rendered
        assert "RMS" in rendered


class TestRenderSegmentationModel:
    """Tests for render_segmentation_model()."""

    def test_embeds_seg_map(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "segmentation-model"
        model_dir.mkdir()
        seg_png = model_dir / "trt_seg.png"
        _make_tiny_png(seg_png)
        r = _make_result(
            name="segmentation-model",
            task_strategy="segmentation",
            artifacts={"trt_segmentation_map": "trt_seg.png"},
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_segmentation_model(r, project_dir=None)
        assert "TRT / Base Segmentation" in html
        assert "data:image/png;base64," in html

    def test_embeds_prompted_segmentation_overlay(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "prompted-segmentation-model"
        model_dir.mkdir()
        trt_overlay = model_dir / "masks" / "segmented.png"
        ref_overlay = model_dir / "ref_segmented.png"
        trt_overlay.parent.mkdir()
        _make_tiny_png(trt_overlay)
        _make_tiny_png(ref_overlay)
        r = _make_result(
            name="prompted-segmentation-model",
            task_strategy="prompted_segmentation",
            artifacts={
                "trt_segmented_image": "masks/segmented.png",
                "ref_segmented_image": "ref_segmented.png",
            },
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_segmentation_model(r, project_dir=None)
        assert "TRT / Base Segmentation" in html
        assert "Reference Segmentation" in html
        assert html.count("data:image/png;base64,") == 2

    def test_prompted_segmentation_shows_text_prompt(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "prompted-text-segmentation-model"
        model_dir.mkdir()
        r = _make_result(
            name="prompted-text-segmentation-model",
            family="prompted_text_segmentation_family",
            task_strategy="prompted_segmentation",
            prompt="car",
        )
        r["_artifact_dir"] = str(model_dir)

        html = mod.render_segmentation_model(r, project_dir=None)

        assert "<strong>Prompt:</strong> car" in html

    def test_prompted_segmentation_shows_point_and_all_visuals(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "prompted-all-visuals"
        model_dir.mkdir()
        _make_tiny_png(tmp_path / "input.png")
        artifacts = {}
        for prefix in ("trt", "ref"):
            for kind in ("segmented_image", "segmentation_map"):
                filename = f"{prefix}_{kind}.png"
                _make_tiny_png(model_dir / filename)
                artifacts[f"{prefix}_{kind}"] = filename
        result = _make_result(
            name="prompted-all-visuals",
            task_strategy="prompted_segmentation",
            artifacts=artifacts,
        )
        result["case_config"]["inputs"].update({
            "image": "input.png",
            "point_x": 12,
            "point_y": 34,
            "is_foreground": True,
        })
        result["_artifact_dir"] = str(model_dir)

        rendered = mod.render_segmentation_model(result, project_dir=tmp_path)

        assert "Point prompt" in rendered
        assert "x=12" in rendered
        assert "y=34" in rendered
        assert rendered.count("data:image/png;base64,") == 5
        assert mod.validate_evidence([result], project_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# Tests: select_frames
# ---------------------------------------------------------------------------


class TestSelectFrames:
    """Tests for _select_frames()."""

    def test_all_returned_if_under_limit(self, tmp_path):
        mod = _import_report()
        paths = [tmp_path / f"f{i}.png" for i in range(3)]
        assert mod._select_frames(paths, 6) == paths

    def test_evenly_spaced(self, tmp_path):
        mod = _import_report()
        paths = [tmp_path / f"f{i}.png" for i in range(17)]
        selected = mod._select_frames(paths, 6)
        assert len(selected) == 6
        # First and last should always be included
        assert selected[0] == paths[0]
        assert selected[-1] == paths[-1]


class TestEvidenceCompleteness:
    def test_passing_audio_requires_both_model_and_reference_files(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "audio"
        model_dir.mkdir()
        _make_tiny_wav(model_dir / "trt.wav")
        result = _make_result(
            name="audio",
            task_strategy="text_to_audio",
            artifacts={"trt_wav": "trt.wav"},
        )
        result["_artifact_dir"] = str(model_dir)

        issues = mod.validate_evidence([result], project_dir=tmp_path)

        assert any("reference audio" in issue for issue in issues)

    def test_omni_audio_contract_requires_both_playable_outputs(self):
        mod = _import_report()
        result = _make_result(
            name="omni-no-audio",
            task_strategy="omni_multimodal",
            stage_outputs={
                "trt_talker_decode": {"data": {"token_ids": [1]}},
                "ref_talker_decode": {"data": {"_invariant_only": True}},
            },
        )
        result["oracle_level"] = "L4_invariants"
        result["case_config"]["reference_backend"] = "torch_reference"
        result["case_config"]["user_contract"] = "tts_audio"

        issues = mod.validate_evidence([result], project_dir=None)

        assert any("TRT/base audio" in issue for issue in issues)
        assert any("reference audio" in issue for issue in issues)

    def test_empty_embedding_is_not_complete_numeric_evidence(self):
        mod = _import_report()
        result = _make_result(
            name="empty-embedding",
            task_strategy="embedding",
            stage_outputs={
                "trt_full_inference": {"data": {"embedding": []}},
                "ref_full_inference": {"data": {"embedding": []}},
            },
        )

        issues = mod.validate_evidence([result], project_dir=None)

        assert any("TRT/base numeric feature" in issue for issue in issues)
        assert any("reference numeric feature" in issue for issue in issues)

    def test_failed_audio_can_render_partial_failure_evidence(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "failed-audio"
        model_dir.mkdir()
        _make_tiny_wav(model_dir / "trt.wav")
        result = _make_result(
            name="failed-audio",
            status="fail",
            task_strategy="text_to_audio",
            artifacts={"trt_wav": "trt.wav"},
        )
        result["_artifact_dir"] = str(model_dir)

        assert mod.validate_evidence([result], project_dir=tmp_path) == []
        assert "Required audio file is unavailable" in mod.render_audio_model(result)
        report = mod.render_report([result], evidence_issues=[])
        assert "E2E status is fail" in report
        assert "evidence may be partial" in report

    def test_personaplex_requires_playable_reference_audio(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "personaplex"
        model_dir.mkdir()
        _make_tiny_wav(tmp_path / "input.wav")
        _make_tiny_wav(model_dir / "trt.wav")
        result = _make_result(
            name="personaplex",
            task_strategy="speech_to_speech",
            artifacts={"trt_wav": "trt.wav"},
            stage_outputs={
                "trt_full_generation": {"data": {"wav_path": str(model_dir / "trt.wav")}},
                "ref_full_generation": {
                    "data": {
                        "reference_tokens": [[1, 2], [3, 4]],
                        "num_frames": 2,
                        "token_shape": [2, 2],
                    }
                },
            },
        )
        result["oracle_level"] = "L2_internal_reference"
        result["case_config"]["reference_backend"] = "torch_reference"
        result["case_config"]["inputs"]["audio"] = "input.wav"
        result["_artifact_dir"] = str(model_dir)

        assert any(
            "reference audio" in issue
            for issue in mod.validate_evidence([result], project_dir=tmp_path)
        )
        rendered = mod.render_audio_model(result, project_dir=tmp_path)
        assert "Configured Reference Evidence" in rendered
        assert "reference_tokens" in rendered

    def test_invariant_only_omni_still_requires_human_reference_audio(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "omni"
        model_dir.mkdir()
        _make_tiny_wav(model_dir / "talker.wav")
        result = _make_result(
            name="omni",
            task_strategy="omni_multimodal",
            stage_outputs={
                "trt_talker_decode": {
                    "data": {"token_ids": [1]},
                    "metadata": {"audio_output_path": str(model_dir / "talker.wav")},
                },
                "ref_talker_decode": {
                    "data": {"_invariant_only": True},
                    "metadata": {"source": "invariant_only"},
                },
            },
        )
        result["oracle_level"] = "L4_invariants"
        result["case_config"]["reference_backend"] = "invariant_only"
        result["_artifact_dir"] = str(model_dir)

        assert any(
            "reference audio" in issue
            for issue in mod.validate_evidence([result], project_dir=tmp_path)
        )
        rendered = mod.render_model_section(result, project_dir=tmp_path)
        assert "No external reference output is configured" in rendered
        assert "TRT / Base Audio" in rendered
        assert "Reference Audio" in rendered

    def test_invariant_omni_embeds_hf_audio_for_human_review(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "omni-with-reference"
        model_dir.mkdir()
        for filename in ("talker.wav", "hf_reference.wav"):
            _make_tiny_wav(model_dir / filename)
        result = _make_result(
            name="omni-with-reference",
            task_strategy="omni_multimodal",
            stage_outputs={
                "trt_talker_decode": {
                    "data": {"token_ids": [1]},
                    "metadata": {
                        "audio_output_path": str(model_dir / "talker.wav")
                    },
                },
                "ref_talker_decode": {
                    "data": {"_invariant_only": True},
                    "metadata": {
                        "source": "hf_human_reference",
                        "audio_output_path": str(model_dir / "hf_reference.wav"),
                    },
                },
            },
        )
        result["oracle_level"] = "L4_invariants"
        result["case_config"]["reference_backend"] = "torch_reference"
        result["_artifact_dir"] = str(model_dir)

        assert mod.validate_evidence([result], project_dir=tmp_path) == []
        rendered = mod.render_model_section(result, project_dir=tmp_path)
        assert rendered.count("data:audio/wav;base64,") == 2
        assert "human-review evidence" in rendered
        assert "waveform-equality gate" in rendered

    def test_empty_but_present_invariant_text_is_not_treated_as_missing(self):
        mod = _import_report()
        result = _make_result(
            name="elf",
            task_strategy="diffusion_text_generation",
            stage_outputs={
                "trt_decoded_text": {"text": "", "data": {"generated_samples": []}},
                "ref_decoded_text": {"text": None, "data": {"_invariant_only": True}},
            },
        )
        result["oracle_level"] = "L4_invariants"
        result["case_config"]["reference_backend"] = "invariant_only"

        assert mod.validate_evidence([result], project_dir=None) == []

    def test_artifact_path_cannot_escape_case_directory(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "audio"
        model_dir.mkdir()
        _make_tiny_wav(model_dir / "trt.wav")
        _make_tiny_wav(tmp_path / "outside.wav")
        result = _make_result(
            name="audio",
            task_strategy="text_to_audio",
            artifacts={"trt_wav": "trt.wav", "ref_wav": "../outside.wav"},
        )
        result["_artifact_dir"] = str(model_dir)

        rendered = mod.render_audio_model(result)
        issues = mod.validate_evidence([result], project_dir=tmp_path)

        assert rendered.count("data:audio/wav;base64,") == 1
        assert any("reference audio" in issue for issue in issues)

    def test_artifact_symlink_cannot_escape_case_directory(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "audio"
        model_dir.mkdir()
        _make_tiny_wav(model_dir / "trt.wav")
        _make_tiny_wav(tmp_path / "outside.wav")
        (model_dir / "ref.wav").symlink_to(tmp_path / "outside.wav")
        result = _make_result(
            name="audio",
            task_strategy="text_to_audio",
            artifacts={"trt_wav": "trt.wav", "ref_wav": "ref.wav"},
        )
        result["_artifact_dir"] = str(model_dir)

        assert any(
            "reference audio" in issue
            for issue in mod.validate_evidence([result], project_dir=tmp_path)
        )

    def test_corrupt_media_cannot_satisfy_strict_evidence(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "corrupt-image"
        model_dir.mkdir()
        (tmp_path / "input.png").write_bytes(b"not a PNG")
        for filename in ("trt.png", "ref.png"):
            _make_tiny_png(model_dir / filename)
        result = _make_result(
            name="corrupt-image",
            task_strategy="segmentation",
            artifacts={
                "trt_segmentation_map": "trt.png",
                "ref_segmentation_map": "ref.png",
            },
        )
        result["case_config"]["inputs"]["image"] = "input.png"
        result["_artifact_dir"] = str(model_dir)

        assert any(
            "input image" in issue
            for issue in mod.validate_evidence([result], project_dir=tmp_path)
        )
        rendered = mod.render_segmentation_model(result, project_dir=tmp_path)
        assert "Image could not be embedded" in rendered

    def test_frame_directory_child_symlink_cannot_escape_case_directory(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "diffusion"
        frames_dir = model_dir / "frames"
        frames_dir.mkdir(parents=True)
        outside = tmp_path / "outside.png"
        _make_tiny_png(outside)
        (frames_dir / "frame_000.png").symlink_to(outside)

        resolved = mod._resolve_frame_paths("frames", model_dir, "frames")

        assert resolved == []

    def test_strict_cli_writes_report_then_fails_for_missing_reference(self, tmp_path):
        mod = _import_report()
        artifacts = tmp_path / "artifacts"
        model_dir = artifacts / "audio"
        model_dir.mkdir(parents=True)
        _make_tiny_wav(model_dir / "trt.wav")
        _write_result(
            artifacts,
            "audio",
            _make_result(
                name="audio",
                task_strategy="text_to_audio",
                artifacts={"trt_wav": "trt.wav"},
            ),
        )
        output = tmp_path / "report.html"

        rc = mod.main([
            "--artifacts-dir", str(artifacts),
            "--output", str(output),
            "--project-dir", str(tmp_path),
            "--strict-evidence",
            "--max-embed-bytes", "33554432",
        ])

        assert rc == 2
        assert output.is_file()
        assert "The report is incomplete" in output.read_text(encoding="utf-8")

    def test_proof_context_renders_revision_steps_and_selected_tests(self):
        mod = _import_report()
        context = {
            "model": "alpha",
            "source_revision": "a" * 40,
            "suite": "premerge",
            "outcome": "passed",
            "runtime_library": "libtrtmc_model_alpha.so",
            "runtime_library_sha256": "b" * 64,
            "staged_runtime_library_sha256": "b" * 64,
            "sibling_model_count": 0,
            "model_dso_count": 1,
            "staged_model_dso_count": 1,
            "engine_builds_per_model": 1,
            "engine_build_count": 1,
            "engine_build_verification": "engine-build-verification.json",
            "gpu_id": "2",
            "gpu_resource_class": "shared",
            "gpu_slot_ids": [1],
            "gpu_slots_per_device": 4,
            "gpu_lease_evidence": "gpu-lease.json",
            "network": "disabled",
            "plugin_search": "strict",
            "e2e_proof_kinds": ["functional_invariant"],
            "e2e_reference_passed": False,
            "steps": {"scratch_build": {"status": "passed", "evidence": "build.log"}},
            "selection": {
                "runtime_tests": ["test_alpha"],
                "python_tests": ["tests/e2e/models/alpha/test_alpha_unit.py"],
                "e2e_cases": [{"name": "alpha-small"}],
                "e2e_test": "tests/e2e/models/alpha/test_alpha_e2e.py",
            },
        }

        rendered = mod.render_report([], proof_context=context, evidence_issues=[])

        assert "Isolation Proof" in rendered
        assert "a" * 40 in rendered
        assert "libtrtmc_model_alpha.so" in rendered
        assert "Host GPU ID" in rendered
        assert "Full bundle builds per model" in rendered
        assert "Staged runtime library SHA-256" in rendered
        assert "Model DSOs staged" in rendered
        assert "E2E proof kind" in rendered
        assert "functional_invariant" in rendered
        assert ">2<" in rendered
        assert "test_alpha" in rendered
        assert "test_alpha_unit.py" in rendered
        assert "alpha-small" in rendered
        assert "All required user-facing evidence is embedded" in rendered

    def test_successful_proof_context_requires_complete_isolation_metadata(self):
        mod = _import_report()
        steps = {
            name: {"status": "passed"}
            for name in (
                "projection_validation",
                "configure",
                "scratch_build",
                "dso_isolation",
                "cpp_tests",
                "e2e_reference",
                "engine_build_budget",
                "result_verification",
            )
        }
        steps["python_tests"] = {"status": "skipped"}
        steps["e2e_snapshot_regression"] = {"status": "skipped"}
        steps["e2e_functional_invariant"] = {"status": "skipped"}
        steps["html_report"] = {"status": "running"}
        status = {
            "model": "alpha",
            "source_revision": "a" * 40,
            "suite": "premerge",
            "gpu_id": "2",
            "gpu_resource_class": "shared",
            "gpu_slot_ids": [1],
            "gpu_slots_per_device": 4,
            "gpu_lease_evidence": "gpu-lease.json",
            "validation_exit_code": 0,
            "e2e_proof_kinds": ["reference"],
            "e2e_reference_passed": True,
            "steps": steps,
        }
        proof = {
            "passed": True,
            "model": "alpha",
            "source_revision": "a" * 40,
            "runtime_model": "alpha",
            "runtime_library": "libtrtmc_model_alpha.so",
            "runtime_library_sha256": "b" * 64,
            "staged_runtime_library_sha256": "b" * 64,
            "sibling_model_count": 0,
            "model_dso_count": 1,
            "staged_model_dso_count": 1,
            "engine_builds_per_model": 1,
            "engine_build_count": 1,
            "engine_build_verification": "engine-build-verification.json",
            "gpu_id": "2",
            "gpu_resource_class": "shared",
            "gpu_slot_ids": [1],
            "gpu_slots_per_device": 4,
            "gpu_lease_evidence": "gpu-lease.json",
            "network": "disabled",
            "plugin_search": "strict",
            "e2e_proof_kinds": ["reference"],
            "e2e_reference_passed": True,
        }
        selection = {
            "requested_model": "alpha",
            "gpu_id": "2",
            "e2e_test": "tests/e2e/models/alpha/test_alpha_e2e.py",
            "e2e_cases": [{"name": "alpha-small"}],
            "gpu_resource_class": "shared",
            "gpu_slot_ids": [1],
            "gpu_slots_per_device": 4,
            "gpu_lease_evidence": "gpu-lease.json",
        }

        assert mod.validate_proof_context(status, proof, selection) == []
        proof["runtime_library_sha256"] = "invalid"
        assert "SHA-256" in " ".join(mod.validate_proof_context(status, proof, selection))
        proof["runtime_library_sha256"] = "b" * 64
        proof["staged_runtime_library_sha256"] = "c" * 64
        assert "does not match" in " ".join(mod.validate_proof_context(status, proof, selection))
        proof["staged_runtime_library_sha256"] = "b" * 64
        proof["staged_model_dso_count"] = 2
        assert "staged model DSO" in " ".join(mod.validate_proof_context(status, proof, selection))

        proof["staged_model_dso_count"] = 1
        proof["e2e_proof_kinds"] = ["functional_invariant"]
        proof["e2e_reference_passed"] = False
        status["e2e_proof_kinds"] = ["functional_invariant"]
        status["e2e_reference_passed"] = False
        status["steps"]["e2e_reference"] = {"status": "skipped"}
        status["steps"]["e2e_functional_invariant"] = {"status": "passed"}
        assert mod.validate_proof_context(status, proof, selection) == []

        status["steps"]["e2e_reference"] = {"status": "passed"}
        issues = mod.validate_proof_context(status, proof, selection)
        assert any("e2e_reference must be skipped" in issue for issue in issues)

    def test_proof_diagnostics_embed_bounded_log_and_junit_failure(self, tmp_path):
        mod = _import_report()
        status = tmp_path / "model-proof-status.json"
        status.write_text("{}", encoding="utf-8")
        (tmp_path / "build.log").write_text("compile failed\n", encoding="utf-8")
        (tmp_path / "python-model-tests.xml").write_text(
            '<testsuite><testcase name="test_model">'
            '<failure message="assertion failed" /></testcase></testsuite>',
            encoding="utf-8",
        )

        diagnostics = mod._load_proof_diagnostics(status)
        rendered = mod.render_proof_section({"diagnostics": diagnostics})

        assert "compile failed" in rendered
        assert "test_model" in rendered
        assert "assertion failed" in rendered


# ---------------------------------------------------------------------------
# Tests: key_metric and total_time
# ---------------------------------------------------------------------------


class TestDashboardHelpers:
    """Tests for _key_metric and _total_time."""

    def test_key_metric_extracts_cosine(self):
        mod = _import_report()
        r = _make_result()
        km = mod._key_metric(r)
        assert "logit_cosine_p5" in km
        assert "0.998" in km

    def test_key_metric_empty_stages(self):
        mod = _import_report()
        r = _make_result()
        r["stages"] = {}
        assert mod._key_metric(r) == ""

    def test_total_time(self):
        mod = _import_report()
        r = _make_result(timing={"a": 1.0, "b": 2.5})
        assert mod._total_time(r) == "3.5s"

    def test_total_time_empty(self):
        mod = _import_report()
        r = _make_result(timing={})
        assert mod._total_time(r) == ""


# ---------------------------------------------------------------------------
# Tests: CLI (parse_args)
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_required_args(self):
        mod = _import_report()
        args = mod.parse_args(
            [
                "--artifacts-dir",
                "/tmp/arts",
                "-o",
                "/tmp/report.html",
            ]
        )
        assert args.artifacts_dir == Path("/tmp/arts")
        assert args.output == Path("/tmp/report.html")
        assert args.title == "E2E Test Report"

    def test_all_args(self):
        mod = _import_report()
        args = mod.parse_args(
            [
                "--artifacts-dir",
                "/a",
                "-o",
                "/b.html",
                "--project-dir",
                "/proj",
                "--manifest-dir",
                "/manifests",
                "--title",
                "Custom Title",
            ]
        )
        assert args.project_dir == Path("/proj")
        assert args.manifest_dir == Path("/manifests")
        assert args.title == "Custom Title"

    def test_main_writes_output(self, tmp_path):
        mod = _import_report()
        arts = tmp_path / "artifacts"
        arts.mkdir()
        _write_result(arts, "m1", _make_result(name="m1"))
        out = tmp_path / "report.html"
        rc = mod.main(
            [
                "--artifacts-dir",
                str(arts),
                "-o",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "m1" in content
        assert "<!DOCTYPE html>" in content

    def test_main_empty_dir(self, tmp_path):
        mod = _import_report()
        arts = tmp_path / "empty"
        arts.mkdir()
        out = tmp_path / "report.html"
        rc = mod.main(
            [
                "--artifacts-dir",
                str(arts),
                "-o",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: render_summary_dashboard
# ---------------------------------------------------------------------------


class TestSummaryDashboard:
    """Tests for the summary dashboard."""

    def test_filter_controls_present(self):
        mod = _import_report()
        html = mod.render_summary_dashboard([_make_result()])
        assert 'id="search-box"' in html
        assert 'id="status-filter"' in html

    def test_anchor_links(self):
        mod = _import_report()
        r = _make_result(name="my-model")
        html = mod.render_summary_dashboard([r])
        assert 'href="#model-my-model"' in html

    def test_counters(self):
        mod = _import_report()
        results = [
            _make_result(name="a", status="pass"),
            _make_result(name="b", status="fail"),
            _make_result(name="c", status="skip"),
        ]
        html = mod.render_summary_dashboard(results)
        assert "1 Passed" in html
        assert "1 Failed" in html
        assert "1 Skipped" in html
        assert "3 Results" in html
        assert "3 Models" in html
        assert "3 Testcases" in html

    def test_summary_rows_sorted_by_total_time_descending(self):
        mod = _import_report()
        results = [
            _make_result(name="fast", timing={"build_s": 1.0}),
            _make_result(name="slow", timing={"build_s": 3.0, "trt_generate_s": 2.0}),
            _make_result(name="medium", timing={"build_s": 2.0}),
            _make_result(name="missing", timing={}),
        ]

        html = mod.render_summary_dashboard(results)

        assert html.index('href="#model-slow"') < html.index('href="#model-medium"')
        assert html.index('href="#model-medium"') < html.index('href="#model-fast"')
        assert html.index('href="#model-fast"') < html.index('href="#model-missing"')

    def test_model_with_three_testcases_renders_as_single_expandable_row(self):
        mod = _import_report()
        base = _make_result(name="canary-1b-v2", timing={"build_s": 5.0})
        probe1 = _make_result(
            name="canary-1b-v2-asr-probe01",
            timing={},
            model_name="canary-1b-v2",
        )
        probe2 = _make_result(
            name="canary-1b-v2-asr-probe02",
            timing={},
            model_name="canary-1b-v2",
        )
        outcome = {
            "pytest_status": "PASSED",
        }
        for result in (base, probe1, probe2):
            result["_pytest_outcome"] = outcome

        html = mod.render_summary_dashboard([base, probe1, probe2])

        assert html.count('class="summary-row"') == 1
        assert 'class="summary-model-details"' in html
        assert "3 testcases" in html
        assert 'href="#model-canary-1b-v2-asr-probe01"' in html
        assert 'href="#model-canary-1b-v2-asr-probe02"' in html
        assert 'data-name="canary-1b-v2 canary-1b-v2-asr-probe01 canary-1b-v2-asr-probe02"' in html

    def test_model_with_one_testcase_uses_same_expandable_structure(self):
        mod = _import_report()
        result = _make_result(name="single-model")

        html = mod.render_summary_dashboard([result])

        assert html.count('class="summary-row"') == 1
        assert 'class="summary-model-details"' in html
        assert "1 testcase" in html
        assert 'href="#model-single-model"' in html
        assert "1 Model" in html
        assert "1 Testcase" in html

    def test_declared_testcases_include_cases_without_results(self):
        mod = _import_report()
        base = _make_result(
            name="canary-1b-v2",
            task_strategy="speech_to_text",
            family="canary",
        )
        model_manifests = [
            {
                "name": "canary-1b-v2",
                "family": "canary",
                "bundle": "canary.trtfb",
                "testcases": [
                    {
                        "name": "canary-1b-v2",
                        "ci_tier": "default",
                        "task_strategy": "speech_to_text",
                    },
                    {
                        "name": "canary-1b-v2-asr-probe01",
                        "ci_tier": "nightly_only",
                        "task_strategy": "speech_to_text",
                    },
                    {
                        "name": "canary-1b-v2-asr-probe02",
                        "ci_tier": "nightly_only",
                        "task_strategy": "speech_to_text",
                    },
                ],
            }
        ]

        html = mod.render_summary_dashboard([base], model_manifests)

        assert html.count('class="summary-row"') == 1
        assert "1 Results" in html
        assert "1 Model" in html
        assert "3 testcases" in html
        assert "3 Testcases" in html
        assert html.count('class="summary-testcase-row manifest-only"') == 2
        assert "NOT RUN: 2" in html
        assert html.count("nightly_only") == 2
        assert 'href="#model-canary-1b-v2-asr-probe01"' not in html

    def test_all_nightly_model_is_visible_without_results(self):
        mod = _import_report()
        model_manifests = [
            {
                "name": "nemotron-labs-diffusion-8b",
                "family": "nemotron_labs_diffusion",
                "bundle": "nemotron-labs-diffusion-8b.trtfb",
                "testcases": [
                    {
                        "name": "nemotron-labs-diffusion-8b-ar",
                        "ci_tier": "nightly_only",
                        "task_strategy": "text_generation_causal",
                    },
                    {
                        "name": "nemotron-labs-diffusion-8b-diffusion",
                        "ci_tier": "nightly_only",
                        "task_strategy": "text_generation_causal",
                    },
                ],
            }
        ]

        html = mod.render_report([], model_manifests=model_manifests)

        assert "nemotron-labs-diffusion-8b" in html
        assert "2 testcases" in html
        assert html.count('class="summary-testcase-row manifest-only"') == 2
        assert "NOT RUN" in html
