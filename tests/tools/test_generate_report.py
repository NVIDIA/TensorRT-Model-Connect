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
    family: str = "qwen",
    hf_id: str = "Qwen/Qwen3-0.6B",
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
            "runtime_strategy": "decoder_kv_cache",
            "task_strategy": task_strategy,
            "reference_backend": "hf_transformers",
            "inputs": {"prompt": prompt},
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
        "stage_outputs": stage_outputs or {
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
    (model_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
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
    fmt_chunk = (
        b"fmt "
        + st.pack("<I", 16)
        + st.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    )
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    wav = b"RIFF" + st.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk
    path.write_bytes(wav)


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

    def test_generic_strategies(self):
        mod = _import_report()
        for ts in ("encoder_only_nlp", "embedding", "reranking"):
            r = _make_result(task_strategy=ts)
            assert mod.classify_modality(r) == "generic", f"Failed for {ts}"

    def test_unknown_strategy_defaults_generic(self):
        mod = _import_report()
        r = _make_result(task_strategy="some_future_strategy")
        assert mod.classify_modality(r) == "generic"

    def test_missing_case_config(self):
        mod = _import_report()
        r = {"case_name": "x"}
        assert mod.classify_modality(r) == "generic"


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
            <testcase classname="tests.test_e2e" name="test_e2e[fnet-base]">
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
        assert (
            "<script>function externalAssetTest() { return true; }\n</script>" in html
        )

    def test_empty_results(self):
        mod = _import_report()
        html = mod.render_report([], title="Empty Report")
        assert "<!DOCTYPE html>" in html
        assert "Empty Report" in html
        assert "0 Total" in html

    def test_single_text_model(self):
        mod = _import_report()
        r = _make_result(name="qwen3-0.6b", prompt="Test prompt")
        html = mod.render_report([r], title="Test Report")
        assert "qwen3-0.6b" in html
        assert "Test prompt" in html
        assert "PASS" in html
        assert "logit_cosine_p5" in html
        assert "1 Passed" in html

    def test_failed_model_shows_failure_type(self):
        mod = _import_report()
        r = _make_result(
            name="bad-model", status="fail", failure_type="compare_fail"
        )
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
                "weights_loading_qwen3_encoder_s": 8.0,
                "weights_loading_z_image_dit_s": 5.0,
                "trt_compile_s": 88.0,
                "trt_compile_qwen3_encoder_s": 30.0,
                "trt_compile_z_image_dit_s": 50.0,
            },
        )
        html = mod.render_report([r])
        assert "<summary>Weights loading</summary>" in html
        assert "qwen3 encoder" in html
        assert "8.00s" in html
        assert "z image dit" in html
        assert "5.00s" in html
        assert "<summary>TRT compile</summary>" in html
        assert "qwen3 encoder" in html
        assert "30.00s" in html
        assert "z image dit" in html
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
                        "stderr": "\n".join([
                            '[trtmc.load_timing] label="denoiser_plan" '
                            "load_deserialize_ms=2500.000000 plan_bytes=1",
                            '[trtmc.engine_timing] label="denoiser_plan" '
                            "execute_ms=600.000000 launches=20",
                        ]),
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
        log_text = "\n".join([
            '[trtmc.load_timing] label="denoiser_plan" '
            "load_deserialize_ms=2500.000000 plan_bytes=1024",
            '[trtmc.engine_timing] label="denoiser_plan" '
            "execute_ms=600.000000 launches=20",
        ])
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
            "\n".join([
                "[trtmc build] Weights loaded [999.0s]",
                "[trtmc build] Engine built [888.0s] (10.0 MB)",
            ]),
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
            family="albert",
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
        html = mod.render_diffusion_vlm_assessment({
            "vlm_assessment": {
                "model_id": "Judge <model>",
                "vlm_judgment": "not a structured judgment",
            }
        })
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

    def test_vlm_assessment_is_rendered(self):
        mod = _import_report()
        r = _make_result(
            name="model-diff",
            task_strategy="diffusion_media_generation",
        )
        r["vlm_assessment"] = {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
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
        assert "Qwen/Qwen2.5-VL-3B-Instruct" in html
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
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
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
                    "reasons": [
                        "HF reference description suggests non-photo/stylized output"
                    ],
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
            json.dumps({
                "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                "results": [{
                    "case_name": "model-diff",
                    "vlm_judgment": {
                        "semantic_similarity_0_to_5": 4.25,
                        "vlm_gate": {"failed": False, "reasons": []},
                    },
                }],
            }),
            encoding="utf-8",
        )
        loaded = mod.load_all_results(artifacts_dir)
        assert loaded[0]["vlm_assessment"]["model_id"] == (
            "Qwen/Qwen2.5-VL-3B-Instruct")
        assert loaded[0]["vlm_assessment"]["vlm_judgment"][
            "semantic_similarity_0_to_5"] == 4.25


class TestRenderAudioModel:
    """Tests for render_audio_model()."""

    def test_embeds_wav(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "bark"
        model_dir.mkdir()
        trt_wav_path = model_dir / "trt_output.wav"
        ref_wav_path = model_dir / "ref_output.wav"
        _make_tiny_wav(trt_wav_path)
        _make_tiny_wav(ref_wav_path)
        r = _make_result(
            name="bark",
            task_strategy="text_to_audio",
            artifacts={"trt_wav": "trt_output.wav", "ref_wav": "ref_output.wav"},
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_audio_model(r)
        assert "<audio" in html
        assert "TRT Audio" in html
        assert "Reference Audio" in html
        assert "data:audio/wav;base64," in html

    def test_missing_reference_audio_failure_is_visible(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "magpie"
        model_dir.mkdir()
        trt_wav_path = model_dir / "trt_output.wav"
        _make_tiny_wav(trt_wav_path)
        r = _make_result(
            name="magpie",
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
                        "stderr_log": "nemo_magpie_ref_stderr.log",
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
            trt_text="Hello from Whisper",
            ref_text="Hello from Whisper",
        )
        html = mod.render_audio_model(r)
        assert "Transcript Comparison" in html
        assert "Hello from Whisper" in html


class TestRenderSegmentationModel:
    """Tests for render_segmentation_model()."""

    def test_embeds_seg_map(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "segformer"
        model_dir.mkdir()
        seg_png = model_dir / "trt_seg.png"
        _make_tiny_png(seg_png)
        r = _make_result(
            name="segformer",
            task_strategy="segmentation",
            artifacts={"trt_segmentation_map": "trt_seg.png"},
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_segmentation_model(r, project_dir=None)
        assert "TRT Segmentation Map" in html
        assert "data:image/png;base64," in html

    def test_embeds_prompted_segmentation_overlay(self, tmp_path):
        mod = _import_report()
        model_dir = tmp_path / "sam"
        model_dir.mkdir()
        trt_overlay = model_dir / "masks" / "segmented.png"
        ref_overlay = model_dir / "hf_sam_segmented.png"
        trt_overlay.parent.mkdir()
        _make_tiny_png(trt_overlay)
        _make_tiny_png(ref_overlay)
        r = _make_result(
            name="sam",
            task_strategy="prompted_segmentation",
            artifacts={
                "trt_segmented_image": "masks/segmented.png",
                "ref_segmented_image": "hf_sam_segmented.png",
            },
        )
        r["_artifact_dir"] = str(model_dir)
        html = mod.render_segmentation_model(r, project_dir=None)
        assert "TRT Segmented Image" in html
        assert "Reference Segmented Image" in html
        assert html.count("data:image/png;base64,") == 2


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
        args = mod.parse_args([
            "--artifacts-dir", "/tmp/arts",
            "-o", "/tmp/report.html",
        ])
        assert args.artifacts_dir == Path("/tmp/arts")
        assert args.output == Path("/tmp/report.html")
        assert args.title == "E2E Test Report"

    def test_all_args(self):
        mod = _import_report()
        args = mod.parse_args([
            "--artifacts-dir", "/a",
            "-o", "/b.html",
            "--project-dir", "/proj",
            "--title", "Custom Title",
        ])
        assert args.project_dir == Path("/proj")
        assert args.title == "Custom Title"

    def test_main_writes_output(self, tmp_path):
        mod = _import_report()
        arts = tmp_path / "artifacts"
        arts.mkdir()
        _write_result(arts, "m1", _make_result(name="m1"))
        out = tmp_path / "report.html"
        rc = mod.main([
            "--artifacts-dir", str(arts),
            "-o", str(out),
        ])
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
        rc = mod.main([
            "--artifacts-dir", str(arts),
            "-o", str(out),
        ])
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
        assert "3 Total" in html

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

    def test_invariant_only_pass_is_rendered_as_weak_pass(self):
        mod = _import_report()
        result = _make_result(name="elf-b-owt-l0", status="pass")
        result["oracle_level"] = "L4_invariants"
        result["case_config"]["reference_backend"] = "invariant_only"

        html = mod.render_report([result])

        assert "0 Passed" in html
        assert "1 Weak Pass" in html
        assert "WEAK_PASS" in html
        assert 'data-status="weak_pass"' in html
        assert '<option value="weak_pass">Weak Pass</option>' in html
        assert "Weak validation: <strong>oracle_level is L4_invariants</strong>" in html

    def test_explicit_weak_validation_reason_is_rendered(self):
        mod = _import_report()
        result = _make_result(name="weak-text", status="pass")
        result["weak_validation_reason"] = "semantic verifier unavailable"

        html = mod.render_report([result])

        assert "WEAK_PASS" in html
        assert "semantic verifier unavailable" in html
