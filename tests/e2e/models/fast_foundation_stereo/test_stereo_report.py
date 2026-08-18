# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import re
import struct
import sys

import numpy as np

from tests.e2e.models.fast_foundation_stereo.e2e_plugins import report

_REPO_ROOT = Path(__file__).resolve().parents[4]

_INPUT_PNG = base64.b64decode(
    report._png_uri(bytes(700 * 700 * 3)).split(",", 1)[1],
    validate=True,
)


def _generator_module():
    path = _REPO_ROOT / "scripts" / "generate_e2e_report.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("stereo_report_generator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_dimensions(markup: str) -> list[tuple[int, int]]:
    dimensions = []
    for encoded in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", markup):
        payload = base64.b64decode(encoded, validate=True)
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert payload[12:16] == b"IHDR"
        dimensions.append(struct.unpack(">II", payload[16:24]))
    return dimensions


def _result(artifact_dir: Path, *, status: str = "pass") -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "left.png").write_bytes(_INPUT_PNG)
    (artifact_dir / "right.png").write_bytes(_INPUT_PNG)
    reference = np.full((700, 700), 12.0, dtype=np.float32)
    actual = reference + np.float32(0.25)
    actual.tofile(artifact_dir / "disparity.f32")
    np.save(artifact_dir / "torch_disparity.npy", reference, allow_pickle=False)
    return {
        "_artifact_dir": str(artifact_dir),
        "status": status,
        "case_config": {"family": "fast_foundation_stereo"},
        "stages": {
            "full_inference": {
                "metrics": {
                    "global_cosine": {
                        "value": 0.9999,
                        "threshold": 0.999,
                        "operator": ">=",
                        "passed": True,
                    },
                    "mean_abs_error": {
                        "value": 0.25,
                        "threshold": 0.5,
                        "operator": "<=",
                        "passed": True,
                    },
                    "bad_2px_fraction": {
                        "value": 0.0,
                        "threshold": 0.02,
                        "operator": "<=",
                        "passed": True,
                    },
                }
            }
        },
    }


def test_stereo_report_embeds_diffusion_style_visual_comparison(tmp_path: Path) -> None:
    result = _result(tmp_path / "fast-foundation-stereo")
    generator = _generator_module()

    rendered = generator.render_model_section(result, _REPO_ROOT)

    assert rendered is not None
    assert _png_dimensions(rendered) == [(700, 700)] * 5
    assert "Fast Foundation Stereo visual review" in rendered
    assert "Left input" in rendered
    assert "Right input" in rendered
    assert "Reference disparity" in rendered
    assert "TensorRT disparity" in rendered
    assert "Absolute error" in rendered
    assert rendered.count("12.000 px (reference p99.5)") == 2
    assert "2 px, and values above it count as bad pixels" in rendered
    assert "PASS" in rendered
    assert "cosine 0.9999 &gt;= 0.999" in rendered
    assert rendered.index("Fast Foundation Stereo visual review") < rendered.index(
        "Numerical evidence"
    )
    assert "unavailable" not in rendered


def test_stereo_report_is_deterministic_and_marks_failed_metrics(tmp_path: Path) -> None:
    result = _result(tmp_path / "case", status="fail")
    result["stages"]["full_inference"]["metrics"]["bad_2px_fraction"].update(
        value=0.5,
        passed=False,
    )

    first = report.render(result, project_dir=_REPO_ROOT)
    second = report.render(result, project_dir=_REPO_ROOT)

    assert first == second
    assert "FAIL" in first
    assert 'class="fail"' in first
    assert "bad-2px 0.5 &lt;= 0.02" in first


def test_stereo_report_fails_closed_for_missing_or_malformed_artifacts(
    tmp_path: Path,
) -> None:
    missing = report.render(
        {"_artifact_dir": str(tmp_path / "missing"), "status": "pass"},
        project_dir=_REPO_ROOT,
    )
    assert "Stereo visualization unavailable" in missing

    result = _result(tmp_path / "malformed")
    np.save(
        Path(result["_artifact_dir"]) / "torch_disparity.npy",
        np.zeros((2, 2), dtype=np.float32),
        allow_pickle=False,
    )
    malformed = report.render(result, project_dir=_REPO_ROOT)
    assert "Stereo visualization unavailable" in malformed
    assert "invalid NumPy file size" in malformed


def test_stereo_report_rejects_symlink_escape_and_invalid_input_png(tmp_path: Path) -> None:
    result = _result(tmp_path / "case")
    artifact_dir = Path(result["_artifact_dir"])
    outside = tmp_path / "outside.png"
    outside.write_bytes(_INPUT_PNG)
    (artifact_dir / "left.png").unlink()
    (artifact_dir / "left.png").symlink_to(outside)

    escaped = report.render(result, project_dir=_REPO_ROOT)

    assert "escapes the artifact directory" in escaped
    assert "data:image/png;base64," not in escaped

    (artifact_dir / "left.png").unlink()
    (artifact_dir / "left.png").write_bytes(b"not a png")
    invalid = report.render(result, project_dir=_REPO_ROOT)
    assert "must be a 700x700 PNG" in invalid


def test_stereo_report_rejects_fortran_order_reference(tmp_path: Path) -> None:
    result = _result(tmp_path / "case")
    np.save(
        Path(result["_artifact_dir"]) / "torch_disparity.npy",
        np.asfortranarray(np.zeros((700, 700), dtype=np.float32)),
        allow_pickle=False,
    )

    rendered = report.render(result, project_dir=_REPO_ROOT)

    assert "C-contiguous 700x700" in rendered
