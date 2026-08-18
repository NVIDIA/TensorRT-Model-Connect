# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned tests for DINOv3 dense-feature HTML evidence."""

from __future__ import annotations

import base64
import importlib.util
import re
import struct
import sys
import zlib
from pathlib import Path

import pytest

from tests.e2e.models.dinov3.e2e_plugins import report

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _result(
    *,
    grid: int = 3,
    width: int = 3,
    registers: int = 1,
    identical: bool = False,
) -> dict:
    patches = [
        [float(1 + (patch * 3 + channel) % 11) for channel in range(width)]
        for patch in range(grid * grid)
    ]
    trt_patches = [
        [value if identical else value + 0.0001 * ((patch + channel) % 3 - 1)
         for channel, value in enumerate(vector)]
        for patch, vector in enumerate(patches)
    ]
    prefix = [[float(100 + token * width + channel) for channel in range(width)]
              for token in range(1 + registers)]

    def stage(spatial: list[list[float]]) -> dict:
        flat = [value for vector in prefix + spatial for value in vector]
        return {
            "stage_name": "full_inference",
            "data": {
                "last_hidden_state": {
                    "shape": [1, 1 + registers + grid * grid, width],
                    "data": flat,
                },
                "pooler_output": {"shape": [1, width], "data": prefix[0]},
                "num_register_tokens": registers,
            },
        }

    return {
        "case_name": "dinov3-report-test",
        "status": "pass",
        "oracle_level": "L1_external_reference",
        "case_config": {
            "name": "dinov3-report-test",
            "family": "dinov3",
            "task_strategy": "image_feature_extraction",
            "reference_backend": "timm_dinov3",
            "inputs": {"image": "tests/e2e/models/dinov3/data/test_img.jpeg"},
        },
        "stage_outputs": {
            "trt_full_inference": stage(trt_patches),
            "ref_full_inference": stage(patches),
        },
        "stages": {
            "full_inference": {
                "metrics": {
                    "full_cosine": {
                        "value": 0.9999,
                        "threshold": 0.999,
                        "operator": ">=",
                        "passed": True,
                    }
                }
            }
        },
        "repro_commands": {"TRT": "trtmc extract-features model.bundle --image input.jpg"},
        "detailed_timing": {"comparison_s": 0.01},
    }


def _pngs(markup: str) -> list[tuple[int, int, bytes]]:
    images = []
    for encoded in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', markup):
        payload = base64.b64decode(encoded, validate=True)
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        offset, width, height = 8, 0, 0
        while offset < len(payload):
            length = struct.unpack(">I", payload[offset : offset + 4])[0]
            kind = payload[offset + 4 : offset + 8]
            data = payload[offset + 8 : offset + 8 + length]
            crc = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])[0]
            assert zlib.crc32(kind + data) & 0xFFFFFFFF == crc
            if kind == b"IHDR":
                width, height = struct.unpack(">II", data[:8])
            offset += 12 + length
        images.append((width, height, payload))
    return images


def _generator_module():
    path = _REPO_ROOT / "scripts" / "generate_e2e_report.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("dinov3_report_generator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_maps_are_bounded_shared_and_self_contained() -> None:
    rendered = report.render(_result(grid=14, width=4, registers=4), project_dir=_REPO_ROOT)
    pngs = _pngs(rendered)

    assert rendered.count('class="dinov3-card"') == 8
    assert rendered.count('data-kind="reference"') == 8
    assert rendered.count('data-kind="trt"') == 8
    assert rendered.count('data-kind="delta"') == 8
    assert [int(value) for value in re.findall(
        r'class="dinov3-card" data-query-index="([0-9]+)"', rendered
    )] == [45, 49, 52, 101, 108, 143, 147, 150]
    assert len(pngs) == 24
    assert all((width, height) == (14, 14) and len(payload) < 1024
               for width, height, payload in pngs)
    for card in re.findall(r'<article class="dinov3-card".*?</article>', rendered):
        scales = re.findall(
            r'data-kind="(?:reference|trt)"[^>]+data-scale-min="([^"]+)" '
            r'data-scale-max="([^"]+)"', card
        )
        assert len(scales) == 2 and scales[0] == scales[1]
        assert 'data-kind="delta"' in card
        assert 'data-scale-min="0.000000000" data-scale-max="0.010000000"' in card
    assert len(rendered) < 120_000


def test_shared_reference_query_excludes_cls_and_registers() -> None:
    result = _result(grid=3, width=2, registers=2)
    trt, ref, grid = report._patches(result)
    assert grid == 3
    assert len(trt) == len(ref) == 9

    first = report.render(result, project_dir=_REPO_ROOT)
    for side in ("trt", "ref"):
        values = result["stage_outputs"][f"{side}_full_inference"]["data"][
            "last_hidden_state"
        ]["data"]
        values[:6] = [-999.0, 777.0] * 3
    second = report.render(result, project_dir=_REPO_ROOT)
    assert re.findall(r'data:image/png;base64,[A-Za-z0-9+/=]+', first) == re.findall(
        r'data:image/png;base64,[A-Za-z0-9+/=]+', second
    )


def test_identical_and_degenerate_features_remain_truthful() -> None:
    identical = report.render(_result(identical=True), project_dir=_REPO_ROOT)
    assert set(re.findall(r'max map \|Δ\| <strong>([0-9.]+)', identical)) == {"0.000000"}

    result = _result(grid=3, width=2, registers=0)
    for side in ("trt", "ref"):
        tensor = result["stage_outputs"][f"{side}_full_inference"]["data"]["last_hidden_state"]
        tensor["data"][2:] = [value for _ in range(9) for value in (1.0, 0.0)]
    trt_values = result["stage_outputs"]["trt_full_inference"]["data"]["last_hidden_state"]["data"]
    trt_values[4:6] = [0.0, 1.0]
    rendered = report.render(result, project_dir=_REPO_ROOT)
    assert 'data-scale-min="-1.000000000" data-scale-max="1.000000000"' in rendered


@pytest.mark.parametrize("mutation", ["nonfinite", "zero", "nonsquare", "oversized"])
def test_invalid_feature_payloads_fail_closed(mutation: str) -> None:
    result = _result()
    if mutation == "nonfinite":
        result["stage_outputs"]["ref_full_inference"]["data"]["last_hidden_state"]["data"][6] = (
            float("nan")
        )
    elif mutation == "zero":
        for side in ("trt", "ref"):
            result["stage_outputs"][f"{side}_full_inference"]["data"]["last_hidden_state"][
                "data"
            ][6:9] = [0.0, 0.0, 0.0]
    elif mutation == "nonsquare":
        for side in ("trt", "ref"):
            tensor = result["stage_outputs"][f"{side}_full_inference"]["data"][
                "last_hidden_state"
            ]
            tensor["shape"] = [1, 9, 3]
            tensor["data"] = tensor["data"][:27]
    else:
        result = _result(grid=33, width=1, registers=0)
    rendered = report.render(result, project_dir=_REPO_ROOT)
    assert "feature visualization unavailable" in rendered
    assert not _pngs(rendered)


def test_shared_generator_loads_only_the_family_owned_renderer() -> None:
    generator = _generator_module()
    result = _result()
    rendered = generator.render_model_section(result, project_dir=_REPO_ROOT)

    assert rendered.index("DINOv3 query-patch similarity maps") < rendered.index(
        "Numerical evidence"
    )
    assert '<details class="model-owned-diagnostics">' in rendered
    assert '<details class="model-owned-diagnostics" open>' not in rendered
    assert "full_cosine=0.999900" == generator._key_metric(result)
    assert "Reproduction Commands" in rendered
    assert "Detailed Timing" in rendered


def test_shared_metrics_are_strict_precise_and_escaped() -> None:
    generator = _generator_module()
    result = _result()
    metric = result["stages"]["full_inference"]["metrics"]["full_cosine"]
    metric["passed"] = "false"
    metric["threshold"] = '</td><script>alert("threshold")</script><td>'
    rendered = generator.render_model_section(result, project_dir=_REPO_ROOT)

    assert "INVALID" in rendered
    assert "0.999900" in rendered
    assert '<script>alert("threshold")</script>' not in rendered
    assert "&lt;/td&gt;&lt;script&gt;alert" in rendered


def test_malformed_payloads_cannot_abort_the_combined_report() -> None:
    generator = _generator_module()
    malformed = _result()
    malformed["stage_outputs"] = []
    rendered = generator.render_model_section(malformed, project_dir=_REPO_ROOT)
    assert "visualization unavailable" in rendered

    huge = _result()
    huge["stage_outputs"]["ref_full_inference"]["data"]["last_hidden_state"]["data"][6] = (
        10**1000
    )
    rendered = generator.render_model_section(huge, project_dir=_REPO_ROOT)
    assert "visualization unavailable" in rendered


def test_renderer_rejects_unsafe_family_and_escapes_input_name(tmp_path: Path) -> None:
    generator = _generator_module()
    result = _result()
    result["case_config"]["family"] = "../dinov3"
    assert generator._render_model_owned_evidence(result, _REPO_ROOT) is None

    image = tmp_path / 'input<script>alert(1).jpeg'
    image.write_bytes(b"\xff\xd8safe\xff\xd9")
    safe = _result()
    safe["case_config"]["inputs"]["image"] = image.name
    rendered = report.render(safe, project_dir=tmp_path)
    assert '<script>alert(1)</script>' not in rendered
    assert "input&lt;script&gt;alert(1).jpeg" in rendered

    safe["case_config"]["inputs"]["image"] = str(image.resolve())
    assert "data:image/jpeg;base64," in report.render(safe, project_dir=tmp_path)

    oversized = tmp_path / "oversized.jpeg"
    oversized.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    safe["case_config"]["inputs"]["image"] = oversized.name
    assert "data:image/jpeg;base64," not in report.render(safe, project_dir=tmp_path)


def test_trt_values_outside_the_reference_scale_are_disclosed() -> None:
    result = _result(grid=3, width=2, registers=0)
    ref = result["stage_outputs"]["ref_full_inference"]["data"]["last_hidden_state"]["data"]
    trt = result["stage_outputs"]["trt_full_inference"]["data"]["last_hidden_state"]["data"]
    ref[2:] = [value for _ in range(9) for value in (1.0, 0.0)]
    ref[4:6] = [0.0, 1.0]
    trt[2:] = list(ref[2:])
    trt[6:8] = [-1.0, 0.0]

    rendered = report.render(result, project_dir=_REPO_ROOT)
    assert "TRT cells clipped" in rendered
