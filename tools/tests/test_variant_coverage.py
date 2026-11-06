# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import variant_coverage


_MODEL_FP8 = """
def build(request, writer):
    if request.quantization not in {None, "none", "fp8"}:
        raise NotImplementedError(request.quantization)
    source_quantization = "nvfp4"
    if source_quantization in {"int8_sq"}:
        pass
    if request.quantization != policy["weights"]:
        pass
"""

_HARDWARE_TEST = """
import pytest
from tools.variant_coverage import covers_variant

@pytest.mark.trt
@covers_variant(precision="bf16", quantization="fp8")
def test_tiny_fp8_engine():
    pass
"""


def _family(
    repo: Path,
    name: str,
    *,
    model: str = "def build(request, writer):\n    pass\n",
    variants: str | None = None,
    manifests: dict[str, dict] | None = None,
    tests: dict[str, str] | None = None,
) -> Path:
    root = repo / "families" / name
    (root / "runtime").mkdir(parents=True)
    (root / "tests/manifests").mkdir(parents=True)
    (root / "model.py").write_text(model)
    argument = f"    variants={variants},\n" if variants is not None else ""
    (root / "support.py").write_text(
        "from tensorrt_model_connect.model_support import Variant, family_support\n\n"
        "describe = family_support(\n"
        f'    model_types=("{name}",),\n    tasks=("generation",),\n'
        f'    default_task="generation",\n{argument})\n'
    )
    (root / "runtime/CMakeLists.txt").write_text("# owned\n")
    (root / "tests/test_e2e.py").write_text("def test_e2e():\n    pass\n")
    for manifest_name, fields in (manifests or {}).items():
        payload = {"name": manifest_name, "family": name, **fields}
        (root / f"tests/manifests/{manifest_name}.json").write_text(json.dumps(payload))
    for file_name, text in (tests or {}).items():
        (root / "tests" / file_name).write_text(text)
    return root


def _repo(tmp_path: Path, pin: str = "11.4.0.106") -> Path:
    (tmp_path / "pyproject.toml").write_text(f'dependencies = ["tensorrt=={pin}"]\n')
    (tmp_path / "tools/ci").mkdir(parents=True)
    (tmp_path / "tools/ci/docker_image.py").write_text('IMAGE = "trtmc-ci:trt11.4"\n')
    (tmp_path / "Dockerfile").write_text(f'RUN pip install "tensorrt=={pin}"\n')
    return tmp_path


def test_accepted_quantizations_read_only_request_literals(tmp_path: Path) -> None:
    root = _family(_repo(tmp_path), "alpha", model=_MODEL_FP8)
    assert variant_coverage.accepted_quantizations(root) == {"fp8"}


def test_undeclared_family_reports_quantization_without_manifest(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _family(repo, "alpha", model=_MODEL_FP8, manifests={"a": {"precision": "bf16"}})
    _family(
        repo,
        "beta",
        model=_MODEL_FP8,
        manifests={"b": {"precision": "bf16"}, "b-fp8": {"precision": "bf16", "quantization": "fp8"}},
    )

    report = variant_coverage.build_report(repo)

    gaps = {item.family: item.undeclared_quantizations for item in report.families}
    assert gaps == {"alpha": ["fp8"], "beta": []}
    assert not any(item.declared for item in report.families)


def test_declared_family_reports_every_gap(tmp_path: Path) -> None:
    root = _family(
        _repo(tmp_path),
        "alpha",
        model=_MODEL_FP8.replace('"fp8"}', '"fp8", "nvfp4"}'),
        variants='(Variant("bf16"), Variant("bf16", "fp8"))',
        manifests={"a": {"precision": "bf16"}, "a-fp16": {"precision": "fp16"}},
        tests={"test_trt.py": "import pytest\n\n@pytest.mark.trt\ndef test_x():\n    pass\n"},
    )

    report = variant_coverage.family_report(root)

    assert report.declared
    assert report.undeclared_quantizations == ["nvfp4"]
    assert report.variants_without_manifest == ["bf16+fp8"]
    assert report.variants_without_test == ["bf16+none", "bf16+fp8"]
    assert report.manifests_without_variant == ["a-fp16"]
    assert report.unannotated_hardware_tests == ["test_trt.py"]


def test_declared_family_with_full_coverage_has_no_gap(tmp_path: Path) -> None:
    root = _family(
        _repo(tmp_path),
        "alpha",
        model=_MODEL_FP8,
        variants=(
            '(Variant("bf16", status="experimental"), '
            'Variant("bf16", "fp8", manifests=("a-fp8-full",)))'
        ),
        manifests={"a-fp8-full": {"precision": "bf16", "quantization": "fp8"}},
        tests={"test_trt.py": _HARDWARE_TEST},
    )

    report = variant_coverage.family_report(root)

    assert report.gaps == 0
    assert report.unannotated_hardware_tests == []


def test_variants_must_be_literal_variant_calls(tmp_path: Path) -> None:
    root = _family(_repo(tmp_path), "alpha", variants="VARIANTS")
    with pytest.raises(ValueError, match="literal tuple"):
        variant_coverage.declared_variants(root)


def test_toolchain_mismatches_cover_dockerfiles_and_image_tag(tmp_path: Path) -> None:
    repo = _repo(tmp_path, pin="11.5.0.1")
    (repo / "Dockerfile.dev.x86-gpu").write_text(
        "RUN python -c \"import tensorrt; assert tensorrt.__version__ == '11.1.0.106'\"\n"
        "ARG TENSORRT_VERSION=11.5.0.1\n"
    )

    mismatches = variant_coverage.toolchain_mismatches(repo)

    assert mismatches == [
        "Dockerfile.dev.x86-gpu:1: 11.1.0.106 != 11.5.0.1",
        "tools/ci/docker_image.py: trt11.4 != 11.5.0.1",
    ]


def test_report_exits_zero_unless_strict(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    _family(repo, "alpha", model=_MODEL_FP8, manifests={"a": {"precision": "bf16"}})
    arguments = ["--repo-root", str(repo), "report"]

    assert variant_coverage.main(arguments) == 0
    assert "alpha: fp8" in capsys.readouterr().out
    assert variant_coverage.main([*arguments, "--strict"]) == 1


def test_bootstrap_drafts_manifest_variants_and_experimental_quantizations(
    tmp_path: Path,
) -> None:
    root = _family(
        _repo(tmp_path),
        "alpha",
        model=_MODEL_FP8,
        manifests={"a": {"precision": "bf16"}},
    )

    draft = variant_coverage.bootstrap(root)

    assert 'Variant(precision="bf16"),' in draft
    assert 'Variant(precision="bf16", quantization="fp8", status="experimental")' in draft


def test_covers_variant_marks_the_test_without_changing_it() -> None:
    def test_case() -> None:
        pass

    marked = variant_coverage.covers_variant(precision="fp16", quantization="nvfp4")(test_case)

    assert marked is test_case
    assert marked.trtmc_variant == ("fp16", "nvfp4")


def test_unknown_family_is_an_error_not_a_clean_report(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    _family(repo, "alpha", model=_MODEL_FP8)

    assert variant_coverage.main(["--repo-root", str(repo), "report", "--family", "alpah"]) == 2
    assert variant_coverage.main(["--repo-root", str(repo), "bootstrap", "alpah"]) == 2
    assert "unknown family: alpah" in capsys.readouterr().err
