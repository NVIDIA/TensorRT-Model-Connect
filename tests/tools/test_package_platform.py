# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-contract tests for release wheel selection and auditing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tools.ci.package import (
    WheelArchiveValidator,
    WheelPackageManager,
    _validate_manylinux_audit,
)


def test_x86_64_wheel_platform_is_accepted_and_selected(tmp_path: Path) -> None:
    wheel = tmp_path / "dist" / (
        "tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_x86_64.whl"
    )
    wheel.parent.mkdir()
    wheel.touch()
    context = SimpleNamespace(
        repository=tmp_path,
        env={"TRTMC_PACKAGE_WHEEL_ARCH": "manylinux_2_39_x86_64"},
    )

    validator = WheelArchiveValidator(context, "manylinux_2_39_x86_64")

    assert validator.architecture == "x86_64"
    assert WheelPackageManager(context).select_wheel("py312") == wheel


def test_x86_64_auditwheel_result_is_validated_for_its_architecture(tmp_path: Path) -> None:
    wheel = tmp_path / (
        "tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_x86_64.whl"
    )

    _validate_manylinux_audit(
        wheel,
        'The wheel is consistent with the following platform tag: '
        '"manylinux_2_39_x86_64".',
        platform="manylinux_2_39_x86_64",
        architecture="x86_64",
        max_glibc_minor=39,
    )
