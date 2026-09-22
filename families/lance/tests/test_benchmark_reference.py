# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from families.lance.tests.benchmark import upstream_reference


ROOT = Path(__file__).resolve().parent


def test_performance_uses_the_family_owned_reference(tmp_path: Path) -> None:
    profile = yaml.safe_load(
        (ROOT / "benchmark/lance-3b-x2t-image.yaml").read_text(encoding="utf-8")
    )
    reference = profile["performance"][0]["reference"]
    assert reference["script"] == "tests/benchmark/reference.py"
    assert "adapter" not in reference

    checkout = tmp_path / "Lance"
    parsed = upstream_reference.build_parser().parse_args(
        [
            "--reference-repo",
            str(checkout),
            "--model",
            "model",
            "--image",
            str(tmp_path / "image.png"),
            "--prompt",
            "describe",
            "--max-new-tokens",
            "4",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output",
            str(tmp_path / "output.json"),
        ]
    )
    assert parsed.reference_repo == checkout
