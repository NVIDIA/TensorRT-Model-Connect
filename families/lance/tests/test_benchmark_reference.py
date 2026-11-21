# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from families.lance.tests.benchmark import upstream_reference


def test_upstream_reference_parses_the_reference_checkout(tmp_path: Path) -> None:
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
