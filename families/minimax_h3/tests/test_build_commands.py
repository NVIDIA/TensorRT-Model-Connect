# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.minimax_h3 import cli_rtx
from tensorrt_model_connect.__main__ import main


def test_turbo_build_routes_local_sources_and_geometry(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli_rtx, "_build", lambda request, options: calls.append((request, options)) or 0
    )
    assert (
        main(
            [
                "minimax_h3",
                "build-rtx",
                str(tmp_path),
                "-o",
                str(tmp_path / "out.bundle"),
                "--turbo",
                "--turbo-base-precision",
                "int8",
                "--turbo-lora",
                str(tmp_path / "lora"),
                "--image-height",
                "736",
                "--image-width",
                "1280",
                "--video-num-frames",
                "362",
            ]
        )
        == 0
    )
    ((request, options),) = calls
    assert (
        request.backend,
        request.image_height,
        request.image_width,
        request.video_num_frames,
    ) == (
        "trt_rtx",
        736,
        1280,
        362,
    )
    assert options["options"] == {
        "turbo": True,
        "turbo_base_precision": "int8",
        "turbo_lora": str(tmp_path / "lora"),
    }


@pytest.mark.parametrize(
    "args,message",
    [
        (["build-rtx", "--turbo", "--first-block-cache", "true"], "without FirstBlockCache"),
        (["build-rtx", "--turbo", "--super-resolution"], "does not support super-resolution"),
        (["build-trt", "--first-block-cache", "--quantization", "fp8"], "monolithic denoiser"),
    ],
)
def test_incompatible_profiles_fail_before_checkpoint_lookup(monkeypatch, tmp_path, args, message):
    monkeypatch.setattr(
        cli_rtx, "resolve_model", lambda *_: pytest.fail("unexpected checkpoint lookup")
    )
    with pytest.raises(ValueError, match=message):
        main(
            [
                "minimax_h3",
                args[0],
                "Example/Checkpoint",
                "-o",
                str(tmp_path / "out.bundle"),
                *args[1:],
            ]
        )


@pytest.mark.parametrize(
    "model,revision",
    [
        ("MiniMaxAI/MiniMax-H3", "main"),
        ("MiniMaxAI/MiniMax-H3", "d" * 40),
        ("Example/Other", cli_rtx.CHECKPOINT_REVISION),
    ],
)
def test_rtx_remote_identity_rejected_before_download(monkeypatch, tmp_path, model, revision):
    monkeypatch.setattr(cli_rtx, "resolve_model", lambda *_: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="MiniMax-H3 RTX requires"):
        main(
            ["minimax_h3", "build-rtx", model, "--revision", revision, "-o", str(tmp_path / "out")]
        )


def test_official_rtx_remote_is_resolved_at_exact_revision(monkeypatch, tmp_path):
    calls = []

    def resolve(model, revision):
        calls.append((model, revision))
        return tmp_path

    monkeypatch.setattr(cli_rtx, "resolve_model", resolve)
    assert cli_rtx._resolve_rtx_checkpoint(cli_rtx.CHECKPOINT_REPOSITORY, None) == tmp_path
    assert calls == [(cli_rtx.CHECKPOINT_REPOSITORY, cli_rtx.CHECKPOINT_REVISION)]


def test_unversioned_local_directory_cannot_attest_to_requested_revision(tmp_path):
    with pytest.raises(ValueError, match="Cannot verify --revision"):
        cli_rtx._resolve_rtx_checkpoint(str(tmp_path), cli_rtx.CHECKPOINT_REVISION)
