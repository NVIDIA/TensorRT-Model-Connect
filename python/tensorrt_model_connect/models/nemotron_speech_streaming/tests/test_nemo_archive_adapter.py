# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for writable staging of cached Nemotron speech archives."""

from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path

import yaml
from tensorrt_model_connect.models import resolve_family_model_dir


def test_model_dir_adapter_stages_nemo_snapshot_without_mutating_source(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        '{"model_type":"nemotron_speech_streaming"}\n', encoding="utf-8"
    )
    archive = snapshot / "nemotron-speech.nemo"
    config = {
        "_target_": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "encoder": {"d_model": 16},
    }
    config_bytes = yaml.safe_dump(config).encode("utf-8")
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("model_config.yaml")
        member.size = len(config_bytes)
        tar.addfile(member, io.BytesIO(config_bytes))

    staged_text = resolve_family_model_dir(snapshot)
    assert staged_text is not None
    staged = Path(staged_text)
    try:
        assert staged != snapshot
        assert (staged / archive.name).resolve() == archive.resolve()
        assert (staged / "config.json").is_file()

        (staged / "tokenizer.model").write_bytes(b"generated-tokenizer")
        assert not (snapshot / "tokenizer.model").exists()
    finally:
        shutil.rmtree(staged)


def test_model_dir_adapter_ignores_invalid_nemo_archive(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        '{"model_type":"example_decoder"}\n', encoding="utf-8"
    )
    (snapshot / "unrelated.nemo").write_text("not a tar archive", encoding="utf-8")

    assert resolve_family_model_dir(snapshot) is None


def test_model_dir_adapter_ignores_valid_unrelated_nemo_archive(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    archive = snapshot / "unrelated.nemo"
    config_bytes = yaml.safe_dump(
        {"_target_": "nemo.collections.tts.models.FastPitchModel"}
    ).encode("utf-8")
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("model_config.yaml")
        member.size = len(config_bytes)
        tar.addfile(member, io.BytesIO(config_bytes))

    assert resolve_family_model_dir(snapshot) is None
