# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from families.minimax_h3 import delivery


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import huggingface_hub
    from torch import hub

    def unexpected_download(*_args, **_kwargs):
        pytest.fail("Delivery test attempted an unmocked download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)
    monkeypatch.setattr(hub, "download_url_to_file", unexpected_download)


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint fixture")
    return path


def test_quantized_source_manifest_pins_full_denoisers_and_nvfp4_text() -> None:
    from families.minimax_h3.nvfp4_text_checkpoint import QUANTIZED_TEXT_CHECKPOINT_IDENTITY
    from families.minimax_h3.quantized_checkpoint import (
        QUANTIZED_CHECKPOINT_IDENTITY,
        QUANTIZED_REF2VA_CHECKPOINT_IDENTITY,
    )

    identities = (
        QUANTIZED_CHECKPOINT_IDENTITY,
        QUANTIZED_REF2VA_CHECKPOINT_IDENTITY,
        QUANTIZED_TEXT_CHECKPOINT_IDENTITY,
    )
    assert tuple(delivery.QUANTIZED_SOURCES.values()) == tuple(
        value.filename for value in identities
    )
    assert all(value.model_id == delivery.COMFY_REPOSITORY for value in identities)
    assert all(value.revision == delivery.COMFY_REVISION for value in identities)
    assert delivery.COMFY_REVISION == "4cc1d817b6184899b41293954329f576cb5ae86b"
    assert not any("pruned" in filename for filename in delivery.QUANTIZED_SOURCES.values())


@pytest.mark.parametrize("location", ("explicit", "model_dir", "hf_cache"))
def test_all_quantized_sources_resolve_with_explicit_offline_and_pinned_cache_precedence(
    tmp_path: Path, monkeypatch, location: str
) -> None:
    import huggingface_hub

    model = tmp_path / "model"
    options, expected, downloads = {}, {}, []
    for option, filename in delivery.QUANTIZED_SOURCES.items():
        if location in {"explicit", "model_dir"}:
            expected[option] = _file(model / filename)
        if location == "explicit":
            # An explicit file wins even when the model directory also has it.
            expected[option] = _file(tmp_path / "override" / filename)
            options[option] = str(expected[option])
        elif location == "hf_cache":
            expected[option] = _file(tmp_path / "cache" / filename)

    def download(repository, filename, *, revision):
        downloads.append((repository, filename, revision))
        return str(tmp_path / "cache" / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert delivery.resolve_quantized_sources(model, options) == expected
    assert downloads == (
        [
            (delivery.COMFY_REPOSITORY, filename, delivery.COMFY_REVISION)
            for filename in delivery.QUANTIZED_SOURCES.values()
        ]
        if location == "hf_cache"
        else []
    )


@pytest.mark.parametrize("option", tuple(delivery.QUANTIZED_SOURCES))
def test_missing_explicit_quantized_source_never_falls_back_to_download(
    tmp_path: Path, option: str
) -> None:
    options = {
        key: str(_file(tmp_path / filename)) for key, filename in delivery.QUANTIZED_SOURCES.items()
    }
    options[option] = str(tmp_path / "missing.safetensors")
    with pytest.raises(FileNotFoundError, match=option):
        delivery.resolve_quantized_sources(tmp_path, options)


@pytest.mark.parametrize(
    "options", ({}, {"height": 480, "width": 864}, {"super_resolution": False})
)
def test_native_resolution_never_automatically_enables_sr(options) -> None:
    assert delivery.resolve_super_resolution_sources(options) == (None, None)


@pytest.mark.parametrize("key", ("super_resolution_model", "super_resolution_weak_model"))
def test_sr_checkpoint_override_requires_explicit_enable(key: str) -> None:
    with pytest.raises(ValueError, match="require super_resolution=true"):
        delivery.resolve_super_resolution_sources({key: "checkpoint.pth"})


def test_explicit_sr_flag_resolves_both_public_models_and_reuses_cached_files(
    tmp_path: Path, monkeypatch
) -> None:
    from torch import hub

    calls = []
    monkeypatch.setattr(hub, "get_dir", lambda: str(tmp_path))

    def download(url, destination):
        calls.append((url, destination))
        _file(Path(destination))

    monkeypatch.setattr(hub, "download_url_to_file", download)
    options = {"super_resolution": True}
    expected = tuple(
        tmp_path / "checkpoints" / filename
        for filename in ("realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth")
    )
    assert delivery.resolve_super_resolution_sources(options) == expected
    assert calls == [
        (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/" + path.name,
            str(path),
        )
        for path in expected
    ]
    assert delivery.resolve_super_resolution_sources(options) == expected
    assert len(calls) == 2
    assert delivery.SR_SOURCE_SHAPE == (480, 864)


def test_explicit_sr_files_work_offline_and_missing_override_fails(tmp_path: Path) -> None:
    primary = _file(tmp_path / "primary.pth")
    weak = _file(tmp_path / "weak.pth")
    options = {
        "super_resolution": True,
        "super_resolution_model": str(primary),
        "super_resolution_weak_model": str(weak),
    }
    assert delivery.resolve_super_resolution_sources(options) == (primary, weak)
    with pytest.raises(FileNotFoundError, match="SR checkpoint is missing"):
        delivery.resolve_super_resolution_sources(
            {**options, "super_resolution_model": "missing.pth"}
        )
