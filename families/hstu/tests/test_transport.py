# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import sys
import pytest
from families.hstu.config import Config
from families.hstu.triton.native_client import NativeClient, save_snapshot, snapshot_metadata
from families.hstu.triton import native_repository


def test_snapshot_atomic_publication_and_bounded_metadata(tmp_path):
    p = tmp_path / "history.kv"
    save_snapshot(p, b"old")
    with p.open("rb") as reader:
        save_snapshot(p, b"new")
        assert reader.read() == b"old"
    assert p.read_bytes() == b"new" and list(tmp_path.iterdir()) == [p]
    header = json.dumps({"revisions": ["public-v1"]}).encode()
    assert snapshot_metadata(b"HSTUKV1\0" + struct.pack("<Q", len(header)) + header)[
        "revisions"
    ] == ["public-v1"]
    for bad in [
        b"",
        b"HSTUKV1\0" + struct.pack("<Q", 65537),
        b"HSTUKV1\0" + struct.pack("<Q", len(header) + 1) + header,
    ]:
        with pytest.raises(ValueError):
            snapshot_metadata(bad)


@pytest.mark.parametrize("option", ["max_entries", "ttl_seconds", "lease_seconds"])
def test_client_rejects_ignored_controls_without_loading(option):
    with pytest.raises(ValueError, match="hydration=True"):
        NativeClient("unused.bundle", "unused-sdk", **{option: 1})


def test_native_repository_uses_public_tensor_metadata(tmp_path, monkeypatch):
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"example bundle")
    monkeypatch.setattr(
        native_repository, "runtime_metadata", lambda _: Config(candidates=64).runtime(4, "bf16")
    )
    out = native_repository.generate(
        bundle, tmp_path / "models", "hstu", parameters={"kv_cache_memory_ratio": 0.05}
    )
    text = (out / "config.pbtxt").read_text()
    assert 'backend: "tensorrt"' in text and 'platform: "model_connect"' in text
    assert "dims: [128]" in text and "dims: [64]" in text and "dims: [50]" in text
    assert " },\n]" not in text
    assert 'key: "kv_cache_memory_ratio" value { string_value: "0.05" }' in text
    assert (out / "1/model.bundle").read_bytes() == bundle.read_bytes()
    with pytest.raises(FileExistsError):
        native_repository.generate(bundle, tmp_path / "models", "hstu")


@pytest.mark.parametrize(
    "flags",
    [
        ["--kv-cache-memory-ratio", "0"],
        ["--kv-cache-memory-ratio", "nan"],
        ["--kv-cache-memory-ratio", "1.1"],
        ["--kv-cache-size-bytes", "0"],
        ["--kv-cache-size-bytes", "8", "--kv-cache-memory-ratio", ".1"],
    ],
)
def test_cli_rejects_invalid_budgets_before_writing(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_repository",
            "--bundle",
            "unused.bundle",
            "--repository",
            str(tmp_path / "models"),
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        native_repository.main()
    assert error.value.code == 2 and not (tmp_path / "models").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        {"kv_cache_size_bytes": -1},
        {"kv_cache_size_bytes": "8388608junk"},
        {"kv_cache_size_bytes": True},
        {"kv_cache_size_bytes": 2**64},
        {"kv_cache_memory_ratio": "0.05junk"},
        {"kv_cache_memory_ratio": float("nan")},
        {"kv_cache_memory_ratio": 0.0},
        {"kv_cache_size_bytes": 8, "kv_cache_memory_ratio": 0.1},
        {"kv_cache_hydration_enabled": True},
    ],
)
def test_repository_api_rejects_invalid_options_before_writing(tmp_path, parameters):
    with pytest.raises(ValueError):
        native_repository.generate(
            tmp_path / "unused.bundle", tmp_path / "models", "hstu", parameters=parameters
        )
    assert not (tmp_path / "models").exists()
