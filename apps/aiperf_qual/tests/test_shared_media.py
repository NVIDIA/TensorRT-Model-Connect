# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every model of a run reads one shared copy of a dataset's media; results carry digests, not bytes."""

import base64
import json
from pathlib import Path

import pytest

from trtmc_aiperf_qual import services
from trtmc_aiperf_qual.config import ConfigError, Environment
from trtmc_aiperf_qual.suites import build_suite
from trtmc_perf_serving.files import FILE_KEY, materialize_files

IMAGE_BYTES = {"cat": b"\xff\xd8cat" * 30_000, "dog": b"\xff\xd8dog" * 30_000}  # ~150 KB each, like a JPEG


def images(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / f"{name}.jpg" for name in IMAGE_BYTES}
    for name, path in paths.items():
        path.write_bytes(IMAGE_BYTES[name])
    return paths


def definition(records) -> dict:
    return {"suite": "media-test", "version": "1", "source": {"kind": "inline", "records": records},
            "selection": {"method": "first", "count": len(records)}}


def media_records(paths: dict[str, Path]) -> list[dict]:
    inline = {FILE_KEY: {"suffix": ".jpg", "b64": base64.b64encode(IMAGE_BYTES["dog"]).decode()}}
    return [{"id": "path", "request": {"image_path": str(paths["cat"]), "prompt": "a"}},
            {"id": "inline", "request": {"image_path": inline, "prompt": "b"}}]


def stored_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_the_media_store_is_one_folder_of_the_datasets_cache_unless_configured():
    assert Environment({"hf_datasets_cache": "/state/ds"}).media_root() == Path("/state/ds/trtmc-media")
    assert Environment({"hf_datasets_cache": "/state/ds", "media_root": "/fast/media"}).media_root() == Path("/fast/media")
    assert Environment({}).media_root() is None


def test_models_share_one_copy_of_each_input_and_requests_carry_digests(tmp_path):
    environment = Environment({"hf_datasets_cache": str(tmp_path / "datasets")})
    first = build_suite(definition(media_records(images(tmp_path / "a"))), environment)
    # Another model builds the same suite later, from the same data in another place.
    second = build_suite(definition(media_records(images(tmp_path / "elsewhere" / "b"))), environment)

    store = environment.media_root()
    assert [path.name.endswith(".jpg") for path in stored_files(store)] == [True, True]  # cat and dog, once
    assert [sample["request_sha"] for sample in first.samples] == [sample["request_sha"] for sample in second.samples]
    assert first.key == second.key
    inputs = first.write_inputs(tmp_path / "inputs.jsonl")
    assert "b64" not in inputs.read_text() and inputs.stat().st_size < 2_000  # not ~300 KB of base64
    request = json.loads(json.loads(inputs.read_text().split("\n")[0])["text"])["request"]
    assert set(request["image_path"][FILE_KEY]) == {"suffix", "sha256"} and request["prompt"] == "a"


def test_a_server_reads_back_exactly_the_suite_inputs_from_the_store(tmp_path):
    environment = Environment({"hf_datasets_cache": str(tmp_path / "datasets")})
    suite = build_suite(definition(media_records(images(tmp_path / "a"))), environment)

    resolved = [materialize_files(sample["request"], tmp_path / f"request-{index}", media_root=environment.media_root())
                for index, sample in enumerate(suite.samples)]

    assert [Path(item["image_path"]).read_bytes() for item in resolved] == [IMAGE_BYTES["cat"], IMAGE_BYTES["dog"]]


def test_the_same_input_in_two_suites_is_stored_once(tmp_path):
    environment = Environment({"hf_datasets_cache": str(tmp_path / "datasets")})
    paths = images(tmp_path / "a")
    build_suite(definition([{"id": "x", "request": {"image_path": str(paths["cat"])}}]), environment)
    other = {**definition([{"id": "y", "request": {"image_path": str(paths["cat"]), "top_k": 5}}]), "suite": "other"}

    build_suite(other, environment)

    assert len(stored_files(environment.media_root())) == 1


def test_media_without_a_store_is_a_configuration_error_but_text_needs_none(tmp_path):
    environment = Environment({})

    with pytest.raises(ConfigError, match="media store"):
        build_suite(definition(media_records(images(tmp_path / "a"))), environment)
    text = build_suite(definition([{"id": "t", "request": {"prompt": "hello"}}]), environment)

    assert text.samples[0]["request"] == {"prompt": "hello"}


def test_the_server_is_started_with_the_runs_media_store(tmp_path, monkeypatch):
    environment = Environment({"repo": str(tmp_path), "bundle_root": str(tmp_path / "engines"),
                               "runtime_root": str(tmp_path / "build"), "worker": str(tmp_path / "worker"),
                               "serve_python": "python3", "ports": {"candidate": 9000, "reference": 9100},
                               "hf_datasets_cache": str(tmp_path / "datasets")})
    model = {"catalog_profile": "p", "candidate": {"bundle": "p/p.bundle", "precision": "fp16"}, "reference": {}}

    def refuse(command, **options):
        raise RuntimeError("not started")

    monkeypatch.setattr(services, "_port_free", lambda port: True)
    monkeypatch.setattr(services.subprocess, "Popen", refuse)
    for backend in ("trtmc", "reference"):
        out = tmp_path / backend
        with pytest.raises(RuntimeError, match="not started"):
            with services.serving(environment, model, backend, out):
                pass

        command = json.loads((out / "command.json").read_text())
        assert command[command.index("--media-root") + 1] == str(tmp_path / "datasets" / "trtmc-media")
