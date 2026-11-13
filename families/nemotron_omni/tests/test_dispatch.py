# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known command failures select full native tasks, never quality-based gates."""

from types import SimpleNamespace
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from families.nemotron_omni.edge_llm import dispatch
from families.nemotron_omni.support import SUPPORTED_TASKS


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_tasks_select_only_available_edge_commands(task, tmp_path, monkeypatch):
    calls = []
    request = SimpleNamespace(task=task, precision="fp16", output_path=tmp_path / "model.bundle")
    writer = object()
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    monkeypatch.setattr(dispatch.builder, "package_present", lambda: True)
    monkeypatch.setattr(dispatch.builder, "local_target", lambda: {})
    monkeypatch.setattr(dispatch.builder, "prepare", lambda *args: ({}, {}))
    monkeypatch.setattr(dispatch.builder, "publish", lambda *args: calls.append("edge"))
    dispatch.build(request, writer, lambda *args: calls.append("native"), 4096)
    edge = {
        "text_generation",
        "text_continuation",
        "vision_language_generation",
        "images_text_to_text",
    }
    assert calls == ["edge" if task in edge else "native"]
    assert not list(tmp_path.iterdir())


def test_builder_error_preserved_and_native_called_once(tmp_path, monkeypatch, caplog):
    request = SimpleNamespace(
        task="text_generation", precision="fp16", output_path=tmp_path / "model.bundle"
    )
    calls = []
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    monkeypatch.setattr(dispatch.builder, "package_present", lambda: True)

    def fail():
        raise RuntimeError("upstream builder fixture failure")

    monkeypatch.setattr(dispatch.builder, "local_target", fail)
    dispatch.build(request, object(), lambda *args: calls.append("native"), 4096)
    assert calls == ["native"]
    assert "Retrying native once" in caplog.text
    logs = list(tmp_path.glob("*.log"))
    assert len(logs) == 1 and "upstream builder fixture failure" in logs[0].read_text()


def test_edge_checkpoint_metadata_and_hf_cache_links(tmp_path, monkeypatch):
    """A normal linked HF snapshot must not fall back or publish private notes."""
    builder = dispatch.builder
    source, staging = tmp_path / "snapshot", tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    blob = tmp_path / "cached-blob"
    blob.write_bytes(b"prepared weights fixture")
    shard = source / "model-00001.safetensors"
    shard.symlink_to(blob)
    index = source / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}))
    (source / "config.json").write_text("{}")
    (source / "private-notes.json").write_text("{}")
    request = SimpleNamespace(model_dir=source, verbose=False)
    monkeypatch.setattr(builder, "installed_package", lambda _: {"plugin": "fixture.so"})
    monkeypatch.setattr(builder, "builder_python", lambda *_: "fixture-python")

    def build_fixture(command, **kwargs):
        engine = Path(command[command.index("--engine-dir") + 1])
        for name in (
            "llm.engine",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "visual/visual.engine",
            "visual/config.json",
        ):
            path = engine / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"builder output fixture")

    monkeypatch.setattr(builder.subprocess, "run", build_fixture)
    files, _ = builder.prepare(request, {}, staging, tmp_path / "build.log", 4096)
    assert files["edge_llm/checkpoint/" + shard.name].read_bytes() == blob.read_bytes()
    assert "edge_llm/checkpoint/config.json" in files
    assert "edge_llm/checkpoint/private-notes.json" not in files
    index.write_text(json.dumps({"weight_map": {"weight": "../cached-blob"}}))
    with pytest.raises(ValueError, match="shard path is not normalized"):
        builder.prepare(request, {}, staging, tmp_path / "build.log", 4096)


@pytest.fixture(scope="module")
def edge_contract_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Owning Edge contract checks require a C++ compiler")
    root = Path(__file__).parents[3]
    directory = tmp_path_factory.mktemp("omni-edge-contract")
    source = directory / "probe.cpp"
    source.write_text(r"""#include "families/nemotron_omni/runtime/edge_llm/contract.h"
#include <iostream>
int main(int argc, char** argv) {
    using namespace trtmc::nemotron_omni::edge_llm;
    try {
        if (argc == 2) std::cout << runtime_os_version("kernel-fixture", argv[1]);
        else if (argc == 3) require_artifact_space(std::stoull(argv[1]), std::stoull(argv[2]));
        else return 2;
    } catch (const std::exception& e) { std::cerr << e.what(); return 1; }
}
""")
    executable = directory / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root),
            "-I",
            str(root / "core/runtime/include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize(
    "etc,usr,expected",
    [
        ('VERSION_ID="24.04"\n', 'VERSION_ID="ignored"\n', "24.04"),
        (None, "VERSION_ID='12'\n", "12"),
        ("NAME=fixture\n", "VERSION_ID=ignored\n", "kernel-fixture"),
        (None, None, "kernel-fixture"),
        ("VERSION_ID=22.04\nVERSION_ID=24.04\n", None, "24.04"),
        ('VERSION_ID=""\n', None, ""),
    ],
)
def test_edge_os_release_matches_builder_fallback(
    edge_contract_probe, tmp_path, etc, usr, expected
):
    for filename, content in (("etc/os-release", etc), ("usr/lib/os-release", usr)):
        if content is not None:
            path = tmp_path / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    result = subprocess.run(
        [str(edge_contract_probe), str(tmp_path)], check=True, capture_output=True, text=True
    )
    assert result.stdout == expected


def test_edge_disk_admission_reports_required_space(edge_contract_probe):
    subprocess.run([str(edge_contract_probe), "1024", "1024"], check=True)
    result = subprocess.run(
        [str(edge_contract_probe), "1024", "1023"], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "1024 bytes" in result.stderr and "1023" in result.stderr and "TMPDIR" in result.stderr
