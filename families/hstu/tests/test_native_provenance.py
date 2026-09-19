# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exported native provenance identifies dependencies without build paths."""

import json
from pathlib import Path
import shlex
from types import SimpleNamespace

from families.hstu import native_attention_build, native_attention_export
from tensorrt_model_connect.build import content_cache_key


def fake_export(monkeypatch, root, changed_header=b"original header"):
    family = root / "checkout" / "family"
    family.mkdir(parents=True)
    kernel = family / "native_attention_kernel.cu"
    kernel.write_bytes(b"host launch glue")
    (family / "native_attention_source.json").write_text('{"revision":"pinned"}\n')
    source = root / "dependency checkout"
    source.mkdir()
    first, second = source / "first.h", source / "second.h"
    first.write_bytes(changed_header)
    second.write_bytes(b"original header")
    torch_include = root / "torch" / "include"
    (torch_include / "ATen").mkdir(parents=True)
    (torch_include / "ATen/ATen.h").write_bytes(b"mock ATen header")
    dependencies = [kernel, first, second, first]
    commands, verified = [], []
    monkeypatch.setattr(native_attention_export, "HERE", family)
    monkeypatch.setattr(native_attention_build, "verify_source", lambda path: verified.append(path))
    monkeypatch.setattr(native_attention_build, "cutlass_directory", lambda path: root / "cutlass")
    monkeypatch.setattr(native_attention_export.shutil, "which", lambda name: "/mock/nvcc")
    monkeypatch.setattr(native_attention_export.subprocess, "check_output",
                        lambda *args, **kwargs: "sm_80 sm_90 sm_103")
    monkeypatch.setattr(native_attention_export.importlib.metadata, "distribution",
                        lambda name: SimpleNamespace(version="test-version",
                            locate_file=lambda path: torch_include))

    def compile_(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled object")
        depfile = Path(command[command.index("-MF") + 1])
        depfile.write_text("kernel.o: " + " \\\n".join(shlex.quote(str(path)) for path in dependencies))
        return SimpleNamespace(returncode=0, stdout="mock compiler output")

    monkeypatch.setattr(native_attention_export.subprocess, "run", compile_)
    output = root / "export"
    _, receipt = native_attention_export.export_attention(output, source, "sm_103")
    return output, receipt, dependencies, commands, verified


def test_packaged_provenance_has_every_content_identity_without_paths(monkeypatch, tmp_path):
    output, receipt, dependencies, _, _ = fake_export(monkeypatch, tmp_path)
    expected = sorted(content_cache_key("hstu-native-compile-input-v1", path.read_bytes())
                      for path in set(dependencies))
    assert receipt["compile_input_content_keys"] == expected
    assert len(expected) == 3
    assert len(set(expected)) == 2
    assert "compile_inputs" not in receipt
    serialized = json.dumps(receipt)
    assert str(tmp_path) not in serialized
    assert all(path.name not in serialized for path in dependencies)
    local = json.loads((output / "compile.json").read_text())
    assert set(local["compile_inputs"]) == {str(path.resolve()) for path in dependencies}
    assert local["compile_input_content_keys"] == expected


def test_export_identity_ignores_build_location_but_tracks_changed_content(monkeypatch, tmp_path):
    _, first, _, _, _ = fake_export(monkeypatch, tmp_path / "one")
    _, relocated, _, _, _ = fake_export(monkeypatch, tmp_path / "two")
    _, changed, _, _, _ = fake_export(monkeypatch, tmp_path / "three", b"changed header")
    assert first == relocated
    assert first["compile_input_content_keys"] != changed["compile_input_content_keys"]


def test_export_keeps_math_flags_and_source_checks_without_lineinfo(monkeypatch, tmp_path):
    output, _, _, commands, verified = fake_export(monkeypatch, tmp_path)
    assert len(commands) == 1
    command = commands[0]
    assert {"-O3", "--use_fast_math", "-DNDEBUG", "-std=c++20"}.issubset(command)
    assert "-lineinfo" not in command
    assert "arch=compute_103,code=sm_103" in command
    assert len(verified) == 2 and verified[0] == verified[1]
    assert json.loads((output / "command.json").read_text()) == command
