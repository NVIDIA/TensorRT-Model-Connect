# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the embedding proof executable in the family GPU plan."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tools.community_gpu_ci import family_plan
from families.qwen.tests import test_e2e


def test_embedding_is_selected_with_its_pinned_checkpoint():
    plan = family_plan(Path(__file__).resolve().parents[3], "qwen")
    assert "qwen3-embedding-0.6b" in plan.testcases
    assert (
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ) in plan.checkpoints


def test_embedding_consumer_is_built_outside_isolated_runtime(tmp_path, monkeypatch):
    if not shutil.which("cmake") or not shutil.which("cc"):
        pytest.skip("native build probe requires CMake and a C compiler")
    source = tmp_path / "source"
    source.mkdir()
    (source / "probe.c").write_text("int main(void) { return 0; }\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\nproject(probe C)\n"
        "add_executable(qwen_embedding_consumer probe.c)\n"
        "set_target_properties(qwen_embedding_consumer PROPERTIES "
        'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/families/qwen")\n'
    )
    build_dir = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build_dir)], check=True)
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(build_dir))
    binary = test_e2e._embedding_consumer_binary()
    assert binary.parent == build_dir / "families/qwen"
    subprocess.run([str(binary)], check=True)


def test_embedding_consumer_requires_native_build(monkeypatch):
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR", raising=False)
    with pytest.raises(AssertionError, match="TRTMC_NATIVE_BUILD_DIR"):
        test_e2e._embedding_consumer_binary()


@pytest.mark.parametrize("cached", [True, False])
def test_checkpoint_uses_only_the_staged_revision(tmp_path, monkeypatch, cached):
    import httpx
    from huggingface_hub import constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    revision = "a" * 40
    snapshot = tmp_path / "models--Qwen--offline-probe" / "snapshots" / revision
    if cached:
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    def reject_network(*args, **kwargs):
        raise AssertionError("checkpoint lookup attempted network access")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    manifest = {"hf_id": "Qwen/offline-probe", "hf_revision": revision}
    if cached:
        assert test_e2e._checkpoint(manifest) == snapshot
    else:
        with pytest.raises(LocalEntryNotFoundError):
            test_e2e._checkpoint(manifest)
