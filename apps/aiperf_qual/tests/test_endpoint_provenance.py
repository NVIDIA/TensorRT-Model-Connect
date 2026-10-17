# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.endpoint_profile import source_identity, tokenizer_identity


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    for arguments in (["init", "-q"], ["config", "user.name", "Test"],
                      ["config", "user.email", "test@example.com"]):
        subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)
    (root / "source.py").write_text("value = 0\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "--no-gpg-sign", "-qm", "fixture"], cwd=root, check=True)
    return root


def test_changes_to_an_already_dirty_tracked_file_change_identity(repository):
    source = repository / "source.py"
    source.write_text("value = 1\n")
    before = source_identity(repository, repository / "output")
    source.write_text("value = 2\n")
    after = source_identity(repository, repository / "output")
    assert before["source_head"] == after["source_head"]
    assert before["source_status"] == after["source_status"]
    assert before["source_contents_sha256"] != after["source_contents_sha256"]


def test_untracked_source_content_changes_are_detected(repository):
    source = repository / "new_source.py"
    source.write_text("value = 1\n")
    before = source_identity(repository, repository / "output")
    source.write_text("value = 2\n")
    after = source_identity(repository, repository / "output")
    assert before["source_status"] == after["source_status"]
    assert before["source_contents_sha256"] != after["source_contents_sha256"]


def test_generated_output_does_not_change_source_identity(repository):
    output = repository / "output [run]"
    output.mkdir()
    before = source_identity(repository, output)
    (output / "nested").mkdir()
    (output / "nested/records.json").write_text('{"requests": 55}\n')
    assert source_identity(repository, output) == before


def test_local_tokenizer_files_are_identified_by_contents(tmp_path):
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    config = directory / "tokenizer.json"
    config.write_text('{"version": 1}\n')
    before = tokenizer_identity(str(directory), None, override=True)
    config.write_text('{"version": 2}\n')
    after = tokenizer_identity(str(directory), None, override=True)
    assert before["name"] == after["name"] and before["revision"] is after["revision"] is None
    assert before["files_sha256"] != after["files_sha256"]


@pytest.mark.parametrize("revision", [None, "main", "v1.0"])
def test_remote_tokenizer_override_requires_an_immutable_revision(revision):
    with pytest.raises(ConfigError, match="full commit SHA"):
        tokenizer_identity("organization/tokenizer", revision, override=True)
    pin = "a" * 40
    assert tokenizer_identity("organization/tokenizer", pin, override=True) == {
        "name": "organization/tokenizer", "revision": pin,
    }
