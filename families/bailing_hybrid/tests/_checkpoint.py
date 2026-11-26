# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe the checkpoint actually used by this family's parity tests."""

from pathlib import Path


def checkpoint_record(directory, manifest, *, local_override):
    directory = Path(directory).resolve()
    record = {
        "source": "local_override" if local_override else "huggingface_snapshot",
        "path": str(directory),
        "pinned": not local_override,
        "precision": manifest["precision"],
    }
    if local_override:
        record["requested_checkpoint"] = {
            "repo_id": manifest["hf_id"],
            "revision": manifest["hf_revision"],
        }
        return record
    expected_cache = "models--" + manifest["hf_id"].replace("/", "--")
    if (
        directory.name != manifest["hf_revision"]
        or directory.parent.name != "snapshots"
        or directory.parent.parent.name != expected_cache
    ):
        raise AssertionError("Cached checkpoint snapshot identity does not match the pinned recipe")
    record.update(repo_id=manifest["hf_id"], revision=directory.name)
    return record
