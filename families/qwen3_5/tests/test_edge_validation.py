# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression checks for the family real-Edge validation harness."""

import copy
from pathlib import Path

import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter
from families.qwen3_5.tests import edge_validation as validation


@pytest.fixture
def reports():
    cases, _ = validation.requests(1024)
    public = {
        "backend": "edge_llm",
        "edge_revision": validation.EDGE_REVISION,
        "passed": True,
        "results": [],
    }
    direct = {"backend": "direct_edge", "passed": True, "results": []}
    for case in cases:
        if case.get("expect_error"):
            public["results"].append({"id": case["id"], "error": "unsupported control"})
        else:
            item = {"id": case["id"], "token_ids": [7], "text": "answer"}
            public["results"].append(item)
            direct["results"].append({**copy.deepcopy(item), "finish_reason": 1})
    return public, direct, cases


def test_accepts_complete_independent_proof(reports):
    validation.compare(*reports)


@pytest.mark.parametrize(
    "failure",
    [
        "native",
        "wrong_revision",
        "failed_driver",
        "missing_case",
        "swallowed_error",
        "empty_tokens",
        "noninteger_tokens",
        "too_many_tokens",
        "wrong_tokens",
        "wrong_text",
        "eos_exhausted",
        "repeat_changed",
    ],
)
def test_rejects_invalid_proof(reports, failure):
    public, direct, cases = reports
    if failure == "native":
        public["backend"] = "native"
    elif failure == "wrong_revision":
        public["edge_revision"] = "wrong"
    elif failure == "failed_driver":
        direct["passed"] = False
    elif failure == "missing_case":
        direct["results"].pop()
    elif failure == "swallowed_error":
        public["results"][3].pop("error")
    elif failure == "empty_tokens":
        public["results"][0]["token_ids"] = []
    elif failure == "noninteger_tokens":
        public["results"][0]["token_ids"] = [True]
    elif failure == "too_many_tokens":
        public["results"][0]["token_ids"] *= 9
    elif failure == "wrong_tokens":
        direct["results"][0]["token_ids"] = [9]
    elif failure == "wrong_text":
        direct["results"][0]["text"] = "different"
    elif failure == "eos_exhausted":
        direct["results"][2]["finish_reason"] = 2
    elif failure == "repeat_changed":
        public["results"][-1]["token_ids"] = [9]
        direct["results"][-1]["token_ids"] = [9]
    with pytest.raises(AssertionError):
        validation.compare(public, direct, cases)


def make_bundle(path, artifact, *, revision=validation.EDGE_REVISION, marker=True):
    """Write a structurally valid generic bundle with a configurable family marker."""
    writer = BundleWriter(path)
    writer.set_header(family="qwen3_5", task="text_generation", backend="trt")
    payload = b"checkpoint data" * 20000
    writer.add_bytes(artifact, payload)
    if marker:
        writer.add_json(
            "edge_llm.json", {"version": 1, "edge_revision": revision, "artifacts": [artifact]}
        )
    writer.finish()
    return payload


def test_streams_large_artifacts_without_unbounded_reads(tmp_path, monkeypatch):
    bundle = tmp_path / "model.bundle"
    artifact = "edge_llm/checkpoint/model.safetensors"
    expected = make_bundle(bundle, artifact)
    original_open = Path.open

    class BoundedReader:
        """Fail if extraction attempts to materialize the complete checkpoint shard."""

        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def seek(self, offset):
            return self.source.seek(offset)

        def read(self, size=-1):
            assert 0 <= size <= 64 * 1024
            return self.source.read(size)

    def open_file(path, *args, **kwargs):
        source = original_open(path, *args, **kwargs)
        return BoundedReader(source) if path == bundle else source

    monkeypatch.setattr(Path, "open", open_file)
    output = tmp_path / "extracted"
    validation.extract_bundle(bundle, output)
    assert (output / artifact).read_bytes() == expected


@pytest.mark.parametrize(
    "artifact",
    [
        "../outside",
        "edge_llm/engine/../../outside",
        "/tmp/outside",
        "edge_llm/engine/./file",
        "edge_llm/engine/file\\path",
    ],
)
def test_rejects_unsafe_artifact_paths(tmp_path, artifact):
    bundle = tmp_path / "model.bundle"
    make_bundle(bundle, artifact)
    with pytest.raises(ValueError, match="Unsafe Edge artifact"):
        validation.extract_bundle(bundle, tmp_path / "extracted")


@pytest.mark.parametrize("options", [{"revision": "different"}, {"marker": False}])
def test_rejects_wrong_revision_or_native_bundle(tmp_path, options):
    bundle = tmp_path / "model.bundle"
    make_bundle(bundle, "edge_llm/engine/llm.engine", **options)
    with pytest.raises(ValueError):
        validation.extract_bundle(bundle, tmp_path / "extracted")
