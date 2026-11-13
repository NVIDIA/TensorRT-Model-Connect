# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import stat

import pytest

from trtmc_perf_serving.files import FILE_KEY, externalize_files, materialize_files, store_media


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference(data: bytes, suffix: str = ".png") -> dict:
    return {FILE_KEY: {"suffix": suffix, "sha256": digest(data)}}


def test_store_media_keeps_one_read_only_copy_named_by_its_digest(tmp_path):
    first = store_media(tmp_path, b"image", ".png")
    again = store_media(tmp_path, b"image", ".png")

    assert first == again == reference(b"image")
    stored = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert [path.name for path in stored] == [f"{digest(b'image')}.png"]
    assert stored[0].parent.name == digest(b"image")[:2]
    assert stored[0].read_bytes() == b"image"
    assert not stored[0].stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_store_media_replaces_a_truncated_copy(tmp_path):
    store_media(tmp_path, b"image", ".png")
    path = next(path for path in tmp_path.rglob("*.png"))
    path.chmod(0o644)
    path.write_bytes(b"im")

    store_media(tmp_path, b"image", ".png")

    assert path.read_bytes() == b"image"


def test_externalize_turns_paths_and_inline_bytes_into_digest_references(tmp_path):
    store = tmp_path / "store"
    image = tmp_path / "cat.jpeg"
    image.write_bytes(b"cat")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")
    inline = {FILE_KEY: {"suffix": ".wav", "b64": base64.b64encode(b"speech").decode()}}
    request = {"image_path": str(image), "frame_paths": [str(frame)], "audio_path": inline, "prompt": str(image),
               "mask_path": "/does/not/exist.png", "top_k": 5}

    result = externalize_files(request, store)

    assert result == {"image_path": reference(b"cat", ".jpeg"), "frame_paths": [reference(b"frame")],
                      "audio_path": reference(b"speech", ".wav"), "prompt": str(image),
                      "mask_path": "/does/not/exist.png", "top_k": 5}
    assert len([path for path in store.rglob("*") if path.is_file()]) == 3
    assert "b64" not in str(result)


def test_equal_content_has_one_reference_wherever_it_came_from(tmp_path):
    left, right = tmp_path / "a" / "x.png", tmp_path / "b" / "y.png"
    for path in (left, right):
        path.parent.mkdir()
        path.write_bytes(b"same pixels")

    one = externalize_files({"image_path": str(left)}, tmp_path / "store")
    other = externalize_files({"image_path": {FILE_KEY: {"suffix": ".png", "b64": base64.b64encode(b"same pixels").decode()}}},
                              tmp_path / "store")

    assert one == other
    assert len([path for path in (tmp_path / "store").rglob("*") if path.is_file()]) == 1


def test_externalize_keeps_existing_references_and_text_requests_need_no_store():
    ready = {"image_path": reference(b"img")}

    assert externalize_files(ready, None) == ready
    assert externalize_files({"prompt": "a cat", "top_k": 5}, None) == {"prompt": "a cat", "top_k": 5}


def test_externalize_without_a_store_rejects_media(tmp_path):
    image = tmp_path / "cat.png"
    image.write_bytes(b"cat")

    with pytest.raises(ValueError, match="media store"):
        externalize_files({"image_path": str(image)}, None)


def test_materialize_writes_a_per_request_copy_of_a_stored_file(tmp_path):
    store = tmp_path / "store"
    request = externalize_files({"image_path": {FILE_KEY: {"suffix": ".png", "b64": base64.b64encode(b"img").decode()}}},
                                store)

    resolved = materialize_files(request, tmp_path / "request", media_root=store)

    copy = tmp_path / "request" / "input-0000.png"
    assert resolved == {"image_path": str(copy)}
    assert copy.read_bytes() == b"img"
    stored = next(path for path in store.rglob("*.png"))
    assert copy != stored and not copy.samefile(stored)  # a backend may write beside its input: the store stays intact


def test_inline_bytes_still_materialize_without_a_store(tmp_path):
    request = {"audio_path": {FILE_KEY: {"suffix": ".wav", "b64": base64.b64encode(b"wav").decode()}}}

    resolved = materialize_files(request, tmp_path / "request")

    assert (tmp_path / "request" / "input-0000.wav").read_bytes() == b"wav"
    assert resolved == {"audio_path": str(tmp_path / "request" / "input-0000.wav")}


@pytest.mark.parametrize("name, body", [
    ("no store", None),
    ("unknown content", {"suffix": ".png", "sha256": "0" * 64}),
    ("uppercase digest", {"suffix": ".png", "sha256": digest(b"img").upper()}),
    ("short digest", {"suffix": ".png", "sha256": digest(b"img")[:32]}),
    ("digest naming a path", {"suffix": ".png", "sha256": "../" * 21 + "x"}),
    ("suffix naming a path", {"suffix": "/../x", "sha256": digest(b"img")}),
    ("both bytes and digest", {"suffix": ".png", "sha256": digest(b"img"), "b64": "aW1n"}),
    ("neither", {"suffix": ".png"}),
])
def test_materialize_rejects_unusable_references(tmp_path, name, body):
    store = tmp_path / "store"
    store_media(store, b"img", ".png")
    request = {"image_path": reference(b"img") if body is None else {FILE_KEY: body}}

    with pytest.raises(ValueError):
        materialize_files(request, tmp_path / "request", media_root=None if body is None else store)


def test_materialize_rejects_a_stored_file_that_no_longer_matches_its_digest(tmp_path):
    store = tmp_path / "store"
    request = {"image_path": store_media(store, b"img", ".png")}
    path = next(path for path in store.rglob("*.png"))
    path.chmod(0o644)
    path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="digest"):
        materialize_files(request, tmp_path / "request", media_root=store)


def test_a_reference_cannot_carry_sibling_fields(tmp_path):
    store = tmp_path / "store"
    request = {"image_path": {**store_media(store, b"img", ".png"), "extra": 1}}

    with pytest.raises(ValueError, match="sibling"):
        materialize_files(request, tmp_path / "request", media_root=store)
