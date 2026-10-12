# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")  # the serving extra (apps/perf_serving/requirements.txt)
Image = pytest.importorskip("PIL.Image")

from trtmc_perf_serving.digests import add_media_digests  # noqa: E402


def test_worker_pngs_and_reference_arrays_give_the_same_image_digest(tmp_path):
    frame = np.random.default_rng(0).integers(0, 256, size=(48, 80, 3), dtype=np.uint8)
    worker, reference = tmp_path / "worker", tmp_path / "reference"
    worker.mkdir()
    reference.mkdir()
    Image.fromarray(frame).save(worker / "output.1.0.png")
    Image.fromarray(frame).save(worker / "output.input.1.0.png")  # conditioning inputs are ignored
    np.save(reference / "output.npy", frame[None].astype(np.float32) / 255.0)
    left = add_media_digests({}, "generate_image", worker)["media_digest"]
    right = add_media_digests({}, "generate_image", reference)["media_digest"]
    assert (left["frames"], left["height"], left["width"]) == (1, 48, 80)
    assert max(abs(a - b) for a, b in zip(left["thumbnails"][0], right["thumbnails"][0])) <= 1


def test_audio_digest_from_wav_or_array_and_other_operations_untouched(tmp_path):
    tone = np.sin(np.arange(24000) / 24000 * 2 * np.pi * 220).astype(np.float32) * 0.5
    soundfile.write(tmp_path / "output.1.wav", tone, 24000, subtype="FLOAT")
    digest = add_media_digests({"sample_rate": 24000}, "generate_audio", tmp_path)["audio_digest"]
    assert abs(digest["seconds"] - 1.0) < 1e-6 and abs(digest["rms"] - 0.5 / 2 ** 0.5) < 1e-3
    assert add_media_digests({"text": "x"}, "generate", tmp_path) == {"text": "x"}


def test_media_frames_reads_worker_pngs_in_frame_order_or_the_reference_array(tmp_path):
    from trtmc_perf_serving.digests import media_frames

    worker = tmp_path / "worker"
    worker.mkdir()
    for index in (10, 2):
        Image.fromarray(np.full((4, 6, 3), index, dtype=np.uint8)).save(worker / f"output.{index}.png")
    assert [int(frame[0, 0, 0]) for frame in media_frames(worker)] == [2, 10]
    np.save(tmp_path / "output.npy", np.ones((1, 3, 4, 6, 3), dtype=np.float32))
    frames = media_frames(tmp_path)
    assert len(frames) == 3 and frames[0].dtype == np.uint8 and frames[0].shape == (4, 6, 3)
    assert media_frames(tmp_path / "worker" / "missing") == []
