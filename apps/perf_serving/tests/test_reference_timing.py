# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pytest

pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from trtmc_perf_serving.backends.reference import ReferenceBackend  # noqa: E402
from trtmc_perf_serving.backends.reference import common  # noqa: E402


class Adapter:
    def __init__(self):
        self.images = []

    def invoke(self, request, artifact_base):
        if "image_path" in request:
            self.images.append(common.load_image(request["image_path"]))
        return common.invocation(common.tensor_observation(np.arange(4.0), artifact_base), 0.0)


def backend(adapter):
    instance = object.__new__(ReferenceBackend)
    instance._adapter = adapter
    return instance


def test_evidence_writes_and_input_decoding_stay_outside_the_task_call_time(tmp_path, monkeypatch):
    save = np.save
    monkeypatch.setattr(common.np, "save", lambda *args, **kwargs: (time.sleep(0.1), save(*args, **kwargs)))
    read = common._read_image
    reads = []
    monkeypatch.setattr(common, "_read_image", lambda path: (reads.append(path), time.sleep(0.1), read(path))[2])
    image = tmp_path / "input.png"
    Image.new("RGB", (8, 8)).save(image)
    adapter = Adapter()

    invocation = backend(adapter).invoke({"image_path": str(image)}, tmp_path / "out" / "output")

    assert invocation.model_call_ms < 50  # neither the 100 ms write nor the 100 ms decode is timed
    assert reads == [str(image)] and adapter.images[0].size == (8, 8)  # decoded once, ahead of the call
    observation = invocation.observation
    assert np.array_equal(np.load(observation["artifact"]), np.arange(4.0)) and len(observation["sha256"]) == 64
    assert "_pending_array" not in observation and common._DECODED == {}
