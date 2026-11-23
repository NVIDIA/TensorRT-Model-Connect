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
    instance._adapter, instance._spec = adapter, common.ReferenceSpec(operation="classify", model="m")
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


def test_json_array_encoding_stays_outside_the_task_call_time(tmp_path, monkeypatch):
    class Array:
        def reshape(self, *args):
            return self

        def tolist(self):
            time.sleep(0.1)
            return [1.0, 2.0]

    class ValuesAdapter:
        def invoke(self, request, artifact_base):
            return common.invocation(common.deferred_values({"shape": [2]}, {"values": Array()}), 0.0)

    asarray = common.np.asarray
    monkeypatch.setattr(common.np, "asarray", lambda value: value if isinstance(value, Array) else asarray(value))
    monkeypatch.setattr(common, "synchronize", lambda: None)
    result = backend(ValuesAdapter()).invoke({}, tmp_path / "values")
    assert result.model_call_ms < 50
    assert result.observation == {"shape": [2], "values": [1.0, 2.0]}


class FakeCuda:
    def __init__(self, *, integrated, reserved_gib, allocated_gib=10, total_gib=120):
        self.integrated, self.total = integrated, total_gib << 30
        self.reserved, self.allocated, self.emptied = reserved_gib << 30, allocated_gib << 30, 0

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def get_device_properties(self, device):
        return type("Properties", (), {"is_integrated": self.integrated, "total_memory": self.total})()

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def empty_cache(self):
        self.emptied += 1


@pytest.mark.parametrize(("integrated", "reserved_gib", "emptied"), [
    (True, 80, 1),   # 70 GiB unused > a quarter of 120 GiB: given back
    (True, 30, 0),   # 20 GiB unused: kept for the next requests
    (False, 80, 0),  # a discrete GPU keeps its cache
])
def test_a_shared_memory_gpu_gives_back_a_cache_larger_than_a_quarter_of_its_memory(
        monkeypatch, integrated, reserved_gib, emptied):
    fake = FakeCuda(integrated=integrated, reserved_gib=reserved_gib)
    monkeypatch.setattr(common.torch, "cuda", fake)
    common.release_inputs()
    assert fake.emptied == emptied
